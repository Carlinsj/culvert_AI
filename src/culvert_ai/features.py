from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.windows import Window

from culvert_ai.io import add_wgs84_coordinates, clean_geometry, project_layers_to_metric


def build_feature_table(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | None = None,
    streams: gpd.GeoDataFrame | None = None,
    dem_path: str | Path | None = None,
    landcover_path: str | Path | None = None,
    positive_radius_m: float = 30.0,
    density_radius_m: float = 75.0,
    density_radii_m: tuple[float, ...] | None = None,
) -> gpd.GeoDataFrame:
    (features,), metric_crs = project_layers_to_metric(clean_geometry(candidates))
    features = features.copy()
    features["x_m"] = features.geometry.x
    features["y_m"] = features.geometry.y
    features = add_candidate_derived_features(features)

    density_radii = _density_radii(density_radius_m, density_radii_m)

    if known_culverts is not None:
        known_m = clean_geometry(known_culverts).to_crs(metric_crs)
        features = add_known_culvert_labels(features, known_m, positive_radius_m)

    if roads is not None:
        roads_m = clean_geometry(roads).to_crs(metric_crs)
        features["distance_to_nearest_road_m"] = _nearest_distance(features.geometry, roads_m)
        for radius in density_radii:
            column = _density_column("road", radius)
            features[column] = _line_density(features.geometry, roads_m, radius)
        features["road_density_m_per_sqkm"] = features[_density_column("road", density_radius_m)]

    if streams is not None:
        streams_m = clean_geometry(streams).to_crs(metric_crs)
        features["distance_to_nearest_stream_m"] = _nearest_distance(features.geometry, streams_m)
        for radius in density_radii:
            column = _density_column("stream", radius)
            features[column] = _line_density(features.geometry, streams_m, radius)
        features["stream_density_m_per_sqkm"] = features[_density_column("stream", density_radius_m)]

    if dem_path:
        features = add_raster_samples(features, dem_path, prefix="dem")

    if landcover_path:
        features = add_raster_samples(features, landcover_path, prefix="landcover")

    features = add_wgs84_coordinates(features)
    return features.reset_index(drop=True)


def add_candidate_derived_features(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    features = candidates.copy()
    if "road_stream_distance_m" in features.columns:
        distances = pd.to_numeric(features["road_stream_distance_m"], errors="coerce").clip(lower=0)
        features["log_road_stream_distance_m"] = np.log1p(distances)
        features["is_exact_road_stream_intersection"] = (distances <= 0.01).astype(int)

    if "crossing_angle_degrees" in features.columns:
        angle = pd.to_numeric(features["crossing_angle_degrees"], errors="coerce")
        features["crossing_angle_abs_from_90"] = (90 - angle).abs()

    return features


def add_known_culvert_labels(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame,
    positive_radius_m: float,
) -> gpd.GeoDataFrame:
    labeled = candidates.copy()
    if known_culverts.empty:
        labeled["dist_to_known_culvert_m"] = np.nan
        labeled["is_culvert"] = 0
        return labeled

    known_union = known_culverts.geometry.unary_union
    labeled["dist_to_known_culvert_m"] = labeled.geometry.apply(
        lambda geom: float(geom.distance(known_union))
    )
    labeled["is_culvert"] = (labeled["dist_to_known_culvert_m"] <= positive_radius_m).astype(int)
    labeled = add_nearest_known_culvert_metadata(labeled, known_culverts)
    return labeled


def add_nearest_known_culvert_metadata(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    enriched = candidates.copy()
    metadata_columns = {
        "report_date": "nearest_field_report_date",
        "route": "nearest_field_report_route",
        "culvert_id": "nearest_field_report_culvert_id",
        "source_file": "nearest_field_report_source_file",
    }

    for output_column in metadata_columns.values():
        enriched[output_column] = ""

    if known_culverts.empty:
        return enriched

    known_reset = known_culverts.reset_index(drop=True)
    for row_index, geometry in enriched.geometry.items():
        distances = known_reset.geometry.distance(geometry)
        nearest_index = int(distances.idxmin())
        nearest = known_reset.iloc[nearest_index]
        for source_column, output_column in metadata_columns.items():
            if source_column in nearest.index and pd.notna(nearest[source_column]):
                enriched.at[row_index, output_column] = str(nearest[source_column])

    return enriched


def add_raster_samples(
    points: gpd.GeoDataFrame,
    raster_path: str | Path,
    prefix: str,
) -> gpd.GeoDataFrame:
    import rasterio

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    enriched = points.copy()
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster is missing a CRS: {raster_path}")

        sample_points = enriched.to_crs(src.crs)
        value_col = "elevation_m" if prefix == "dem" else f"{prefix}_value"
        slope_col = "slope_degrees" if prefix == "dem" else f"{prefix}_local_slope_degrees"
        values: list[float] = []
        stats_by_window = {
            3: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
            9: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
            15: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
        }

        for point in sample_points.geometry:
            value = _sample_value(src, point.x, point.y)
            values.append(value)
            for window_size, output in stats_by_window.items():
                stats = _local_raster_stats(src, point.x, point.y, window_size=window_size)
                output["slope"].append(stats["slope_degrees"])
                output["mean"].append(stats["mean"])
                output["relief"].append(stats["relief"])
                output["std"].append(stats["std"])
                output["tpi"].append(stats["topographic_position"])
                output["valley_depth"].append(stats["valley_depth"])

        enriched[value_col] = values
        if prefix == "dem":
            enriched[slope_col] = stats_by_window[3]["slope"]
            for window_size, output in stats_by_window.items():
                enriched[f"elevation_mean_{window_size}x{window_size}_m"] = output["mean"]
                enriched[f"elevation_relief_{window_size}x{window_size}_m"] = output["relief"]
                enriched[f"terrain_roughness_{window_size}x{window_size}_m"] = output["std"]
                enriched[f"topographic_position_{window_size}x{window_size}_m"] = output["tpi"]
                enriched[f"valley_depth_{window_size}x{window_size}_m"] = output["valley_depth"]

    return enriched


def _density_radii(base_radius_m: float, extra_radii_m: tuple[float, ...] | None) -> tuple[float, ...]:
    radii = {float(base_radius_m), 50.0, 100.0, 250.0}
    if extra_radii_m:
        radii.update(float(radius) for radius in extra_radii_m)
    return tuple(sorted(radius for radius in radii if radius > 0))


def _density_column(layer: str, radius_m: float) -> str:
    radius_label = int(radius_m) if float(radius_m).is_integer() else radius_m
    return f"{layer}_density_{radius_label}m_m_per_sqkm"


def _nearest_distance(points: Iterable, targets: gpd.GeoDataFrame) -> list[float]:
    if targets.empty:
        return [np.nan for _point in points]
    target_union = targets.geometry.unary_union
    return [float(point.distance(target_union)) for point in points]


def _line_density(points: Iterable, lines: gpd.GeoDataFrame, radius_m: float) -> list[float]:
    area_sqkm = np.pi * radius_m * radius_m / 1_000_000
    densities = []

    for point in points:
        buffer = point.buffer(radius_m)
        total_length_m = 0.0
        for position in _query_positions(lines, buffer):
            segment = lines.iloc[int(position)].geometry.intersection(buffer)
            if not segment.is_empty:
                total_length_m += float(segment.length)
        densities.append(total_length_m / area_sqkm if area_sqkm else 0.0)

    return densities


def _query_positions(gdf: gpd.GeoDataFrame, geometry) -> list[int]:
    try:
        return list(gdf.sindex.query(geometry, predicate="intersects"))
    except Exception:
        return list(range(len(gdf)))


def _sample_value(src, x: float, y: float) -> float:
    try:
        value = next(src.sample([(x, y)]))[0]
    except Exception:
        return np.nan

    if src.nodata is not None and value == src.nodata:
        return np.nan
    return float(value)


def _local_raster_stats(src, x: float, y: float, window_size: int = 3) -> dict[str, float]:
    try:
        row, col = src.index(x, y)
    except Exception:
        return _empty_raster_stats()

    half = window_size // 2
    if row < half or col < half or row >= src.height - half or col >= src.width - half:
        return _empty_raster_stats()

    window = Window(col - half, row - half, window_size, window_size)
    data = src.read(1, window=window, masked=True).astype(float)
    if data.shape != (window_size, window_size):
        return _empty_raster_stats()

    filled = data.filled(np.nan)
    if np.isnan(filled).any():
        return _empty_raster_stats()

    center = float(filled[half, half])
    mean = float(np.mean(filled))
    yres = abs(src.transform.e) or 1.0
    xres = abs(src.transform.a) or 1.0
    dz_dy, dz_dx = np.gradient(filled, yres, xres)
    rise_run = np.sqrt(dz_dx[half, half] ** 2 + dz_dy[half, half] ** 2)
    return {
        "slope_degrees": float(np.degrees(np.arctan(rise_run))),
        "mean": mean,
        "relief": float(np.max(filled) - np.min(filled)),
        "std": float(np.std(filled)),
        "topographic_position": center - mean,
        "valley_depth": max(0.0, mean - center),
    }


def _empty_raster_stats() -> dict[str, float]:
    return {
        "slope_degrees": np.nan,
        "mean": np.nan,
        "relief": np.nan,
        "std": np.nan,
        "topographic_position": np.nan,
        "valley_depth": np.nan,
    }
