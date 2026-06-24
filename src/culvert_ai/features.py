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
    negative_culverts: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | None = None,
    streams: gpd.GeoDataFrame | None = None,
    dem_path: str | Path | None = None,
    flow_accumulation_path: str | Path | None = None,
    drainage_area_path: str | Path | None = None,
    landcover_path: str | Path | None = None,
    positive_radius_m: float = 10.0,
    negative_radius_m: float = 10.0,
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

    if negative_culverts is not None:
        negative_m = clean_geometry(negative_culverts).to_crs(metric_crs)
        features = add_negative_culvert_labels(features, negative_m, negative_radius_m)

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
        features = add_dem_hydrology_proxies(features)

    if flow_accumulation_path:
        features = add_raster_samples(features, flow_accumulation_path, prefix="flow_accumulation")
        features = add_hydrology_raster_features(features, "flow_accumulation")

    if drainage_area_path:
        features = add_raster_samples(features, drainage_area_path, prefix="drainage_area")
        features = add_hydrology_raster_features(features, "drainage_area")

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
        features["road_stream_proximity_signal"] = (1.0 / (1.0 + distances / 20.0)).fillna(0.0)

    if "crossing_angle_degrees" in features.columns:
        angle = pd.to_numeric(features["crossing_angle_degrees"], errors="coerce")
        features["crossing_angle_abs_from_90"] = (90 - angle).abs()
        features["crossing_angle_perpendicularity"] = (
            1.0 - (features["crossing_angle_abs_from_90"] / 90.0)
        ).clip(0, 1)

    if {"road_stream_proximity_signal", "crossing_angle_perpendicularity"}.issubset(features.columns):
        features["crossing_geometry_signal"] = (
            0.65 * features["road_stream_proximity_signal"].fillna(0.0)
            + 0.35 * features["crossing_angle_perpendicularity"].fillna(0.0)
        ).clip(0, 1)

    if "source" in features.columns:
        source = features["source"].fillna("").astype(str).str.lower()
        features["source_exact_intersection"] = source.eq("exact_road_stream_intersection").astype(int)
        features["source_nearest_approach"] = source.eq("nearest_road_stream_approach").astype(int)
        features["source_route_interval_sample"] = source.eq("route_interval_sample").astype(int)
        features["source_field_report_observed"] = source.eq("field_report_observed_culvert").astype(int)

    for column in ("road_name", "stream_name", "matched_route"):
        if column in features.columns:
            text = features[column].fillna("").astype(str).str.strip()
            features[f"has_{column}"] = (text != "").astype(int)

    for column in ("road_bridge", "road_tunnel", "stream_culvert", "stream_tunnel"):
        if column in features.columns:
            features[f"{column}_flag"] = _boolean_score(features[column])

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


def add_negative_culvert_labels(
    candidates: gpd.GeoDataFrame,
    negative_culverts: gpd.GeoDataFrame,
    negative_radius_m: float,
) -> gpd.GeoDataFrame:
    labeled = candidates.copy()
    labeled["field_denied"] = 0
    labeled["dist_to_denied_culvert_m"] = np.nan
    labeled["nearest_denied_observation_id"] = ""
    labeled["nearest_denied_notes"] = ""

    if negative_culverts.empty:
        return labeled

    if "is_culvert" not in labeled.columns:
        labeled["is_culvert"] = 0

    negative_reset = negative_culverts.reset_index(drop=True)
    negative_by_candidate_id = {}
    if "candidate_id" in negative_reset.columns:
        for negative_index, candidate_id in negative_reset["candidate_id"].fillna("").astype(str).items():
            if candidate_id:
                negative_by_candidate_id[candidate_id] = negative_reset.iloc[int(negative_index)]

    for row_index, geometry in labeled.geometry.items():
        distances = negative_reset.geometry.distance(geometry)
        nearest_index = int(distances.idxmin())
        distance = float(distances.iloc[nearest_index])
        nearest = negative_reset.iloc[nearest_index]
        labeled.at[row_index, "dist_to_denied_culvert_m"] = distance
        candidate_id = str(labeled.at[row_index, "candidate_id"]) if "candidate_id" in labeled else ""
        exact_negative = negative_by_candidate_id.get(candidate_id)
        if exact_negative is not None:
            nearest = exact_negative
            miss_distance_m = _optional_float(nearest.get("miss_distance_m"))
            if miss_distance_m is not None:
                labeled.at[row_index, "dist_to_denied_culvert_m"] = miss_distance_m
            mark_negative = True
        else:
            mark_negative = distance <= negative_radius_m and not _is_missed_prediction(nearest)

        if mark_negative:
            _mark_negative_label(labeled, row_index, nearest)

    return labeled


def _mark_negative_label(labeled: gpd.GeoDataFrame, row_index, nearest: pd.Series) -> None:
    labeled.at[row_index, "field_denied"] = 1
    labeled.at[row_index, "is_culvert"] = 0
    if "observation_id" in nearest.index and pd.notna(nearest["observation_id"]):
        labeled.at[row_index, "nearest_denied_observation_id"] = str(nearest["observation_id"])
    if "notes" in nearest.index and pd.notna(nearest["notes"]):
        labeled.at[row_index, "nearest_denied_notes"] = str(nearest["notes"])


def _is_missed_prediction(row: pd.Series) -> bool:
    return str(row.get("label", "") or "").strip() == "missed_prediction"


def _optional_float(value) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


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
            31: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
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


def add_dem_hydrology_proxies(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    enriched = points.copy()
    if "slope_degrees" not in enriched.columns:
        return enriched

    slope = pd.to_numeric(enriched["slope_degrees"], errors="coerce").clip(lower=0)
    slope_damping = 1.0 / (1.0 + slope)
    for window_size in (3, 9, 15, 31):
        relief_col = f"elevation_relief_{window_size}x{window_size}_m"
        roughness_col = f"terrain_roughness_{window_size}x{window_size}_m"
        tpi_col = f"topographic_position_{window_size}x{window_size}_m"
        valley_col = f"valley_depth_{window_size}x{window_size}_m"
        if not {relief_col, roughness_col, tpi_col, valley_col}.issubset(enriched.columns):
            continue

        relief = pd.to_numeric(enriched[relief_col], errors="coerce").clip(lower=0)
        roughness = pd.to_numeric(enriched[roughness_col], errors="coerce").clip(lower=0)
        tpi = pd.to_numeric(enriched[tpi_col], errors="coerce")
        valley_depth = pd.to_numeric(enriched[valley_col], errors="coerce").clip(lower=0)

        enriched[f"valley_depth_relief_ratio_{window_size}x{window_size}"] = (
            valley_depth / relief.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        enriched[f"topographic_wetness_proxy_{window_size}x{window_size}"] = (
            np.log1p(valley_depth) * slope_damping
        )
        enriched[f"low_slope_valley_score_{window_size}x{window_size}"] = (
            _robust_0_to_1(valley_depth) * slope_damping
        )
        enriched[f"terrain_break_score_proxy_{window_size}x{window_size}"] = (
            np.log1p(relief) * np.log1p(roughness)
        )
        enriched[f"negative_tpi_{window_size}x{window_size}_m"] = (-tpi).clip(lower=0)

    return enriched


def add_hydrology_raster_features(points: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
    enriched = points.copy()
    value_col = f"{prefix}_value"
    if value_col not in enriched.columns:
        return enriched

    values = pd.to_numeric(enriched[value_col], errors="coerce").clip(lower=0)
    enriched[f"{prefix}_log"] = np.log1p(values)
    enriched[f"{prefix}_rank_pct"] = values.rank(pct=True).fillna(0.0)
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


def _boolean_score(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .isin({"1", "true", "yes", "y", "bridge", "tunnel", "culvert", "covered"})
        .astype(int)
    )


def _robust_0_to_1(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    low = numeric.quantile(0.05)
    high = numeric.quantile(0.95)
    if pd.isna(low) or pd.isna(high) or high <= low:
        return pd.Series(0.0, index=values.index)
    return ((numeric - low) / (high - low)).clip(0, 1).fillna(0.0)


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
