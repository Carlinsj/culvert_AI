"""Label-free physical road/drainage crossing features for research.

This module is deliberately separate from the frozen production feature path.
It uses only road centerlines, mapped/DEM drainage geometry, and DEM-derived
rasters.  No inventory, label, route identity, or fitted-model output is read.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry


METRIC_CRS = "EPSG:32618"
WINDOWS_M = (20.0, 40.0, 60.0)
THRESHOLD_KEYS = ("0p1km2", "0p5km2", "2km2", "10km2")
THRESHOLD_AREAS_KM2 = (0.1, 0.5, 2.0, 10.0)


FEATURE_COLUMNS = [
    "crossing_geometry_available",
    "road_drainage_crossing_angle_degrees",
    "parallel_drainage_alignment",
    "bilateral_drainage_20m",
    "bilateral_drainage_40m",
    "bilateral_drainage_60m",
    "upstream_downstream_area_ratio_20m",
    "upstream_downstream_log_area_difference_20m",
    "nearest_actual_crossing_distance_m",
    "mapped_dem_crossing_distance_gap_m",
    "road_low_point_depth_30m",
    "road_low_point_depth_60m",
    "road_elevation_percentile_60m",
    "road_neighbor_depth_60m",
    "terrain_gradient_parallel_20m",
    "terrain_gradient_perpendicular_20m",
    "terrain_gradient_perp_minus_parallel_20m",
    "terrain_gradient_perp_parallel_ratio_20m",
    "mapped_dem_both_crossing_20m",
    "mapped_crossing_only_20m",
    "dem_crossing_only_20m",
    "neither_crossing_20m",
    "dem_crossing_threshold_count_20m",
    "dem_crossing_threshold_count_40m",
    "dem_crossing_threshold_count_60m",
    "dem_crossing_max_threshold_km2_40m",
]


def acute_angle_degrees(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float:
    """Return the undirected acute angle between two non-zero vectors."""

    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 0:
        return np.nan
    cosine = float(np.clip(abs(np.dot(a, b)) / norm, 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def local_line_tangent(
    line: BaseGeometry,
    point: Point,
    half_window_m: float = 15.0,
) -> tuple[Point, tuple[float, float]]:
    """Project a point to a line and estimate a unit tangent around it."""

    if line is None or line.is_empty or not isinstance(line, LineString):
        return point, (np.nan, np.nan)
    position = float(line.project(point))
    origin = line.interpolate(position)
    left = line.interpolate(max(0.0, position - half_window_m))
    right = line.interpolate(min(float(line.length), position + half_window_m))
    dx = float(right.x - left.x)
    dy = float(right.y - left.y)
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return origin, (np.nan, np.nan)
    return origin, (dx / norm, dy / norm)


def _coordinates(geometry: BaseGeometry) -> Iterable[tuple[float, float]]:
    if geometry is None or geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _coordinates(part)
        return
    if hasattr(geometry, "coords"):
        yield from geometry.coords


def bilateral_continuity(
    drainage: BaseGeometry,
    road_origin: Point,
    road_tangent: tuple[float, float],
    window_m: float,
    side_tolerance_m: float = 2.0,
) -> float:
    """Whether drainage occupies both signed sides of the local road line."""

    if drainage is None or drainage.is_empty:
        return 0.0
    tx, ty = road_tangent
    if not np.isfinite([tx, ty]).all():
        return np.nan
    local = drainage.intersection(road_origin.buffer(window_m))
    signed = [
        (-ty) * (x - road_origin.x) + tx * (y - road_origin.y)
        for x, y in _coordinates(local)
    ]
    if not signed:
        return 0.0
    return float(min(signed) < -side_tolerance_m and max(signed) > side_tolerance_m)


def _sample(dataset: rasterio.io.DatasetReader, coordinates: list[tuple[float, float]]) -> np.ndarray:
    values = np.asarray([row[0] for row in dataset.sample(coordinates)], dtype=float)
    nodata = dataset.nodata
    if nodata is not None:
        values[np.isclose(values, nodata)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def _road_geometry_lookup(candidates: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> list[BaseGeometry | None]:
    road_map = dict(zip(roads["id"].astype(str), roads.geometry, strict=False))
    result = [road_map.get(str(value)) for value in candidates["road_id"]]
    missing = [position for position, geometry in enumerate(result) if geometry is None]
    if missing:
        nearest = gpd.sjoin_nearest(
            candidates.iloc[missing][["geometry"]],
            roads[["geometry"]],
            how="left",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        for position in missing:
            if position in nearest.index and pd.notna(nearest.at[position, "index_right"]):
                result[position] = roads.geometry.iloc[int(nearest.at[position, "index_right"])]
    return result


def _nearest_drainage(candidates: gpd.GeoDataFrame, drainage: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin_nearest(
        candidates[["candidate_id", "geometry"]],
        drainage[["geometry"]],
        how="left",
        max_distance=120.0,
        distance_col="nearest_drainage_distance_m",
    )
    joined = joined.sort_values(["candidate_id", "nearest_drainage_distance_m", "index_right"])
    return joined.drop_duplicates("candidate_id").set_index("candidate_id")


def build_crossing_geometry_features(
    candidates: gpd.GeoDataFrame,
    roads_path: str | Path,
    hydrology_dir: str | Path,
) -> pd.DataFrame:
    """Build the one predeclared crossing-geometry feature family.

    ``candidates`` must already contain the frozen FLOW-C physical crossing
    distances.  Output order and candidate IDs are preserved exactly.
    """

    required = {
        "candidate_id",
        "road_id",
        "geometry",
        "distance_to_nearest_mapped_crossing_m",
        *{
            f"distance_to_dem_road_drainage_crossing_{key}_m"
            for key in THRESHOLD_KEYS
        },
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Crossing feature inputs are missing: {missing}")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs must be unique for crossing features.")

    hydro = Path(hydrology_dir)
    points = candidates.to_crs(METRIC_CRS).copy().reset_index(drop=True)
    points["candidate_id"] = points["candidate_id"].astype(str)
    roads = gpd.read_file(roads_path).to_crs(METRIC_CRS)
    drainage = gpd.read_parquet(hydro / "drainage_0p1km2.parquet").to_crs(METRIC_CRS)

    road_lines = _road_geometry_lookup(points, roads)
    road_origins: list[Point] = []
    road_tangents: list[tuple[float, float]] = []
    for point, line in zip(points.geometry, road_lines, strict=False):
        origin, tangent = local_line_tangent(line, point)
        road_origins.append(origin)
        road_tangents.append(tangent)

    nearest = _nearest_drainage(points, drainage)
    drainage_lines: list[BaseGeometry | None] = []
    drainage_origins: list[Point | None] = []
    drainage_tangents: list[tuple[float, float]] = []
    drainage_distances: list[float] = []
    for candidate_id, point in zip(points["candidate_id"], points.geometry, strict=False):
        if candidate_id not in nearest.index or pd.isna(nearest.at[candidate_id, "index_right"]):
            drainage_lines.append(None)
            drainage_origins.append(None)
            drainage_tangents.append((np.nan, np.nan))
            drainage_distances.append(np.nan)
            continue
        row = nearest.loc[candidate_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        line = drainage.geometry.iloc[int(row["index_right"])]
        origin, tangent = local_line_tangent(line, point)
        drainage_lines.append(line)
        drainage_origins.append(origin)
        drainage_tangents.append(tangent)
        drainage_distances.append(float(row["nearest_drainage_distance_m"]))

    output = pd.DataFrame({"candidate_id": points["candidate_id"]})
    angles = [
        acute_angle_degrees(road, drain)
        for road, drain in zip(road_tangents, drainage_tangents, strict=False)
    ]
    available = np.isfinite(angles) & (np.asarray(drainage_distances) <= 60.0)
    angles = np.where(available, angles, np.nan)
    output["crossing_geometry_available"] = available.astype(int)
    output["road_drainage_crossing_angle_degrees"] = angles
    output["parallel_drainage_alignment"] = np.where(available, 1.0 - angles / 90.0, np.nan)
    for window in WINDOWS_M:
        output[f"bilateral_drainage_{int(window)}m"] = [
            bilateral_continuity(line, origin, tangent, window)
            if line is not None and distance <= window
            else 0.0
            for line, origin, tangent, distance in zip(
                drainage_lines, road_origins, road_tangents, drainage_distances, strict=False
            )
        ]

    dem_coordinates: list[tuple[float, float]] = []
    road_sample_positions: list[list[tuple[float, float]]] = []
    for origin, tangent in zip(road_origins, road_tangents, strict=False):
        tx, ty = tangent
        if not np.isfinite([tx, ty]).all():
            coords = [(origin.x, origin.y)] * 9
        else:
            nx, ny = -ty, tx
            coords = [
                (origin.x - 60 * tx, origin.y - 60 * ty),
                (origin.x - 30 * tx, origin.y - 30 * ty),
                (origin.x, origin.y),
                (origin.x + 30 * tx, origin.y + 30 * ty),
                (origin.x + 60 * tx, origin.y + 60 * ty),
                (origin.x - 20 * tx, origin.y - 20 * ty),
                (origin.x + 20 * tx, origin.y + 20 * ty),
                (origin.x - 20 * nx, origin.y - 20 * ny),
                (origin.x + 20 * nx, origin.y + 20 * ny),
            ]
        road_sample_positions.append(coords)
        dem_coordinates.extend(coords)
    with rasterio.open(hydro / "dem_conditioned.tif") as dataset:
        dem_values = _sample(dataset, dem_coordinates).reshape(len(points), 9)

    center = dem_values[:, 2]
    output["road_low_point_depth_30m"] = np.nanmedian(dem_values[:, [1, 3]], axis=1) - center
    output["road_low_point_depth_60m"] = np.nanmedian(dem_values[:, [0, 1, 3, 4]], axis=1) - center
    output["road_elevation_percentile_60m"] = np.nanmean(dem_values[:, :5] <= center[:, None], axis=1)
    output["road_neighbor_depth_60m"] = np.nanmin(dem_values[:, [0, 4]], axis=1) - center
    parallel = np.abs(dem_values[:, 6] - dem_values[:, 5]) / 40.0
    perpendicular = np.abs(dem_values[:, 8] - dem_values[:, 7]) / 40.0
    output["terrain_gradient_parallel_20m"] = parallel
    output["terrain_gradient_perpendicular_20m"] = perpendicular
    output["terrain_gradient_perp_minus_parallel_20m"] = perpendicular - parallel
    output["terrain_gradient_perp_parallel_ratio_20m"] = perpendicular / (parallel + 1e-3)

    flow_coordinates: list[tuple[float, float]] = []
    flow_valid = []
    for origin, tangent, distance in zip(
        drainage_origins, drainage_tangents, drainage_distances, strict=False
    ):
        valid = origin is not None and distance <= 60.0 and np.isfinite(tangent).all()
        flow_valid.append(valid)
        if valid:
            dx, dy = tangent
            flow_coordinates.extend(
                [(origin.x - 20 * dx, origin.y - 20 * dy), (origin.x + 20 * dx, origin.y + 20 * dy)]
            )
        else:
            flow_coordinates.extend([(0.0, 0.0), (0.0, 0.0)])
    with rasterio.open(hydro / "upstream_area_m2.tif") as dataset:
        area = _sample(dataset, flow_coordinates).reshape(len(points), 2)
    area[~np.asarray(flow_valid), :] = np.nan
    valid_area = np.isfinite(area).any(axis=1)
    small = np.full(len(area), np.nan)
    large = np.full(len(area), np.nan)
    small[valid_area] = np.nanmin(area[valid_area], axis=1)
    large[valid_area] = np.nanmax(area[valid_area], axis=1)
    output["upstream_downstream_area_ratio_20m"] = small / np.where(large > 0, large, np.nan)
    output["upstream_downstream_log_area_difference_20m"] = np.abs(np.log1p(area[:, 1]) - np.log1p(area[:, 0]))

    mapped_distance = pd.to_numeric(points["distance_to_nearest_mapped_crossing_m"], errors="coerce")
    dem_distances = pd.DataFrame(
        {
            key: pd.to_numeric(
                points[f"distance_to_dem_road_drainage_crossing_{key}_m"], errors="coerce"
            )
            for key in THRESHOLD_KEYS
        }
    )
    minimum_dem = dem_distances.min(axis=1)
    output["nearest_actual_crossing_distance_m"] = pd.concat(
        [mapped_distance, minimum_dem], axis=1
    ).min(axis=1)
    output["mapped_dem_crossing_distance_gap_m"] = (mapped_distance - minimum_dem).abs()
    mapped20 = mapped_distance.le(20.0)
    dem20 = minimum_dem.le(20.0)
    output["mapped_dem_both_crossing_20m"] = (mapped20 & dem20).astype(int)
    output["mapped_crossing_only_20m"] = (mapped20 & ~dem20).astype(int)
    output["dem_crossing_only_20m"] = (~mapped20 & dem20).astype(int)
    output["neither_crossing_20m"] = (~mapped20 & ~dem20).astype(int)
    for window in WINDOWS_M:
        output[f"dem_crossing_threshold_count_{int(window)}m"] = dem_distances.le(window).sum(axis=1)
    output["dem_crossing_max_threshold_km2_40m"] = [
        max(
            [area for area, distance in zip(THRESHOLD_AREAS_KM2, row, strict=False) if distance <= 40.0],
            default=0.0,
        )
        for row in dem_distances.to_numpy(float)
    ]

    if output["candidate_id"].tolist() != points["candidate_id"].tolist():
        raise AssertionError("Crossing feature construction changed candidate order.")
    return output[["candidate_id", *FEATURE_COLUMNS]]

