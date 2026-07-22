from __future__ import annotations

import re
from dataclasses import dataclass
from math import atan2, cos, degrees, floor, isnan, radians, sin

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, Point
from shapely.ops import nearest_points

from culvert_ai.io import add_wgs84_coordinates, clean_geometry, project_layers_to_metric


@dataclass(frozen=True)
class CandidateSettings:
    snap_tolerance_m: float = 20.0
    min_spacing_m: float = 25.0
    road_id_column: str | None = None
    stream_id_column: str | None = None


def generate_candidates(
    roads: gpd.GeoDataFrame,
    streams: gpd.GeoDataFrame,
    settings: CandidateSettings | None = None,
) -> gpd.GeoDataFrame:
    settings = settings or CandidateSettings()
    roads = clean_geometry(roads)
    streams = clean_geometry(streams)
    (roads_m, streams_m), metric_crs = project_layers_to_metric(roads, streams)

    records: list[dict] = []
    for road_pos, road in roads_m.iterrows():
        road_geom = road.geometry
        search_area = road_geom.buffer(settings.snap_tolerance_m)

        for stream_pos in _query_positions(streams_m, search_area):
            stream = streams_m.iloc[int(stream_pos)]
            stream_geom = stream.geometry
            distance_m = float(road_geom.distance(stream_geom))
            if distance_m > settings.snap_tolerance_m:
                continue

            points = _intersection_points(road_geom.intersection(stream_geom))
            source = "exact_road_stream_intersection"
            if not points:
                road_point, _stream_point = nearest_points(road_geom, stream_geom)
                points = [road_point]
                source = "nearest_road_stream_approach"

            for point in points:
                records.append(
                    {
                        "candidate_id": f"cand_{len(records) + 1:06d}",
                        "road_id": _row_identifier(road, road_pos, settings.road_id_column, "road"),
                        "stream_id": _row_identifier(
                            stream, stream_pos, settings.stream_id_column, "stream"
                        ),
                        "road_name": _optional_name(road),
                        "stream_name": _optional_name(stream),
                        "road_speed_limit": _first_numeric(
                            road,
                            (
                                "speed_limit",
                                "SPEED_LIMIT",
                                "speed",
                                "SPEED",
                                "posted_speed",
                                "maxspeed",
                            ),
                        ),
                        "road_highway": _first_value(road, ("highway", "HIGHWAY", "road_class")),
                        "road_bridge": _truthy(_first_value(road, ("road_bridge", "bridge", "BRIDGE"))),
                        "road_tunnel": _truthy(_first_value(road, ("road_tunnel", "tunnel", "TUNNEL"))),
                        "stream_waterway": _first_value(stream, ("waterway", "WATERWAY")),
                        "stream_tunnel": _first_value(stream, ("stream_tunnel", "tunnel", "TUNNEL")),
                        "stream_culvert": _truthy(
                            _first_value(stream, ("stream_culvert", "culvert", "CULVERT"))
                        ),
                        "stream_order": _first_numeric(
                            stream,
                            ("stream_order", "STREAM_ORDER", "order", "ORDER", "strahler"),
                        ),
                        "source": source,
                        "road_stream_distance_m": distance_m,
                        "crossing_angle_degrees": _crossing_angle_degrees(
                            road_geom,
                            stream_geom,
                            point,
                        ),
                        "geometry": point,
                    }
                )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "candidate_id",
                "road_id",
                "stream_id",
                "road_name",
                "stream_name",
                "source",
                "road_stream_distance_m",
                "crossing_angle_degrees",
                "geometry",
            ],
            geometry="geometry",
            crs=metric_crs,
        )

    candidates = gpd.GeoDataFrame(records, geometry="geometry", crs=metric_crs)
    candidates = _deduplicate(candidates, settings.min_spacing_m)
    candidates = add_wgs84_coordinates(candidates)
    candidates["priority_seed"] = 1.0 / (1.0 + candidates["road_stream_distance_m"])
    return candidates.reset_index(drop=True)


def generate_road_route_candidates(
    roads: gpd.GeoDataFrame,
    routes: list[str] | None = None,
    interval_m: float = 75.0,
    include_numbered_roads: bool = False,
    lateral_offsets_m: tuple[float, ...] = (0.0,),
) -> gpd.GeoDataFrame:
    roads = clean_geometry(roads)
    if interval_m <= 0:
        raise ValueError("interval_m must be positive.")
    lateral_offsets = _normalized_offsets(lateral_offsets_m)

    route_tokens = {_route_token(route) for route in routes or [] if _route_token(route)}
    if not route_tokens and not include_numbered_roads:
        raise ValueError("At least one usable route is required.")

    (roads_m,), metric_crs = project_layers_to_metric(roads)
    records = []
    for road_pos, road in roads_m.iterrows():
        road_name = _optional_name(road) or ""
        road_tokens = _road_route_tokens(road)
        matched = sorted(route_tokens & road_tokens)
        if include_numbered_roads and road_tokens and _auto_sample_numbered_road(row=road):
            matched = sorted(set(matched) or road_tokens)
        if not matched:
            continue

        for part_index, line in enumerate(_line_parts(road.geometry)):
            length = float(line.length)
            if length <= 0:
                continue

            distances = np.arange(interval_m / 2, length, interval_m)
            if len(distances) == 0:
                distances = np.array([length / 2])

            for distance in distances:
                center_point = line.interpolate(float(distance))
                for offset_m in lateral_offsets:
                    point = _offset_point_from_line(line, center_point, float(offset_m))
                    records.append(
                        {
                            "candidate_id": f"route_{len(records) + 1:06d}",
                            "road_id": _row_identifier(road, road_pos, None, "road"),
                            "stream_id": "",
                            "road_name": road_name or "Unnamed road",
                            "stream_name": "route sample",
                            "road_highway": _first_value(road, ("highway", "HIGHWAY", "road_highway")),
                            "source": "route_interval_sample",
                            "matched_route": ",".join(matched),
                            "route_sample_distance_m": float(distance),
                            "route_lateral_offset_m": float(offset_m),
                            "route_part_index": int(part_index),
                            "road_stream_distance_m": np.nan,
                            "crossing_angle_degrees": np.nan,
                            "geometry": point,
                        }
                    )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "candidate_id",
                "road_id",
                "stream_id",
                "road_name",
                "stream_name",
                "source",
                "matched_route",
                "route_sample_distance_m",
                "route_lateral_offset_m",
                "geometry",
            ],
            geometry="geometry",
            crs=metric_crs,
        )

    candidates = gpd.GeoDataFrame(records, geometry="geometry", crs=metric_crs)
    candidates = add_wgs84_coordinates(candidates)
    candidates["priority_seed"] = 0.0
    return candidates.reset_index(drop=True)


def merge_candidate_layers(
    layers: list[gpd.GeoDataFrame],
    min_spacing_m: float = 5.0,
) -> gpd.GeoDataFrame:
    cleaned = [clean_geometry(layer) for layer in layers if layer is not None and not layer.empty]
    if not cleaned:
        raise ValueError("No non-empty candidate layers were provided.")

    crs = cleaned[0].crs
    aligned = [layer.to_crs(crs) if layer.crs != crs else layer.copy() for layer in cleaned]
    merged = gpd.GeoDataFrame(pd.concat(aligned, ignore_index=True), geometry="geometry", crs=crs)
    if "road_stream_distance_m" not in merged.columns:
        merged["road_stream_distance_m"] = np.nan
    merged = _deduplicate(merged, min_spacing_m)
    merged["candidate_id"] = [f"cand_{index + 1:06d}" for index in range(len(merged))]
    return merged


def _query_positions(gdf: gpd.GeoDataFrame, geometry) -> list[int]:
    try:
        return list(gdf.sindex.query(geometry, predicate="intersects"))
    except Exception:
        return list(range(len(gdf)))


def _intersection_points(geometry) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []

    geom_type = geometry.geom_type
    if geom_type == "Point":
        return [geometry]
    if geom_type == "MultiPoint":
        return list(geometry.geoms)
    if geom_type in {"LineString", "LinearRing", "Polygon"}:
        return [geometry.centroid]
    if geom_type in {"MultiLineString", "MultiPolygon"}:
        return [part.centroid for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        points: list[Point] = []
        for part in geometry.geoms:
            points.extend(_intersection_points(part))
        return points
    return []


def _row_identifier(row: pd.Series, fallback, explicit_column: str | None, prefix: str):
    if explicit_column and explicit_column in row.index:
        return row[explicit_column]

    for name in (f"{prefix}_id", "id", "ID", "objectid", "OBJECTID", "fid", "FID"):
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return fallback


def _optional_name(row: pd.Series) -> str | None:
    for name in ("name", "Name", "NAME", "road_name", "stream_name", "FULLNAME"):
        if name in row.index and pd.notna(row[name]):
            return str(row[name])
    return None


def _first_numeric(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return None


def _first_value(row: pd.Series, names: tuple[str, ...]):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def _truthy(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).lower() in {"yes", "true", "1", "bridge", "tunnel", "culvert"}


def _crossing_angle_degrees(road_geom, stream_geom, point: Point) -> float | None:
    road_angle = _local_line_angle_degrees(road_geom, point)
    stream_angle = _local_line_angle_degrees(stream_geom, point)
    if road_angle is None or stream_angle is None:
        return None

    difference = abs(road_angle - stream_angle) % 180
    crossing_angle = min(difference, 180 - difference)
    if isnan(crossing_angle):
        return None
    return float(crossing_angle)


def _local_line_angle_degrees(line, point: Point, window_m: float = 8.0) -> float | None:
    try:
        length = float(line.length)
        if length <= 0:
            return None
        distance = float(line.project(point))
        before = line.interpolate(max(0.0, distance - window_m))
        after = line.interpolate(min(length, distance + window_m))
    except Exception:
        return None

    dx = after.x - before.x
    dy = after.y - before.y
    if dx == 0 and dy == 0:
        return None
    return float(degrees(atan2(dy, dx)) % 180)


def _deduplicate(candidates: gpd.GeoDataFrame, min_spacing_m: float) -> gpd.GeoDataFrame:
    if min_spacing_m <= 0 or candidates.empty:
        return candidates

    if "road_stream_distance_m" in candidates.columns:
        ordered = candidates.sort_values(
            "road_stream_distance_m",
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        ordered = candidates.reset_index(drop=True)

    cell_size = float(min_spacing_m)
    accepted_rows = []
    accepted_cells: dict[tuple[int, int], list[pd.Series]] = {}
    for _, row in ordered.iterrows():
        point = row.geometry
        cell = (floor(point.x / cell_size), floor(point.y / cell_size))
        too_close = False
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                nearby_rows = accepted_cells.get((cell[0] + x_offset, cell[1] + y_offset), [])
                if any(
                    point.distance(existing.geometry) < _duplicate_radius_m(row, existing, min_spacing_m)
                    for existing in nearby_rows
                ):
                    too_close = True
                    break
            if too_close:
                break

        if not too_close:
            accepted_rows.append(row)
            accepted_cells.setdefault(cell, []).append(row)

    if not accepted_rows:
        return candidates.iloc[0:0].copy()

    return gpd.GeoDataFrame(accepted_rows, geometry="geometry", crs=candidates.crs).reset_index(
        drop=True
    )


def _duplicate_radius_m(candidate: pd.Series, existing: pd.Series, min_spacing_m: float) -> float:
    """Return a deduplication radius, preserving distinct nearby crossing identities."""

    coordinate_duplicate_radius_m = min(1.0, float(min_spacing_m))
    candidate_source = str(candidate.get("source", "") or "")
    existing_source = str(existing.get("source", "") or "")
    candidate_route = candidate_source == "route_interval_sample"
    existing_route = existing_source == "route_interval_sample"

    if candidate_route or existing_route:
        if _same_road(candidate, existing):
            return float(min_spacing_m)
        return coordinate_duplicate_radius_m

    same_pair = _same_identity(candidate, existing, "road") and _same_identity(
        candidate, existing, "stream"
    )
    if not same_pair:
        return coordinate_duplicate_radius_m

    exact_sources = {"exact_road_stream_intersection"}
    if candidate_source in exact_sources and existing_source in exact_sources:
        return coordinate_duplicate_radius_m
    return float(min_spacing_m)


def _same_road(left: pd.Series, right: pd.Series) -> bool:
    return _same_identity(left, right, "road")


def _same_identity(left: pd.Series, right: pd.Series, prefix: str) -> bool:
    for column in (f"{prefix}_id", f"{prefix}_name"):
        left_value = _identity_value(left.get(column))
        right_value = _identity_value(right.get(column))
        if left_value and right_value:
            return left_value == right_value
    return False


def _identity_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = " ".join(str(value).strip().lower().split())
    return "" if text in {"", "nan", "none", "<na>"} else text


def _line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return [part for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        parts = []
        for part in geometry.geoms:
            parts.extend(_line_parts(part))
        return parts
    return []


def _normalized_offsets(offsets: tuple[float, ...]) -> tuple[float, ...]:
    values = [float(value) for value in offsets]
    if not values:
        values = [0.0]
    if 0.0 not in values:
        values.insert(0, 0.0)
    return tuple(dict.fromkeys(values))


def _offset_point_from_line(line, point: Point, offset_m: float) -> Point:
    if abs(offset_m) <= 1e-9:
        return point

    angle = _local_line_angle_degrees(line, point)
    if angle is None:
        return point

    angle_rad = radians(angle)
    normal_x = -sin(angle_rad)
    normal_y = cos(angle_rad)
    return Point(point.x + normal_x * offset_m, point.y + normal_y * offset_m)


def _road_route_tokens(row: pd.Series) -> set[str]:
    values = [
        _optional_name(row),
        _first_value(row, ("ref", "REF")),
        _first_value(row, ("route", "ROUTE")),
        _first_value(row, ("road_name", "FULLNAME")),
    ]
    tokens = set()
    for value in values:
        tokens.update(_route_tokens_from_text(value))
    return tokens


def _auto_sample_numbered_road(row: pd.Series) -> bool:
    route_type = _first_value(row, ("RTTYP", "rttyp", "route_type", "ROUTE_TYPE"))
    if route_type is None or str(route_type).strip() == "":
        return True
    return str(route_type).strip().upper() in {"S", "U", "I"}


def _route_tokens_from_text(value) -> set[str]:
    text = str(value or "").upper()
    tokens = set()
    route_patterns = (
        r"\b(?:NY|NYS)\s*-?\s*(?:RTE|ROUTE|RT|HWY|HIGHWAY)?\s*-?\s*(\d+[A-Z]?)\b",
        r"\b(?:STATE\s+RTE|STATE\s+ROUTE|STATE\s+RT|ROUTE|RTE|RT)\s*-?\s*(\d+[A-Z]?)\b",
        r"\b(?:US|U\.S\.)\s*-?\s*(?:HWY|HIGHWAY|RTE|ROUTE|RT)?\s*-?\s*(\d+[A-Z]?)\b",
        r"\b(?:INTERSTATE|I)\s*-?\s*(\d+[A-Z]?)\b",
        r"\b(?:CR|CO\s+RD|COUNTY\s+ROAD|COUNTY\s+ROUTE|COUNTY\s+RTE)\s*-?\s*(\d+[A-Z]?)\b",
    )
    for pattern in route_patterns:
        for match in re.finditer(pattern, text):
            tokens.add(match.group(1))
    return tokens


def _route_token(route: str) -> str:
    tokens = _route_tokens_from_text(route)
    if tokens:
        return sorted(tokens)[0]
    match = re.search(r"\d+[A-Za-z]?", str(route or ""))
    return match.group(0).upper() if match else ""
