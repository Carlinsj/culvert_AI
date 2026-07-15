from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, Point, Polygon

from culvert_ai.io import ensure_parent_dir, project_layers_to_metric, read_vector, write_vector


DEFAULT_SCORE_THRESHOLDS = (70.0, 68.0, 65.0, 60.0, 55.0)
DEFAULT_PROBABILITY_THRESHOLDS = (0.85, 0.70, 0.50, 0.35, 0.20)
DEFAULT_SCORE_COLUMNS = (
    "discovery_score",
    "culvert_likelihood_score",
    "culvert_probability",
)
ROUTE_TOKEN_RE = re.compile(
    r"\b(?P<prefix>NY|NYS|US|U\.S\.|I|CR|COUNTY|STATE)\s*-?\s*"
    r"(?:(?:HWY|HIGHWAY|RTE|ROUTE|RT|ROAD|RD)\s*-?\s*)?"
    r"(?P<number>\d+[A-Z]?)\b",
    re.IGNORECASE,
)
GENERIC_ROUTE_TOKEN_RE = re.compile(
    r"\b(?:ROUTE|RTE|RT|HIGHWAY|HWY)\s*-?\s*(?P<number>\d+[A-Z]?)\b",
    re.IGNORECASE,
)


def write_route_count_report(
    predictions_path: str | Path,
    output_path: str | Path,
    route: str | None = None,
    segment_path: str | Path | None = None,
    buffer_m: float = 30.0,
    cluster_radius_m: float = 30.0,
    score_column: str | None = None,
    probability_column: str = "culvert_probability",
    score_threshold: float | None = None,
    thresholds: Iterable[float] | None = None,
    exclude_known_matches: bool = True,
    rank_limit: int | None = None,
    top_n: int = 50,
    csv_output: str | Path | None = None,
    geojson_output: str | Path | None = None,
) -> dict:
    predictions = read_vector(predictions_path)
    segment = read_vector(segment_path) if segment_path else None
    report, clusters = build_route_count_report(
        predictions,
        route=route,
        segment=segment,
        buffer_m=buffer_m,
        cluster_radius_m=cluster_radius_m,
        score_column=score_column,
        probability_column=probability_column,
        score_threshold=score_threshold,
        thresholds=thresholds,
        exclude_known_matches=exclude_known_matches,
        rank_limit=rank_limit,
        top_n=top_n,
    )

    ensure_parent_dir(output_path)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if csv_output:
        write_vector(clusters, csv_output)
    if geojson_output:
        write_vector(clusters.to_crs("EPSG:4326"), geojson_output)
    return {
        "route_count_report": Path(output_path),
        "clusters": int(len(clusters)),
        "predicted_count": report["recommended"]["predicted_count"],
        "expected_count": report["recommended"]["expected_count"],
        "csv_output": Path(csv_output) if csv_output else None,
        "geojson_output": Path(geojson_output) if geojson_output else None,
    }


def build_route_count_report(
    predictions: gpd.GeoDataFrame,
    route: str | None = None,
    segment: gpd.GeoDataFrame | None = None,
    buffer_m: float = 30.0,
    cluster_radius_m: float = 30.0,
    score_column: str | None = None,
    probability_column: str = "culvert_probability",
    score_threshold: float | None = None,
    thresholds: Iterable[float] | None = None,
    exclude_known_matches: bool = True,
    rank_limit: int | None = None,
    top_n: int = 50,
) -> tuple[dict, gpd.GeoDataFrame]:
    if predictions.empty:
        raise ValueError("Predictions file has no rows.")

    pool = predictions.copy().reset_index(drop=True)
    source_rows = int(len(pool))
    if exclude_known_matches:
        pool = _unknown_prediction_pool(pool).reset_index(drop=True)
    if rank_limit and "discovery_rank" in pool.columns:
        rank = pd.to_numeric(pool["discovery_rank"], errors="coerce")
        pool = pool[rank <= int(rank_limit)].reset_index(drop=True)
    if route:
        pool = pool[_route_mask(pool, route)].reset_index(drop=True)

    segment_length_m = None
    segment_area_sq_m = None
    if pool.empty:
        empty = _empty_clusters(predictions.crs)
        return _report(
            source_rows=source_rows,
            filtered_rows=0,
            clusters=empty,
            route=route,
            segment_length_m=segment_length_m,
            segment_area_sq_m=segment_area_sq_m,
            buffer_m=buffer_m,
            cluster_radius_m=cluster_radius_m,
            score_column=score_column or _score_column(predictions),
            probability_column=probability_column,
            score_threshold=score_threshold,
            thresholds=tuple(thresholds or ()),
            top_n=top_n,
        ), empty

    if segment is not None:
        pool, pool_m, segment_length_m, segment_area_sq_m = _filter_by_segment(
            pool,
            segment,
            buffer_m=buffer_m,
        )
    else:
        (pool_m,), _metric_crs = project_layers_to_metric(pool)
        pool_m = pool_m.reset_index(drop=True)

    if pool.empty:
        empty = _empty_clusters(pool_m.crs if "pool_m" in locals() else predictions.crs)
        return _report(
            source_rows=source_rows,
            filtered_rows=0,
            clusters=empty,
            route=route,
            segment_length_m=segment_length_m,
            segment_area_sq_m=segment_area_sq_m,
            buffer_m=buffer_m,
            cluster_radius_m=cluster_radius_m,
            score_column=score_column or _score_column(predictions),
            probability_column=probability_column,
            score_threshold=score_threshold,
            thresholds=tuple(thresholds or ()),
            top_n=top_n,
        ), empty

    selected_score_column = score_column or _score_column(pool_m)
    if not selected_score_column:
        raise ValueError("No usable score column found in predictions.")

    clusters = _cluster_predictions(
        pool_m,
        cluster_radius_m=cluster_radius_m,
        score_column=selected_score_column,
        probability_column=probability_column,
    )
    report = _report(
        source_rows=source_rows,
        filtered_rows=int(len(pool)),
        clusters=clusters,
        route=route,
        segment_length_m=segment_length_m,
        segment_area_sq_m=segment_area_sq_m,
        buffer_m=buffer_m,
        cluster_radius_m=cluster_radius_m,
        score_column=selected_score_column,
        probability_column=probability_column,
        score_threshold=score_threshold,
        thresholds=tuple(thresholds) if thresholds else _default_thresholds(selected_score_column),
        top_n=top_n,
    )
    return report, clusters


def _filter_by_segment(
    predictions: gpd.GeoDataFrame,
    segment: gpd.GeoDataFrame,
    buffer_m: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, float | None, float | None]:
    if segment.empty:
        raise ValueError("Segment file has no geometry rows.")
    (predictions_m, segment_m), _metric_crs = project_layers_to_metric(predictions, segment)
    predictions_m = predictions_m.reset_index(drop=True)
    segment_m = segment_m.reset_index(drop=True)
    segment_geometry = segment_m.geometry.union_all()
    corridor = _segment_corridor(segment_geometry, buffer_m=buffer_m)
    mask = predictions_m.geometry.intersects(corridor)
    filtered = predictions.loc[mask.to_numpy()].reset_index(drop=True)
    filtered_m = predictions_m.loc[mask.to_numpy()].reset_index(drop=True)
    length_m = _line_length_m(segment_geometry)
    area_sq_m = float(corridor.area) if corridor is not None else None
    return filtered, filtered_m, length_m, area_sq_m


def _segment_corridor(geometry, buffer_m: float):
    if geometry is None or geometry.is_empty:
        raise ValueError("Segment geometry is empty.")
    if isinstance(geometry, Polygon) or geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    return geometry.buffer(max(0.0, float(buffer_m)))


def _line_length_m(geometry) -> float | None:
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"LineString", "LinearRing", "MultiLineString"}:
        return float(geometry.length)
    if isinstance(geometry, GeometryCollection):
        total = 0.0
        found = False
        for part in geometry.geoms:
            length = _line_length_m(part)
            if length is not None:
                total += length
                found = True
        return total if found else None
    return None


def _cluster_predictions(
    predictions_m: gpd.GeoDataFrame,
    cluster_radius_m: float,
    score_column: str,
    probability_column: str,
) -> gpd.GeoDataFrame:
    if predictions_m.empty:
        return _empty_clusters(predictions_m.crs)

    ordered = predictions_m.copy()
    ordered["_route_count_score"] = _numeric_series(ordered, score_column)
    ordered["_route_count_probability"] = _cluster_probability_series(ordered, probability_column)
    if "discovery_rank" in ordered.columns:
        ordered["_route_count_rank"] = pd.to_numeric(ordered["discovery_rank"], errors="coerce")
    else:
        ordered["_route_count_rank"] = np.nan
    ordered = ordered.sort_values(
        ["_route_count_score", "_route_count_probability", "_route_count_rank"],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    )

    radius = max(0.0, float(cluster_radius_m))
    cell_size = max(radius, 1.0)
    cluster_points: list[Point] = []
    cluster_members: list[list[int]] = []
    grid: dict[tuple[int, int], list[int]] = {}

    for row_index, row in ordered.iterrows():
        point = row.geometry
        cell = (math.floor(point.x / cell_size), math.floor(point.y / cell_size))
        cluster_id = _nearby_cluster_id(point, cell, grid, cluster_points, radius)
        if cluster_id is None:
            cluster_id = len(cluster_points)
            cluster_points.append(point)
            cluster_members.append([])
            grid.setdefault(cell, []).append(cluster_id)
        cluster_members[cluster_id].append(int(row_index))
        member_count = len(cluster_members[cluster_id])
        if member_count > 1:
            current = cluster_points[cluster_id]
            cluster_points[cluster_id] = Point(
                current.x + (point.x - current.x) / member_count,
                current.y + (point.y - current.y) / member_count,
            )

    rows = []
    for cluster_id, member_indices in enumerate(cluster_members, start=1):
        members = ordered.loc[member_indices].copy()
        representative = members.iloc[0].copy()
        probabilities = pd.to_numeric(
            members["_route_count_probability"],
            errors="coerce",
        ).fillna(0.0)
        scores = pd.to_numeric(members["_route_count_score"], errors="coerce").fillna(0.0)
        representative["route_count_cluster_id"] = f"rc_{cluster_id:05d}"
        representative["route_count_member_count"] = int(len(members))
        representative["route_count_probability"] = float(probabilities.max()) if len(probabilities) else 0.0
        representative["route_count_score"] = float(scores.max()) if len(scores) else 0.0
        representative["route_count_expected"] = representative["route_count_probability"]
        representative["geometry"] = cluster_points[cluster_id - 1]
        rows.append(representative)

    clusters = gpd.GeoDataFrame(rows, geometry="geometry", crs=predictions_m.crs)
    clusters = clusters.drop(
        columns=[
            column
            for column in ("_route_count_score", "_route_count_probability", "_route_count_rank")
            if column in clusters.columns
        ],
        errors="ignore",
    )
    clusters = clusters.sort_values(
        ["route_count_score", "route_count_probability"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    clusters["route_count_rank"] = np.arange(1, len(clusters) + 1)
    return clusters


def _nearby_cluster_id(
    point: Point,
    cell: tuple[int, int],
    grid: dict[tuple[int, int], list[int]],
    cluster_points: list[Point],
    radius_m: float,
) -> int | None:
    if radius_m <= 0:
        return None
    best_cluster_id = None
    best_distance = None
    for x_offset in (-1, 0, 1):
        for y_offset in (-1, 0, 1):
            for cluster_id in grid.get((cell[0] + x_offset, cell[1] + y_offset), []):
                distance = float(point.distance(cluster_points[cluster_id]))
                if distance <= radius_m and (best_distance is None or distance < best_distance):
                    best_cluster_id = cluster_id
                    best_distance = distance
    return best_cluster_id


def _report(
    source_rows: int,
    filtered_rows: int,
    clusters: gpd.GeoDataFrame,
    route: str | None,
    segment_length_m: float | None,
    segment_area_sq_m: float | None,
    buffer_m: float,
    cluster_radius_m: float,
    score_column: str | None,
    probability_column: str,
    score_threshold: float | None,
    thresholds: tuple[float, ...],
    top_n: int,
) -> dict:
    probabilities = (
        pd.to_numeric(clusters.get("route_count_probability", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    expected_count = float(probabilities.sum())
    interval = _prediction_interval(probabilities)
    threshold_rows = _threshold_counts(clusters, thresholds)
    recommended = _recommended_count(
        clusters,
        expected_count=expected_count,
        interval=interval,
        score_threshold=score_threshold,
    )
    segment_miles = float(segment_length_m / 1609.344) if segment_length_m else None
    density = (
        float(recommended["predicted_count"] / segment_miles)
        if segment_miles and segment_miles > 0
        else None
    )

    return {
        "route": route or "",
        "warnings": _report_warnings(
            route=route,
            segment_length_m=segment_length_m,
            filtered_rows=filtered_rows,
        ),
        "source_rows": int(source_rows),
        "filtered_prediction_rows": int(filtered_rows),
        "cluster_radius_m": float(cluster_radius_m),
        "segment_buffer_m": float(buffer_m),
        "segment_length_m": float(segment_length_m) if segment_length_m is not None else None,
        "segment_length_miles": segment_miles,
        "segment_area_sq_m": float(segment_area_sq_m) if segment_area_sq_m is not None else None,
        "score_column": score_column or "",
        "probability_column": probability_column,
        "candidate_clusters": int(len(clusters)),
        "recommended": {
            **recommended,
            "predicted_count_per_mile": density,
        },
        "threshold_counts": threshold_rows,
        "top_clusters": _top_clusters(clusters, top_n=top_n),
    }


def _report_warnings(
    route: str | None,
    segment_length_m: float | None,
    filtered_rows: int,
) -> list[str]:
    warnings = []
    if route and segment_length_m is None:
        warnings.append(
            "Route-only reports cover every matching candidate in the prediction file. "
            "For a field-walk count, pass --segment with the walked route line or clipped corridor."
        )
    if filtered_rows == 0:
        warnings.append("No predictions matched the requested route/segment filters.")
    return warnings


def _recommended_count(
    clusters: gpd.GeoDataFrame,
    expected_count: float,
    interval: dict,
    score_threshold: float | None,
) -> dict:
    if score_threshold is not None and not clusters.empty:
        score = pd.to_numeric(clusters["route_count_score"], errors="coerce").fillna(0.0)
        count = int((score >= float(score_threshold)).sum())
        return {
            "method": "cluster_count_at_score_threshold",
            "score_threshold": float(score_threshold),
            "predicted_count": count,
            "expected_count": expected_count,
            "prediction_interval_90": interval["prediction_interval_90"],
        }

    return {
        "method": "sum_of_cluster_probabilities",
        "score_threshold": None,
        "predicted_count": int(round(expected_count)),
        "expected_count": expected_count,
        "prediction_interval_90": interval["prediction_interval_90"],
    }


def _prediction_interval(probabilities: pd.Series) -> dict:
    if probabilities.empty:
        return {"prediction_interval_90": [0, 0], "standard_deviation": 0.0}
    variance = float((probabilities * (1.0 - probabilities)).sum())
    standard_deviation = math.sqrt(max(0.0, variance))
    expected = float(probabilities.sum())
    lower = max(0, int(math.floor(expected - 1.64 * standard_deviation)))
    upper = max(lower, int(math.ceil(expected + 1.64 * standard_deviation)))
    return {
        "prediction_interval_90": [lower, upper],
        "standard_deviation": standard_deviation,
    }


def _threshold_counts(clusters: gpd.GeoDataFrame, thresholds: tuple[float, ...]) -> list[dict]:
    if clusters.empty:
        return [
            {"score_threshold": float(threshold), "cluster_count": 0, "expected_count": 0.0}
            for threshold in thresholds
        ]
    score = pd.to_numeric(clusters["route_count_score"], errors="coerce").fillna(0.0)
    probability = pd.to_numeric(clusters["route_count_probability"], errors="coerce").fillna(0.0)
    rows = []
    for threshold in thresholds:
        mask = score >= float(threshold)
        rows.append(
            {
                "score_threshold": float(threshold),
                "cluster_count": int(mask.sum()),
                "expected_count": float(probability[mask].sum()),
            }
        )
    return rows


def _top_clusters(clusters: gpd.GeoDataFrame, top_n: int) -> list[dict]:
    if clusters.empty or top_n <= 0:
        return []
    wgs84 = clusters.to_crs("EPSG:4326").head(top_n)
    rows = []
    for _, row in wgs84.iterrows():
        rows.append(
            {
                "rank": int(row.get("route_count_rank", len(rows) + 1)),
                "candidate_id": str(row.get("candidate_id", "")),
                "cluster_id": str(row.get("route_count_cluster_id", "")),
                "predicted_site_probability": _optional_float(
                    row.get("route_count_probability"),
                ),
                "score": _optional_float(row.get("route_count_score")),
                "discovery_rank": _optional_float(row.get("discovery_rank")),
                "discovery_score": _optional_float(row.get("discovery_score")),
                "culvert_probability": _optional_float(row.get("culvert_probability")),
                "road_name": str(row.get("road_name", "")),
                "matched_route": str(row.get("matched_route", "")),
                "source": str(row.get("source", "")),
                "member_count": int(row.get("route_count_member_count", 1)),
                "latitude": float(row.geometry.y),
                "longitude": float(row.geometry.x),
            }
        )
    return rows


def _unknown_prediction_pool(predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    filtered = predictions.copy()
    if "discovery_status" in filtered.columns:
        status = filtered["discovery_status"].fillna("").astype(str)
        filtered = filtered[~status.isin({"known_field_match", "field_denied"})]
    if "source" in filtered.columns:
        filtered = filtered[filtered["source"].fillna("").astype(str) != "field_report_observed_culvert"]
    if "is_known_field_match" in filtered.columns:
        known = pd.to_numeric(filtered["is_known_field_match"], errors="coerce").fillna(0).astype(int)
        filtered = filtered[known != 1]
    return filtered.copy()


def _route_mask(predictions: gpd.GeoDataFrame, route: str) -> pd.Series:
    query_tokens = _route_tokens(route)
    if not query_tokens:
        normalized_query = _normalize_route_text(route)
        return predictions.apply(
            lambda row: normalized_query in _normalize_route_text(_route_text(row)),
            axis=1,
        )
    return predictions.apply(
        lambda row: bool(query_tokens & _row_route_tokens(row)),
        axis=1,
    )


def _row_route_tokens(row: pd.Series) -> set[str]:
    tokens: set[str] = set()
    for column in ("matched_route", "road_name", "road_id", "nearest_field_report_route"):
        value = row.get(column, "")
        if pd.notna(value):
            for piece in str(value).split(","):
                tokens |= _route_tokens(piece)
    return tokens


def _route_tokens(value) -> set[str]:
    text = _normalize_route_text(value)
    tokens = set()
    for match in ROUTE_TOKEN_RE.finditer(text):
        number = match.group("number").upper()
        prefix = match.group("prefix").upper().replace(".", "")
        if prefix in {"NYS", "STATE"}:
            prefix = "NY"
        if prefix in {"COUNTY"}:
            prefix = "CR"
        tokens.add(f"{prefix}{number}")
        tokens.add(number)
    for match in GENERIC_ROUTE_TOKEN_RE.finditer(text):
        tokens.add(match.group("number").upper())
    bare = re.fullmatch(r"\s*(\d+[A-Z]?)\s*", text)
    if bare:
        tokens.add(bare.group(1).upper())
    return tokens


def _route_text(row: pd.Series) -> str:
    values = []
    for column in ("matched_route", "road_name", "road_id", "nearest_field_report_route"):
        value = row.get(column, "")
        if pd.notna(value):
            values.append(str(value))
    return " ".join(values)


def _normalize_route_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).upper().replace(".", " ")).strip()


def _score_column(table: gpd.GeoDataFrame) -> str | None:
    for column in DEFAULT_SCORE_COLUMNS:
        if column in table.columns:
            return column
    return None


def _default_thresholds(score_column: str) -> tuple[float, ...]:
    if score_column == "culvert_probability":
        return DEFAULT_PROBABILITY_THRESHOLDS
    return DEFAULT_SCORE_THRESHOLDS


def _cluster_probability_series(table: pd.DataFrame, probability_column: str) -> pd.Series:
    if probability_column in table.columns:
        return pd.to_numeric(table[probability_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    score_column = _score_column(gpd.GeoDataFrame(table, geometry="geometry", crs=getattr(table, "crs", None)))
    if score_column:
        score = pd.to_numeric(table[score_column], errors="coerce").fillna(0.0)
        scale = 1.0 if score_column == "culvert_probability" else 100.0
        return (score / scale).clip(0.0, 1.0)
    return pd.Series(0.0, index=table.index)


def _numeric_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(0.0, index=table.index)
    return pd.to_numeric(table[column], errors="coerce").fillna(0.0)


def _empty_clusters(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "route_count_cluster_id",
            "route_count_rank",
            "route_count_member_count",
            "route_count_probability",
            "route_count_score",
            "route_count_expected",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def _optional_float(value) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)
