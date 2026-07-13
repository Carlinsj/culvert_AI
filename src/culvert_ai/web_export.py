from __future__ import annotations

import gzip
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from culvert_ai.io import ensure_parent_dir, read_vector


WEB_COLUMNS = [
    "candidate_id",
    "discovery_rank",
    "discovery_score",
    "discovery_status",
    "is_known_field_match",
    "evidence_score",
    "model_probability_score",
    "model_rank_score",
    "field_recall_score",
    "priority_rank",
    "priority_bucket",
    "culvert_likelihood_score",
    "culvert_probability",
    "road_name",
    "stream_name",
    "matched_route",
    "road_id",
    "stream_id",
    "source",
    "evidence_summary",
    "google_earth_url",
    "latitude",
    "longitude",
    "road_stream_distance_m",
    "crossing_angle_degrees",
    "road_stream_proximity_score",
    "drainage_strength_score",
    "valley_position_score",
    "crossing_geometry_score",
    "terrain_break_score",
    "road_context_score",
    "dem_route_drainage_score",
    "osm_culvert_tag_score",
    "field_report_support_score",
    "field_corridor_support_score",
    "non_culvert_structure_penalty",
    "dist_to_known_culvert_m",
    "is_culvert",
    "stream_order",
    "road_highway",
    "road_bridge",
    "road_tunnel",
    "stream_waterway",
    "stream_tunnel",
    "stream_culvert",
    "field_report_source_file",
    "field_report_date",
    "nearest_field_report_date",
    "nearest_field_report_route",
    "nearest_field_report_culvert_id",
    "nearest_field_report_source_file",
    "slope_degrees",
    "valley_depth_9x9_m",
    "valley_depth_31x31_m",
    "topographic_position_9x9_m",
    "topographic_wetness_proxy_9x9",
    "topographic_wetness_proxy_31x31",
    "low_slope_valley_score_9x9",
    "terrain_break_score_proxy_9x9",
    "flow_accumulation_log",
    "flow_accumulation_rank_pct",
    "drainage_area_log",
    "drainage_area_rank_pct",
    "terrain_roughness_9x9_m",
    "stream_density_250m_m_per_sqkm",
    "road_density_250m_m_per_sqkm",
]

ROUTE_COUNT_COLUMNS = [
    "candidate_id",
    "discovery_rank",
    "discovery_score",
    "discovery_status",
    "is_known_field_match",
    "priority_rank",
    "priority_bucket",
    "culvert_likelihood_score",
    "culvert_probability",
    "road_name",
    "stream_name",
    "matched_route",
    "road_id",
    "source",
    "evidence_summary",
    "latitude",
    "longitude",
    "dist_to_known_culvert_m",
    "is_culvert",
    "nearest_field_report_route",
]

KNOWN_EXPORT_EXCLUSION_RADIUS_M = 15.0
WEB_EXPORT_MIN_SPACING_M = 15.0
WEB_EXPORT_MAX_PER_ROAD = 1000
WEB_EXPORT_FIELD_RECALL_SHARE = 0.35
WEB_EXPORT_FIELD_RECALL_MIN_SCORE = 45.0
WEB_EXPORT_FIELD_RECALL_MIN_SPACING_M = 15.0
WEB_EXPORT_FIELD_RECALL_MAX_PER_ROAD = 1000


def export_web_data(
    predictions_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict:
    predictions = read_vector(predictions_path)
    if predictions.empty:
        raise ValueError("Predictions file has no rows.")

    known_matches_total = _known_match_count(predictions)
    predictions = _prediction_export_pool(predictions)
    if predictions.empty:
        raise ValueError("No discovery candidate rows remain after filtering known matches.")

    sort_column = _score_column(predictions)
    if "discovery_rank" in predictions.columns:
        predictions = predictions.sort_values("discovery_rank")
    elif sort_column:
        predictions = predictions.sort_values(sort_column, ascending=False)
    elif "priority_rank" in predictions.columns:
        predictions = predictions.sort_values("priority_rank")

    route_count_predictions = predictions.copy()

    if limit:
        predictions = _limit_for_web(predictions, limit)

    web = predictions.to_crs("EPSG:4326").copy()
    if "longitude" not in web.columns:
        web["longitude"] = web.geometry.x
    if "latitude" not in web.columns:
        web["latitude"] = web.geometry.y

    selected_columns = [column for column in WEB_COLUMNS if column in web.columns]
    web = web[[*selected_columns, "geometry"]].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = output_dir / "findings.geojson"
    summary_path = output_dir / "summary.json"
    route_count_path = output_dir / "route_count_source.geojson.gz"

    ensure_parent_dir(geojson_path)
    web.to_file(geojson_path, driver="GeoJSON")
    route_count_rows = _write_route_count_source(route_count_predictions, route_count_path)
    summary = _summary(web, sort_column, known_matches_total=known_matches_total)
    summary["route_count_rows"] = route_count_rows
    summary["route_count_source"] = route_count_path.name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "findings_geojson": geojson_path,
        "summary_json": summary_path,
        "route_count_source": route_count_path,
        "route_count_rows": route_count_rows,
        "rows": int(len(web)),
        "score_column": sort_column or "",
    }


def _write_route_count_source(predictions: gpd.GeoDataFrame, output_path: Path) -> int:
    route_count = predictions.to_crs("EPSG:4326").copy()
    if "longitude" not in route_count.columns:
        route_count["longitude"] = route_count.geometry.x
    if "latitude" not in route_count.columns:
        route_count["latitude"] = route_count.geometry.y

    selected_columns = [column for column in ROUTE_COUNT_COLUMNS if column in route_count.columns]
    route_count = route_count[[*selected_columns, "geometry"]].copy()

    ensure_parent_dir(output_path)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write('{"type":"FeatureCollection","features":[\n')
        for index, feature in enumerate(route_count.iterfeatures(na="null")):
            if index:
                handle.write(",\n")
            handle.write(json.dumps(feature, separators=(",", ":")))
        handle.write("\n]}\n")
    return int(len(route_count))


def _score_column(table: gpd.GeoDataFrame) -> str | None:
    for column in ("discovery_score", "culvert_likelihood_score", "culvert_probability"):
        if column in table.columns:
            return column
    return None


def _limit_for_web(predictions: gpd.GeoDataFrame, limit: int) -> gpd.GeoDataFrame:
    if "discovery_status" not in predictions.columns:
        return predictions.head(limit)

    pool = _prediction_export_pool(predictions)
    recall = _field_recall_export_pool(pool)
    recall_limit = int(round(limit * WEB_EXPORT_FIELD_RECALL_SHARE))
    recall = _decluster_for_web(
        recall,
        limit=recall_limit,
        min_spacing_m=WEB_EXPORT_FIELD_RECALL_MIN_SPACING_M,
        max_per_road=WEB_EXPORT_FIELD_RECALL_MAX_PER_ROAD,
    )

    remaining = _drop_exported_candidates(pool, recall)
    discovery = _decluster_for_web(
        remaining,
        limit=limit,
        min_spacing_m=WEB_EXPORT_MIN_SPACING_M,
        max_per_road=WEB_EXPORT_MAX_PER_ROAD,
    )
    remaining_slots = max(0, limit - len(recall))
    discovery = discovery.head(remaining_slots)
    combined = pd.concat([recall, discovery], ignore_index=True)
    if "candidate_id" in combined.columns:
        combined = combined.drop_duplicates("candidate_id")
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=predictions.crs)
    return _sort_for_web(combined).head(limit)


def _prediction_export_pool(predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    filtered = predictions.copy()
    if "discovery_status" in filtered.columns:
        status = filtered["discovery_status"].fillna("").astype(str)
        filtered = filtered[~status.isin({"known_field_match", "field_denied"})]
    if "is_known_field_match" in filtered.columns:
        known = pd.to_numeric(filtered["is_known_field_match"], errors="coerce").fillna(0).astype(int)
        filtered = filtered[known != 1]
    if "dist_to_known_culvert_m" in filtered.columns:
        distance = pd.to_numeric(filtered["dist_to_known_culvert_m"], errors="coerce")
        filtered = filtered[distance.isna() | (distance > KNOWN_EXPORT_EXCLUSION_RADIUS_M)]
    if "source" in filtered.columns:
        source = filtered["source"].fillna("").astype(str)
        filtered = filtered[source != "field_report_observed_culvert"]
    return gpd.GeoDataFrame(filtered, geometry="geometry", crs=predictions.crs)


def _field_recall_export_pool(predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if predictions.empty or "field_recall_score" not in predictions.columns:
        return predictions.head(0)

    score = pd.to_numeric(predictions["field_recall_score"], errors="coerce").fillna(0.0)
    recall = predictions[score >= WEB_EXPORT_FIELD_RECALL_MIN_SCORE].copy()
    if recall.empty:
        return recall

    sort_columns = [
        column
        for column in ("field_recall_score", "discovery_score", "culvert_likelihood_score")
        if column in recall.columns
    ]
    if sort_columns:
        recall = recall.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return gpd.GeoDataFrame(recall, geometry="geometry", crs=predictions.crs)


def _drop_exported_candidates(
    predictions: gpd.GeoDataFrame,
    exported: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if exported.empty:
        return predictions
    if "candidate_id" in predictions.columns and "candidate_id" in exported.columns:
        exported_ids = set(exported["candidate_id"].fillna("").astype(str))
        return predictions[~predictions["candidate_id"].fillna("").astype(str).isin(exported_ids)]
    return predictions.drop(index=exported.index.intersection(predictions.index), errors="ignore")


def _sort_for_web(predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if predictions.empty:
        return predictions
    if "discovery_rank" in predictions.columns:
        return predictions.sort_values("discovery_rank")
    sort_column = _score_column(predictions)
    if sort_column:
        return predictions.sort_values(sort_column, ascending=False)
    if "priority_rank" in predictions.columns:
        return predictions.sort_values("priority_rank")
    return predictions


def _decluster_for_web(
    predictions: gpd.GeoDataFrame,
    limit: int,
    min_spacing_m: float,
    max_per_road: int,
) -> gpd.GeoDataFrame:
    if predictions.empty or limit <= 0:
        return predictions.head(0)
    if min_spacing_m <= 0 and max_per_road <= 0:
        return predictions.head(limit)

    metric_crs = predictions.estimate_utm_crs() or "EPSG:3857"
    metric = predictions.to_crs(metric_crs)
    cell_size = max(float(min_spacing_m), 1.0)
    accepted_indices = []
    accepted_cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    road_counts: dict[str, int] = {}

    for row_index, row in metric.iterrows():
        road_key = _road_export_key(row)
        if max_per_road > 0 and road_counts.get(road_key, 0) >= max_per_road:
            continue

        point = row.geometry
        cell = (int(np.floor(point.x / cell_size)), int(np.floor(point.y / cell_size)))
        if min_spacing_m > 0 and _too_close_to_accepted(point.x, point.y, cell, accepted_cells, min_spacing_m):
            continue

        accepted_indices.append(row_index)
        accepted_cells.setdefault(cell, []).append((float(point.x), float(point.y)))
        road_counts[road_key] = road_counts.get(road_key, 0) + 1
        if len(accepted_indices) >= limit:
            break

    return predictions.loc[accepted_indices]


def _road_export_key(row: pd.Series) -> str:
    for column in ("matched_route", "road_name", "road_id"):
        value = _export_key_value(row.get(column, ""))
        if value:
            return f"{column}:{value.lower()}"
    return "road:unknown"


def _export_key_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    return " ".join(text.split())


def _too_close_to_accepted(
    x: float,
    y: float,
    cell: tuple[int, int],
    accepted_cells: dict[tuple[int, int], list[tuple[float, float]]],
    min_spacing_m: float,
) -> bool:
    min_sq = float(min_spacing_m) * float(min_spacing_m)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for accepted_x, accepted_y in accepted_cells.get((cell[0] + dx, cell[1] + dy), []):
                if (x - accepted_x) ** 2 + (y - accepted_y) ** 2 < min_sq:
                    return True
    return False


def _summary(
    table: gpd.GeoDataFrame,
    score_column: str | None,
    known_matches_total: int | None = None,
) -> dict:
    buckets = {}
    if "priority_bucket" in table.columns:
        buckets = {
            str(bucket): int(count)
            for bucket, count in table["priority_bucket"].fillna("unknown").value_counts().items()
        }

    exported_known_matches = _known_match_count(table)
    known_matches = exported_known_matches if known_matches_total is None else int(known_matches_total)

    score_table = table
    if "discovery_status" in table.columns:
        score_table = table[table["discovery_status"] != "known_field_match"]
    scores = (
        pd.to_numeric(score_table[score_column], errors="coerce")
        if score_column
        else pd.Series(dtype=float)
    )
    bounds = [float(value) for value in table.total_bounds]
    return {
        "rows": int(len(table)),
        "discovery_candidates": int(len(table) - exported_known_matches),
        "known_field_matches": known_matches,
        "exported_known_field_matches": exported_known_matches,
        "score_column": score_column,
        "max_score": float(scores.max()) if not scores.empty else None,
        "mean_score": float(scores.mean()) if not scores.empty else None,
        "priority_buckets": buckets,
        "bounds": bounds,
    }


def _known_match_count(table: gpd.GeoDataFrame) -> int:
    if "is_known_field_match" in table.columns:
        return int(pd.to_numeric(table["is_known_field_match"], errors="coerce").fillna(0).sum())
    if "discovery_status" in table.columns:
        return int((table["discovery_status"].fillna("").astype(str) == "known_field_match").sum())
    if "is_culvert" in table.columns:
        return int(pd.to_numeric(table["is_culvert"], errors="coerce").fillna(0).sum())
    return 0
