from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
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
    "priority_rank",
    "priority_bucket",
    "culvert_likelihood_score",
    "culvert_probability",
    "road_name",
    "stream_name",
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
    "osm_culvert_tag_score",
    "field_report_support_score",
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


def export_web_data(
    predictions_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict:
    predictions = read_vector(predictions_path)
    if predictions.empty:
        raise ValueError("Predictions file has no rows.")

    sort_column = _score_column(predictions)
    if "discovery_rank" in predictions.columns:
        predictions = predictions.sort_values("discovery_rank")
    elif sort_column:
        predictions = predictions.sort_values(sort_column, ascending=False)
    elif "priority_rank" in predictions.columns:
        predictions = predictions.sort_values("priority_rank")

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

    ensure_parent_dir(geojson_path)
    web.to_file(geojson_path, driver="GeoJSON")
    summary = _summary(web, sort_column)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "findings_geojson": geojson_path,
        "summary_json": summary_path,
        "rows": int(len(web)),
        "score_column": sort_column or "",
    }


def _score_column(table: gpd.GeoDataFrame) -> str | None:
    for column in ("discovery_score", "culvert_likelihood_score", "culvert_probability"):
        if column in table.columns:
            return column
    return None


def _limit_for_web(predictions: gpd.GeoDataFrame, limit: int) -> gpd.GeoDataFrame:
    if "discovery_status" not in predictions.columns:
        return predictions.head(limit)

    known = predictions[predictions["discovery_status"] == "known_field_match"]
    discovery = predictions[predictions["discovery_status"] != "known_field_match"].head(limit)
    combined = pd.concat([discovery, known], ignore_index=True)
    if "candidate_id" in combined.columns:
        combined = combined.drop_duplicates("candidate_id")
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=predictions.crs)


def _summary(table: gpd.GeoDataFrame, score_column: str | None) -> dict:
    buckets = {}
    if "priority_bucket" in table.columns:
        buckets = {
            str(bucket): int(count)
            for bucket, count in table["priority_bucket"].fillna("unknown").value_counts().items()
        }

    known_matches = 0
    if "is_known_field_match" in table.columns:
        known_matches = int(pd.to_numeric(table["is_known_field_match"], errors="coerce").fillna(0).sum())
    elif "is_culvert" in table.columns:
        known_matches = int(pd.to_numeric(table["is_culvert"], errors="coerce").fillna(0).sum())

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
        "discovery_candidates": int(len(table) - known_matches),
        "known_field_matches": known_matches,
        "score_column": score_column,
        "max_score": float(scores.max()) if not scores.empty else None,
        "mean_score": float(scores.mean()) if not scores.empty else None,
        "priority_buckets": buckets,
        "bounds": bounds,
    }
