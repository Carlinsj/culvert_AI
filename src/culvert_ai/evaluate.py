from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from culvert_ai.io import ensure_parent_dir, project_layers_to_metric


def evaluate_predictions(
    predictions: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
    probability_column: str = "culvert_probability",
    probability_threshold: float = 0.7,
    match_radius_m: float = 30.0,
) -> dict:
    if probability_column not in predictions.columns:
        raise ValueError(f"Prediction probability column not found: {probability_column}")

    (predictions_m, known_m), _metric_crs = project_layers_to_metric(predictions, known_culverts)
    high_priority = predictions_m[predictions_m[probability_column] >= probability_threshold]

    if high_priority.empty:
        metrics = {
            "high_priority_predictions": 0,
            "known_culverts": int(len(known_m)),
            "precision_at_threshold": 0.0,
            "known_culvert_recall_at_threshold": 0.0,
        }
    else:
        known_union = known_m.geometry.unary_union
        pred_hits = high_priority.geometry.apply(lambda geom: geom.distance(known_union) <= match_radius_m)
        high_union = high_priority.geometry.unary_union
        known_hits = known_m.geometry.apply(lambda geom: geom.distance(high_union) <= match_radius_m)

        metrics = {
            "high_priority_predictions": int(len(high_priority)),
            "known_culverts": int(len(known_m)),
            "precision_at_threshold": float(pred_hits.sum() / len(high_priority)),
            "known_culvert_recall_at_threshold": float(known_hits.sum() / len(known_m))
            if len(known_m)
            else 0.0,
            "probability_threshold": probability_threshold,
            "match_radius_m": match_radius_m,
        }

    if output_path:
        ensure_parent_dir(output_path)
        Path(output_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def evaluate_success_rate_at_actuals(
    predictions: gpd.GeoDataFrame,
    actual_culverts: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
    max_distance_m: float = 15.0,
    exclude_known_matches: bool = True,
    rank_limit: int | None = None,
) -> dict:
    """Measure field success as actual culverts with a prediction within max_distance_m."""

    if predictions.empty:
        raise ValueError("Predictions file has no rows.")
    if actual_culverts.empty:
        raise ValueError("Actual culvert file has no rows.")

    prediction_pool = predictions.copy()
    if exclude_known_matches:
        prediction_pool = _unknown_prediction_pool(prediction_pool)
    if rank_limit and "discovery_rank" in prediction_pool.columns:
        prediction_pool = prediction_pool[
            pd.to_numeric(prediction_pool["discovery_rank"], errors="coerce") <= rank_limit
        ].copy()
    if prediction_pool.empty:
        raise ValueError("No prediction rows remain after filtering.")

    (predictions_m, actuals_m), _metric_crs = project_layers_to_metric(
        prediction_pool,
        actual_culverts,
    )

    rows = []
    for actual_index, actual in actuals_m.reset_index(drop=True).iterrows():
        distances = predictions_m.geometry.distance(actual.geometry)
        nearest_index = distances.idxmin()
        nearest = predictions_m.loc[nearest_index]
        distance_m = float(distances.loc[nearest_index])
        rows.append(
            {
                "actual_index": int(actual_index),
                "actual_id": _actual_id(actual),
                "nearest_candidate_id": str(nearest.get("candidate_id", "")),
                "nearest_distance_m": distance_m,
                "hit": distance_m <= max_distance_m,
                "discovery_rank": _optional_number(nearest.get("discovery_rank")),
                "discovery_score": _optional_number(nearest.get("discovery_score")),
                "source": str(nearest.get("source", "")),
                "road_name": str(nearest.get("road_name", "")),
            }
        )

    hit_count = sum(1 for row in rows if row["hit"])
    metrics = {
        "actual_culverts": int(len(rows)),
        "prediction_candidates": int(len(predictions_m)),
        "hits_within_distance": int(hit_count),
        "success_rate": float(hit_count / len(rows)) if rows else 0.0,
        "max_distance_m": float(max_distance_m),
        "exclude_known_matches": bool(exclude_known_matches),
        "rank_limit": int(rank_limit) if rank_limit else None,
        "mean_nearest_distance_m": float(pd.Series(row["nearest_distance_m"] for row in rows).mean()),
        "misses": [row for row in rows if not row["hit"]],
        "matches": rows,
    }

    if output_path:
        ensure_parent_dir(output_path)
        Path(output_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _unknown_prediction_pool(predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    filtered = predictions.copy()
    if "discovery_status" in filtered.columns:
        filtered = filtered[filtered["discovery_status"] != "known_field_match"]
    if "source" in filtered.columns:
        filtered = filtered[filtered["source"] != "field_report_observed_culvert"]
    if "is_known_field_match" in filtered.columns:
        known = pd.to_numeric(filtered["is_known_field_match"], errors="coerce").fillna(0).astype(int)
        filtered = filtered[known != 1]
    return filtered.copy()


def _actual_id(row) -> str:
    for column in ("culvert_id", "field_culvert_id", "observation_id", "candidate_id"):
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value)
    return str(row.name)


def _optional_number(value):
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)
