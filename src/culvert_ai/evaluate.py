from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

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
