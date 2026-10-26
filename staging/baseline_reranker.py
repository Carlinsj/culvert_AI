"""Executable inference adapter for the isolated v5 baseline-preserving reranker."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from staging.hydrology_service import ROOT

MODEL = ROOT / "models/experiments/baseline_preserving_v5_20261019/reranker.joblib"
VERSION = "baseline-preserving-physical-reranker-v5-experimental"


def load_bundle(path: Path = MODEL) -> dict:
    bundle = joblib.load(path)
    if bundle.get("version") != VERSION or not bundle.get("experimental"):
        raise ValueError("An isolated experimental v5 bundle is required")
    return bundle


def rank(frame: gpd.GeoDataFrame, bundle: dict | None = None) -> gpd.GeoDataFrame:
    bundle = load_bundle() if bundle is None else bundle
    required = [bundle["original_rf_score_column"], *bundle["physical_feature_columns"]]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing v5 inference fields: {missing}")
    base = pd.to_numeric(frame[bundle["original_rf_score_column"]], errors="coerce")
    if not np.isfinite(base).all():
        raise ValueError("Original RF score must be finite for every v5 candidate")
    matrix = frame[bundle["physical_feature_columns"]].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    physical = bundle["physical_estimator"].predict_proba(matrix)[:, 1]
    alpha = float(bundle["blend_alpha"])
    combined = (1 - alpha) * base.rank(method="average", pct=True).to_numpy(float)
    combined += alpha * pd.Series(physical).rank(method="average", pct=True).to_numpy(float)
    result = frame.copy()
    result["score_v5_physical"] = physical
    result["score_v5"] = combined
    result = result.sort_values(["score_v5", "candidate_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["model_score"] = result.score_v5
    return result
