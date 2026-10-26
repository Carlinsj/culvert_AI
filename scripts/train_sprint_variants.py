"""Persist every bounded RF/HGB sprint variant plus one documented PU control."""
# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_model_improvement_sprint import (
    FEATURE_FAMILIES,
    MODEL_OUT,
    OLD,
    SELECTIVE,
    VERSION,
    compatible_features,
    estimator,
    numeric_matrix,
    sample_weights,
)
from staging.model_sprint import build_candidate_sites


def main():
    variants = MODEL_OUT / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(SELECTIVE / "supervised_site_manifest.csv")
    y = training.target.astype(int)
    manifest = []
    for feature_set, declared in FEATURE_FAMILIES.items():
        features = compatible_features(training, declared)
        for family in ("rf", "hgb"):
            pipeline = estimator(family, features)
            pipeline.fit(
                numeric_matrix(training, features), y,
                model__sample_weight=sample_weights(y),
            )
            path = variants / f"{family}_{feature_set}.joblib"
            joblib.dump(dict(
                version=VERSION, experimental=True, calibrated=False,
                model_family=family, feature_set=feature_set,
                feature_columns=features, pipeline=pipeline,
                label_contract="authoritative positives and field-verified negatives only",
                acquisition_proxy_screen=True,
            ), path)
            manifest.append(dict(model=path.name, family=family, feature_set=feature_set,
                                 features=len(features), training_rows=len(training)))

    # Controlled PU alternative: unlabeled is sampled as an imperfect mixture,
    # never asserted to be a verified negative and never used for accuracy claims.
    raw = gpd.read_parquet(OLD / "baseline_raw.parquet")
    hybrid = gpd.read_parquet(OLD / "variant_HYBRID-GROUPED.parquet")
    baseline = gpd.read_parquet(OLD / "baseline_sites.parquet")
    sites, _ = build_candidate_sites(hybrid, raw, baseline)
    positives = training.loc[training.target.eq(1)].copy()
    selected_features = joblib.load(MODEL_OUT / "model.joblib")["feature_columns"]
    distance = cKDTree(positives[["x_m", "y_m"]]).query(
        np.c_[sites.geometry.x, sites.geometry.y]
    )[0]
    unlabeled = sites.loc[distance > 40].sample(
        n=min(20000, int((distance > 40).sum())), random_state=20261017
    )
    pu_train = pd.concat([positives, unlabeled], ignore_index=True, sort=False)
    pu_y = np.r_[np.ones(len(positives), int), np.zeros(len(unlabeled), int)]
    pu = estimator("hgb", selected_features)
    weights = np.where(pu_y == 1, .5 / len(positives), .5 / len(unlabeled))
    pu.fit(numeric_matrix(pu_train, selected_features), pu_y, model__sample_weight=weights)
    pu_path = variants / "hgb_positive_unlabeled_control.joblib"
    joblib.dump(dict(
        version=VERSION, experimental=True, calibrated=False,
        model_family="hgb_positive_unlabeled_control", feature_set="all_physical_screened",
        feature_columns=selected_features, pipeline=pu,
        assumptions="SCAR-like ranking control; sampled unlabeled mixture is not a negative class",
        authoritative_positive_rows=len(positives), sampled_unlabeled_rows=len(unlabeled),
        evaluation_status="controlled alternative, not selected",
    ), pu_path)
    manifest.append(dict(model=pu_path.name, family="hgb_positive_unlabeled_control",
                         feature_set="all_physical_screened", features=len(selected_features),
                         training_rows=len(pu_train)))
    (variants / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
