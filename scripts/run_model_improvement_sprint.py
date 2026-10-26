"""Train and evaluate the isolated Culvert AI v3 experimental checkpoint."""
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from culvert_ai.flow_c_bundle import sha256_file
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.model_sprint import (
    FEATURE_FAMILIES,
    VERSION,
    build_candidate_sites,
    feature_parity,
    numeric_matrix,
    rank_sites,
)


OUT = ROOT / "reports/experiments/model_improvement_sprint_20261017"
MODEL_OUT = ROOT / "models/experiments/culvert_retrieval_v3_20261017"
OLD = ROOT / "reports/experiments/physical_discovery_20261016"
SELECTIVE = ROOT / "reports/experiments/selective_admission_20261016"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
KS = (25, 50, 100, 250, 500, 1000)
ACQUISITION_PROXY_FEATURES = {
    "distance_to_nearest_road_m",
    "hydrology_edge_risk",
    "candidate_edge_risk",
    "road_profile_available",
    "dem_data_available",
}


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return sha256_file(path)


def sample_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    return labels.map({label: len(labels) / (2 * count) for label, count in counts.items()}).to_numpy(float)


def estimator(family: str, features: list[str]):
    preprocess = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", add_indicator=True), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if family == "rf":
        model = RandomForestClassifier(
            n_estimators=600, min_samples_leaf=4, max_features="sqrt",
            class_weight=None, n_jobs=2, random_state=20261017,
        )
    elif family == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=5.0,
            early_stopping=False, random_state=20261017,
        )
    else:
        raise ValueError(family)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def compatible_features(training: pd.DataFrame, declared) -> list[str]:
    """Reject acquisition proxies and class-dependent missingness before fitting."""
    labels = training.target.astype(int)
    result = []
    for column in declared:
        if column not in training or column in ACQUISITION_PROXY_FEATURES:
            continue
        values = pd.to_numeric(training[column], errors="coerce").astype(float)
        missing = values.isna().groupby(labels).mean()
        if values.notna().sum() < 20 or values.dropna().nunique() <= 1:
            continue
        if abs(float(missing.get(0, 1)) - float(missing.get(1, 1))) > .2:
            continue
        result.append(column)
    if len(result) < 3:
        raise RuntimeError("Too few label-compatible physical features")
    return result


def grouped_validation(training: pd.DataFrame):
    rows, predictions = [], []
    groups = training.route.fillna("<unassigned>").astype(str)
    splitter = GroupKFold(n_splits=3)
    for feature_set, declared in FEATURE_FAMILIES.items():
        features = compatible_features(training, declared)
        x = numeric_matrix(training, features)
        for family in ("rf", "hgb"):
            oof = np.full(len(training), np.nan)
            fold_rows = []
            for fold, (train_index, test_index) in enumerate(splitter.split(x, training.target, groups)):
                y_train = training.target.iloc[train_index].astype(int)
                y_test = training.target.iloc[test_index].astype(int)
                if y_train.nunique() != 2:
                    raise RuntimeError("Grouped training fold lacks one verified label class")
                pipeline = estimator(family, features)
                pipeline.fit(
                    x.iloc[train_index], y_train,
                    model__sample_weight=sample_weights(y_train),
                )
                score = pipeline.predict_proba(x.iloc[test_index])[:, 1]
                oof[test_index] = score
                both = y_test.nunique() == 2
                fold_rows.append(dict(
                    feature_set=feature_set, model_family=family, fold=fold,
                    heldout_routes="|".join(sorted(groups.iloc[test_index].unique())),
                    rows=len(test_index), positives=int(y_test.sum()),
                    verified_negatives=int(y_test.eq(0).sum()),
                    average_precision=float(average_precision_score(y_test, score)) if y_test.sum() else np.nan,
                    roc_auc=float(roc_auc_score(y_test, score)) if both else np.nan,
                    balanced_accuracy=float(balanced_accuracy_score(y_test, score >= .5)) if both else np.nan,
                ))
            if not np.isfinite(oof).all():
                raise RuntimeError("Incomplete grouped out-of-fold predictions")
            y = training.target.astype(int)
            summary = dict(
                feature_set=feature_set, model_family=family, fold="combined",
                heldout_routes="all grouped OOF", rows=len(training), positives=int(y.sum()),
                verified_negatives=int(y.eq(0).sum()),
                average_precision=float(average_precision_score(y, oof)),
                roc_auc=float(roc_auc_score(y, oof)),
                balanced_accuracy=float(balanced_accuracy_score(y, oof >= .5)),
            )
            rows.extend([*fold_rows, summary])
            predictions.append(pd.DataFrame({
                "candidate_id": training.candidate_id,
                "route": groups,
                "target": y,
                "feature_set": feature_set,
                "model_family": family,
                "oof_score": oof,
            }))
    metrics = pd.DataFrame(rows)
    combined = metrics.loc[metrics.fold.eq("combined")].sort_values(
        ["average_precision", "roc_auc", "feature_set", "model_family"],
        ascending=[False, False, True, True], kind="stable",
    )
    choice = combined.iloc[0]
    return metrics, pd.concat(predictions, ignore_index=True), str(choice.model_family), str(choice.feature_set)


def train_final(training, family, feature_set):
    features = compatible_features(training, FEATURE_FAMILIES[feature_set])
    x = numeric_matrix(training, features)
    y = training.target.astype(int)
    pipeline = estimator(family, features)
    pipeline.fit(x, y, model__sample_weight=sample_weights(y))
    return pipeline, features


def evaluate_queue(name, queue, truth):
    rows = []
    truth_xy = np.c_[truth.geometry.x, truth.geometry.y]
    for k in (*KS, None):
        selected = queue if k is None else queue.head(k)
        matched = match_one_to_one(selected, truth_xy, 20.0)
        rows.append(dict(
            system=name, budget="full" if k is None else k,
            sites=len(selected), matches=len(matched), recall=len(matched) / len(truth),
            median_localization_error_m=float(matched.distance_m.median()) if len(matched) else np.nan,
        ))
    return rows


def subset_metrics(queue, truth, label, mask):
    subset = truth.loc[mask].copy()
    matched = match_one_to_one(queue, np.c_[subset.geometry.x, subset.geometry.y], 20.0)
    return dict(group=label, records=len(subset), matches=len(matched), recall=len(matched) / max(len(subset), 1))


def main():
    if (OUT / "COMPLETE.json").exists():
        raise RuntimeError("Completed experiment exists; use a new version instead of overwriting")
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "raw_candidates": OLD / "baseline_raw.parquet",
        "hybrid_sites": OLD / "variant_HYBRID-GROUPED.parquet",
        "baseline_sites": OLD / "baseline_sites.parquet",
        "training_manifest": SELECTIVE / "supervised_site_manifest.csv",
        "inventory_extract": SUMMER / "inventory_coverage.parquet",
        "successful_query_population": OLD / "all_959_record_diagnosis.csv",
        "baseline_queue": SELECTIVE / "queue_BASELINE-RF.parquet",
        "hybrid_queue": SELECTIVE / "queue_FULL-HYBRID-RF.parquet",
    }
    save_json(OUT / "input_hashes.json", {key: file_hash(path) for key, path in input_paths.items()})
    raw = gpd.read_parquet(input_paths["raw_candidates"])
    hybrid = gpd.read_parquet(input_paths["hybrid_sites"])
    baseline_sites = gpd.read_parquet(input_paths["baseline_sites"])
    training = pd.read_csv(input_paths["training_manifest"])
    truth = gpd.read_parquet(input_paths["inventory_extract"])
    query_population = pd.read_csv(input_paths["successful_query_population"])[
        ["record_id", "query_success"]
    ]
    truth = truth.merge(query_population, on="record_id", validate="one_to_one")
    truth = truth.loc[truth.query_success].copy()
    if set(training.target.astype(int)) != {0, 1}:
        raise RuntimeError("Both positive and field-verified negative labels are required")
    allowed = (
        (training.target.eq(1) & training.label_source.eq("authoritative_inventory_positive_20m"))
        | (training.target.eq(0) & training.label_source.eq("field_verified_negative"))
    )
    if not allowed.all():
        raise RuntimeError("Unlabeled candidates must never be converted to negatives")

    sites, ungrouped_extrema = build_candidate_sites(hybrid, raw, baseline_sites)
    sites["existing_score"] = pd.to_numeric(sites.get("score_RF"), errors="coerce")
    if "model_score" in sites:
        sites["existing_score"] = sites.existing_score.fillna(
            pd.to_numeric(sites.model_score, errors="coerce")
        )
    if sites.existing_score.isna().any():
        raise RuntimeError("Every site needs a compatible existing-model score")
    parity = feature_parity(training, sites)
    parity.to_csv(OUT / "feature_parity.csv", index=False)
    by_route = []
    for route, group in sites.groupby(sites.route_id.fillna(sites.road_name).astype(str)):
        for feature in FEATURE_FAMILIES["all_physical"]:
            values = pd.to_numeric(
                group[feature] if feature in group else pd.Series(np.nan, index=group.index),
                errors="coerce",
            ).astype(float)
            finite = values[np.isfinite(values)]
            by_route.append(dict(
                route=route, feature=feature, rows=len(group),
                missing_rate=float(1 - len(finite) / len(group)),
                median=float(finite.median()) if len(finite) else np.nan,
            ))
    pd.DataFrame(by_route).to_csv(OUT / "feature_distributions_by_route.csv", index=False)

    cv, oof, family, feature_set = grouped_validation(training)
    cv.to_csv(OUT / "grouped_validation.csv", index=False)
    oof.to_parquet(OUT / "grouped_oof_predictions.parquet", index=False)
    pipeline, features = train_final(training, family, feature_set)
    bundle = dict(
        version=VERSION, experimental=True, calibrated=False, pipeline=pipeline,
        feature_columns=features, feature_set=feature_set, model_family=family,
        missing_value_handling="median imputation fitted on training plus explicit missing indicators",
        label_contract={"positive": "authoritative inventory physical site <=20m", "negative": "field verified only", "unlabeled": "excluded from supervised loss"},
        training_rows=len(training), training_routes=sorted(training.route.astype(str).unique()),
        inference_requires_training_data=False,
    )
    joblib.dump(bundle, MODEL_OUT / "model.joblib")
    joblib.dump(pipeline.named_steps["preprocess"], MODEL_OUT / "preprocessor.joblib")
    save_json(MODEL_OUT / "feature_schema.json", {
        "version": VERSION, "ordered_features": features, "feature_set": feature_set,
        "crs": "EPSG:32618", "missing_values": "preserved as NaN until fitted median imputer; indicators appended",
    })
    save_json(MODEL_OUT / "training_config.json", {
        "selected_by": "maximum combined grouped-OOF average precision on genuinely labeled sites",
        "model_family": family, "feature_set": feature_set, "random_state": 20261017,
        "candidate_generation": "existing hybrid plus grouped 120m-window multi-scale physical extrema",
    })
    save_json(MODEL_OUT / "dataset_provenance.json", {
        "inputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)} for key, path in input_paths.items()},
        "summer_inventory_used_for_training": False,
        "summer_inventory_used_for_retrospective_evaluation": True,
        "historical_inventory_exposure_warning": True,
    })
    cv.to_csv(MODEL_OUT / "validation_metrics.csv", index=False)
    save_json(MODEL_OUT / "inference.json", {
        "command": "scripts/python.sh scripts/predict_sprint_candidates.py --input <candidate parquet> --output <ranked parquet>",
        "required_candidate_contract": "the ordered feature schema plus EPSG:32618 coordinates",
    })

    sites["score_v3"] = pipeline.predict_proba(numeric_matrix(sites, features))[:, 1]
    ranked_raw = rank_sites(sites, "score_v3", "raw").frame
    # Existing soft-diversity implementation is deliberately evaluated separately.
    ranked_diverse = rank_sites(sites, "score_v3", "soft_local_diversity").frame
    baseline = gpd.read_parquet(input_paths["baseline_queue"])
    hybrid_queue = gpd.read_parquet(input_paths["hybrid_queue"])
    baseline_ids = set(baseline.candidate_id.astype(str))
    protected_tail = ranked_raw.loc[~ranked_raw.candidate_id.astype(str).isin(baseline_ids)]
    baseline_first = pd.concat([baseline, protected_tail], ignore_index=True, sort=False)
    baseline_first = gpd.GeoDataFrame(baseline_first, geometry="geometry", crs=sites.crs)
    baseline_first["global_rank"] = np.arange(1, len(baseline_first) + 1)
    baseline_first["ranking_policy"] = "baseline_first_then_v3"

    queues = {
        "existing_baseline_raw_rf": baseline,
        "existing_hybrid_raw_rf": hybrid_queue,
        "v3_model_raw": ranked_raw,
        "v3_model_soft_local_diversity": ranked_diverse,
        "v3_baseline_protected": baseline_first,
    }
    metrics = []
    for name, queue in queues.items():
        metrics.extend(evaluate_queue(name, queue, truth))
    pd.DataFrame(metrics).to_csv(OUT / "retrieval_metrics.csv", index=False)

    center = gpd.GeoSeries.from_xy([-74.110507], [41.880519], crs=4326).to_crs(32618).iloc[0]
    regional = []
    for name, queue in queues.items():
        regional.append(subset_metrics(queue, truth, "Region 6", truth.Region.eq("Region 6")))
        regional[-1]["system"] = name
        regional.append(subset_metrics(queue, truth, "Region 8", truth.Region.eq("Region 8")))
        regional[-1]["system"] = name
        regional.append(subset_metrics(queue, truth, "Rock Ledge area", truth.geometry.distance(center).le(1500)))
        regional[-1]["system"] = name
    pd.DataFrame(regional).to_csv(OUT / "regional_metrics.csv", index=False)

    if family == "rf":
        transformed = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance = pipeline.named_steps["model"].feature_importances_
        pd.DataFrame({"feature": transformed, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(OUT / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance"]).to_csv(OUT / "feature_importance.csv", index=False)

    export_columns = [
        "physical_site_id", "candidate_id", "candidate_source", "road_id", "road_name",
        "route_id", "road_part_index", "road_chainage_m", "proposal_kind",
        "physical_evidence_sources", "physical_evidence_count", "quality_missing_count",
        "existing_score", "score_v3", "research_priority", "global_rank",
        "member_candidate_ids", "member_count", "geometry",
    ]
    export_columns = [column for column in export_columns if column in ranked_raw]
    export = ranked_raw[export_columns].copy()
    export.to_parquet(OUT / "ranked_physical_sites.parquet", index=False)
    tabular = export.copy()
    tabular["longitude"] = tabular.geometry.to_crs(4326).x
    tabular["latitude"] = tabular.geometry.to_crs(4326).y
    tabular.drop(columns="geometry").to_csv(OUT / "ranked_physical_sites.csv", index=False)
    export.head(10000).to_crs(4326).to_file(OUT / "ranked_physical_sites_top10000.geojson", driver="GeoJSON")
    ungrouped_extrema[["candidate_id", "road_id", "road_part_index", "road_chainage_m", "physical_evidence_sources", "geometry"]].to_parquet(
        OUT / "raw_multiscale_evidence.parquet", index=False
    )

    save_json(OUT / "candidate_accounting.json", {
        "existing_baseline_sites": len(baseline), "existing_hybrid_sites": len(hybrid),
        "raw_multiscale_extrema_not_already_represented": len(ungrouped_extrema),
        "new_grouped_total_sites": len(sites),
        "grouping_rule": "same road and part, complete chainage span <=20m; existing hybrid anchors never merged or moved",
        "raw_evidence_retained": True,
    })
    complete = {
        "version": VERSION, "selected_model": f"{family}:{feature_set}",
        "model_sha256": file_hash(MODEL_OUT / "model.joblib"),
        "sites": len(sites), "target_records": len(truth),
        "production_changed": False, "frozen_models_changed": False,
    }
    save_json(OUT / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
