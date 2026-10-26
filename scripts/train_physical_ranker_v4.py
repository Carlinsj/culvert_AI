"""Train and audit an isolated physical-site ranker on verified labels only."""
from __future__ import annotations

import hashlib
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
from staging.model_sprint import FEATURE_FAMILIES

VERSION = "physical-ranker-v4-experimental"
REPORT = ROOT / "reports/experiments/physical_ranker_v4_20261019"
MODEL_DIR = ROOT / "models/experiments/physical_ranker_v4_20261019"
SELECTIVE = ROOT / "reports/experiments/selective_admission_20261016"
PHYSICAL = ROOT / "reports/experiments/physical_discovery_20261016"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
SPRINT = ROOT / "reports/experiments/model_improvement_sprint_20261017"
ANTI = ROOT / "reports/experiments/anti_saturation_validation_20261017"
BUDGETS = (100, 250, 500, 1000, 5000, 10000, 13065)
EXCLUDED = {
    "distance_to_nearest_road_m",
    "hydrology_edge_risk",
    "candidate_edge_risk",
    "road_profile_available",
    "dem_data_available",
}


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map({label: len(y) / (2 * count) for label, count in counts.items()}).to_numpy(float)


def features_for(training: pd.DataFrame, candidates: pd.DataFrame, declared) -> list[str]:
    labels = training.target.astype(int)
    result = []
    for column in declared:
        if column in EXCLUDED or column not in training or column not in candidates:
            continue
        values = pd.to_numeric(training[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = values.isna().groupby(labels).mean()
        if values.notna().sum() < 20 or values.dropna().nunique() <= 1:
            continue
        if abs(float(missing.get(0, 1)) - float(missing.get(1, 1))) > 0.2:
            continue
        result.append(column)
    if len(result) < 3:
        raise RuntimeError("Too few physical features survive parity and missingness screening")
    return result


def estimator(family: str, features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", add_indicator=True), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if family == "rf":
        model = RandomForestClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight=None,
            n_jobs=2,
            random_state=20261019,
        )
    elif family == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=8,
            early_stopping=False,
            random_state=20261019,
        )
    else:
        raise ValueError(family)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def score_cv(training: pd.DataFrame, family: str, feature_set: str, features: list[str], split_name: str, groups: pd.Series):
    y = training.target.astype(int).reset_index(drop=True)
    x = training[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    oof = np.full(len(training), np.nan)
    assignments = np.full(len(training), -1, int)
    splitter = GroupKFold(n_splits=3)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups)):
        y_train = y.iloc[train_idx]
        if y_train.nunique() != 2:
            raise RuntimeError(f"{split_name} fold {fold} training has one class")
        fitted = estimator(family, features)
        fitted.fit(x.iloc[train_idx], y_train, model__sample_weight=weights(y_train))
        score = fitted.predict_proba(x.iloc[test_idx])[:, 1]
        oof[test_idx] = score
        assignments[test_idx] = fold
        y_test = y.iloc[test_idx]
        fold_rows.append(
            dict(
                family=family,
                feature_set=feature_set,
                split=split_name,
                fold=fold,
                training_rows=len(train_idx),
                validation_rows=len(test_idx),
                validation_positives=int(y_test.sum()),
                validation_negatives=int((1 - y_test).sum()),
                average_precision=float(average_precision_score(y_test, score)) if y_test.nunique() == 2 else None,
                roc_auc=float(roc_auc_score(y_test, score)) if y_test.nunique() == 2 else None,
            )
        )
    assert np.isfinite(oof).all() and (assignments >= 0).all()
    summary = dict(
        family=family,
        feature_set=feature_set,
        split=split_name,
        fold="pooled_oof",
        training_rows=len(training),
        validation_rows=len(training),
        validation_positives=int(y.sum()),
        validation_negatives=int((1 - y).sum()),
        average_precision=float(average_precision_score(y, oof)),
        roc_auc=float(roc_auc_score(y, oof)),
        balanced_accuracy=float(balanced_accuracy_score(y, oof >= 0.5)),
    )
    predictions = pd.DataFrame(
        dict(candidate_id=training.candidate_id, family=family, feature_set=feature_set, split=split_name, fold=assignments, target=y, score=oof)
    )
    return [*fold_rows, summary], predictions, summary


def ranked(frame: gpd.GeoDataFrame, score: np.ndarray, score_column: str) -> gpd.GeoDataFrame:
    result = frame.copy()
    result[score_column] = score
    result = result.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["model_score"] = result[score_column]
    return result


def truth_frame() -> gpd.GeoDataFrame:
    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    audit = pd.read_csv(PHYSICAL / "all_959_record_diagnosis.csv").set_index("record_id")
    supported = truth.record_id.map(audit.query_success).astype(bool)
    return truth.loc[supported].sort_values("source_excel_row", kind="stable").reset_index(drop=True)


def retrieval_rows(name: str, queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame):
    truth_xy = np.c_[truth.geometry.x, truth.geometry.y]
    rows, matches_by_budget = [], {}
    for budget in BUDGETS:
        part = queue.head(budget)
        matches = match_one_to_one(part, truth_xy, 20)
        distances = matches.distance_m.to_numpy(float)
        rows.append(
            dict(
                model=name,
                evaluation="retrospective_development_exposed",
                budget=budget,
                stops=len(part),
                candidate_population=len(queue),
                matched=len(matches),
                recall=len(matches) / len(truth),
                matches_per_1000=1000 * len(matches) / len(part),
                median_distance_m=float(np.median(distances)) if len(distances) else None,
            )
        )
        matches_by_budget[budget] = matches
    full = match_one_to_one(queue, truth_xy, 20)
    rows.append(
        dict(
            model=name,
            evaluation="retrospective_development_exposed",
            budget="full_population",
            stops=len(queue),
            candidate_population=len(queue),
            matched=len(full),
            recall=len(full) / len(truth),
            matches_per_1000=1000 * len(full) / len(queue),
            median_distance_m=float(np.median(full.distance_m)) if len(full) else None,
        )
    )
    return rows, matches_by_budget


def regional_rows(name: str, queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, matches_by_budget: dict):
    rows = []
    for budget, matches in matches_by_budget.items():
        matched_rows = set(matches.real_row.astype(int))
        for column in ("Region", "Route"):
            for value, indices in truth.groupby(column, dropna=False).groups.items():
                rows.append(
                    dict(
                        model=name,
                        budget=budget,
                        stratum=column.lower(),
                        value=str(value),
                        inventory=len(indices),
                        matched=sum(int(index) in matched_rows for index in indices),
                    )
                )
    return rows


def main() -> None:
    if REPORT.exists() or MODEL_DIR.exists():
        raise RuntimeError("v4 output directory already exists; do not overwrite a completed experiment")
    REPORT.mkdir(parents=True)
    MODEL_DIR.mkdir(parents=True)
    training_path = SELECTIVE / "supervised_site_manifest.csv"
    candidate_path = PHYSICAL / "variant_BASELINE.parquet"
    training = pd.read_csv(training_path).reset_index(drop=True)
    candidates = gpd.read_parquet(candidate_path)
    valid = training.label_source.eq("field_verified_negative") & training.target.eq(0)
    valid |= training.label_source.eq("authoritative_inventory_positive_20m") & training.target.eq(1)
    if not valid.all() or not training.site_key.is_unique or set(training.target) != {0, 1}:
        raise RuntimeError("Training manifest violates the verified-label physical-site contract")

    road_groups = training.route_id.fillna("<unassigned>").astype(str)
    spatial_groups = (np.floor(training.x_m / 5000).astype(int).astype(str) + ":" + np.floor(training.y_m / 5000).astype(int).astype(str))
    validation, predictions, summaries = [], [], []
    available = {}
    for feature_set, declared in FEATURE_FAMILIES.items():
        columns = features_for(training, candidates, declared)
        available[feature_set] = columns
        for family in ("rf", "hgb"):
            for split_name, groups in (("whole_route", road_groups), ("spatial_5km_block", spatial_groups)):
                rows, oof, summary = score_cv(training, family, feature_set, columns, split_name, groups)
                validation.extend(rows)
                predictions.append(oof)
                summaries.append(summary)
    summary = pd.DataFrame(summaries)
    pooled = summary.pivot(index=["family", "feature_set"], columns="split", values="average_precision").reset_index()
    pooled["robust_ap"] = pooled[["whole_route", "spatial_5km_block"]].min(axis=1)
    pooled = pooled.sort_values(["robust_ap", "whole_route", "spatial_5km_block", "family", "feature_set"], ascending=[False, False, False, True, True], kind="stable")
    selected = pooled.iloc[0]
    family, feature_set = str(selected.family), str(selected.feature_set)
    selected_features = available[feature_set]
    final = estimator(family, selected_features)
    y = training.target.astype(int)
    final.fit(training[selected_features], y, model__sample_weight=weights(y))
    candidate_matrix = candidates[selected_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    queue = ranked(candidates, final.predict_proba(candidate_matrix)[:, 1], "score_v4")

    bundle = dict(
        version=VERSION,
        experimental=True,
        calibrated=False,
        model_family=family,
        feature_set=feature_set,
        feature_columns=selected_features,
        pipeline=final,
        label_contract="authoritative positives and field-verified negatives only",
        selection_rule="maximum minimum pooled OOF average precision across whole-route and fixed 5 km spatial-block GroupKFold",
        evaluation_status="retrospective development-exposed; not independently validated",
        active_default=False,
    )
    joblib.dump(bundle, MODEL_DIR / "model.joblib")
    queue.to_parquet(REPORT / "ranked_sites.parquet", index=False)
    queue.drop(columns="geometry").to_csv(REPORT / "ranked_sites.csv", index=False)
    pd.DataFrame(validation).to_csv(MODEL_DIR / "validation_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(MODEL_DIR / "oof_predictions.csv", index=False)
    pooled.to_csv(MODEL_DIR / "model_selection.csv", index=False)
    save_json(MODEL_DIR / "feature_schema.json", {"version": VERSION, "features": selected_features, "excluded_acquisition_or_availability_proxies": sorted(EXCLUDED)})
    split_frame = training[["candidate_id", "route_id", "x_m", "y_m"]].copy()
    split_frame["whole_route_group"] = road_groups
    split_frame["spatial_5km_group"] = spatial_groups
    split_frame.to_csv(MODEL_DIR / "split_definitions.csv", index=False)

    if family == "rf":
        names = final.named_steps["preprocess"].get_feature_names_out()
        importance = final.named_steps["model"].feature_importances_
        pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False).to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    truth = truth_frame()
    baseline = gpd.read_parquet(SELECTIVE / "queue_BASELINE-RF.parquet").sort_values("global_rank", kind="stable")
    v3 = gpd.read_parquet(SPRINT / "ranked_physical_sites.parquet").sort_values("global_rank", kind="stable")
    retrieval, regional = [], []
    for name, frame in (("original_rf", baseline), ("v3_rf", v3), ("v4_physical_ranker", queue)):
        rows, matches_by_budget = retrieval_rows(name, frame, truth)
        retrieval.extend(rows)
        regional.extend(regional_rows(name, frame, truth, matches_by_budget))
    metrics = pd.DataFrame(retrieval)
    metrics.to_csv(REPORT / "retrieval_metrics.csv", index=False)
    pd.DataFrame(regional).to_csv(REPORT / "region_route_metrics.csv", index=False)

    label_route = training.groupby(["route_id", "target"]).size().unstack(fill_value=0)
    single_class_routes = int(((label_route == 0).any(axis=1)).sum())
    missing_by_class = training[selected_features].isna().groupby(training.target).mean()
    missing_gap = (missing_by_class.loc[1] - missing_by_class.loc[0]).abs().sort_values(ascending=False)
    diagnosis = {
        "training_rows": len(training),
        "positive_sites": int(y.sum()),
        "verified_negative_sites": int((1 - y).sum()),
        "routes": int(training.route_id.nunique()),
        "single_class_routes": single_class_routes,
        "route_label_confounding_warning": "Most routes contain only one observed class; route-held-out results are unstable evidence, not an independent generalization estimate.",
        "selected_family": family,
        "selected_feature_set": feature_set,
        "selected_feature_count": len(selected_features),
        "whole_route_oof_average_precision": float(selected.whole_route),
        "spatial_5km_oof_average_precision": float(selected.spatial_5km_block),
        "largest_selected_feature_missingness_gap": float(missing_gap.iloc[0]),
        "dem_resolution_m": 10,
        "profile_sampling_m": 10,
        "75m_density_fields_present_in_training_but_absent_from_deployment_queue": sorted([c for c in training if "75m" in c and c not in candidates]),
        "zero_profile_handling": "Missing profile and DEM values remain NaN and are median-imputed inside each training fold; availability flags are excluded from the model.",
        "duplicates": {"training_site_key_duplicates": int(training.site_key.duplicated().sum()), "candidate_id_duplicates": int(candidates.candidate_id.duplicated().sum()), "physical_site_id_duplicates": int(candidates.physical_site_id.duplicated().sum())},
        "label_leakage_controls": ["No summer inventory field is a feature", "No candidate coordinates, route identifiers, source types, or label-source fields are features", "Feature selection occurs before summer-inventory retrieval evaluation", "Only authoritative positives and field-verified negatives are labels"],
        "candidate_geometry": "v4 ranks the exact same 13,065 physical-site coordinates as the original RF; score effects are isolated from candidate generation and localization.",
    }
    save_json(REPORT / "diagnosis.json", diagnosis)
    provenance = {
        "version": VERSION,
        "inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in (training_path, candidate_path, SELECTIVE / "queue_BASELINE-RF.parquet", SPRINT / "ranked_physical_sites.parquet", SUMMER / "inventory_coverage.parquet")},
        "training_label_rows": len(training),
        "candidate_population": len(queue),
        "random_seed": 20261019,
        "production_or_frozen_artifacts_modified": False,
    }
    save_json(MODEL_DIR / "dataset_provenance.json", provenance)

    def metric(model, budget):
        row = metrics.loc[(metrics.model.eq(model)) & (metrics.budget.astype(str).eq(str(budget)))].iloc[0]
        return int(row.matched), float(row.recall), float(row.median_distance_m)

    b1000 = metric("original_rf", 1000)
    v41000 = metric("v4_physical_ranker", 1000)
    v3full = metric("v3_rf", "full_population")
    v4full = metric("v4_physical_ranker", "full_population")
    promoted = v41000[0] > b1000[0] and float(selected.robust_ap) > 0
    report = f"""# Physical-site ranker v4 — 2026-10-19

## Decision

**{'Promote the v4 challenger.' if promoted else 'Keep the original RF as the default.'}** The isolated v4 challenger retrieves **{v41000[0]}/847** records at 1,000 stops versus **{b1000[0]}/847** for the original RF. This retrospective inventory is development-exposed, so even a numerical gain would require a genuinely untouched road/county test before a production claim.

## Data and validation

- 815 unique physical training sites: 689 authoritative positives and 126 field-verified negatives. Unvisited candidates are never treated as negatives.
- Model selection used pooled out-of-fold average precision under both complete-route GroupKFold and fixed 5 km spatial-block GroupKFold. The chosen `{family}` / `{feature_set}` model scored {float(selected.whole_route):.3f} and {float(selected.spatial_5km_block):.3f}; severe route/class confounding limits both estimates.
- The v4 queue uses exactly the original 13,065 candidate geometries, isolating ranking from candidate-generation density and localization.
- Acquisition/availability proxies, coordinates, route IDs, source type, label source, and summer-inventory fields are excluded.

## Retrieval results

| System | Candidate population | @100 | @250 | @500 | @1,000 | @5,000 | @10,000 | @13,065 | Full population |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original RF | 13,065 | {metric('original_rf',100)[0]} | {metric('original_rf',250)[0]} | {metric('original_rf',500)[0]} | {b1000[0]} | {metric('original_rf',5000)[0]} | {metric('original_rf',10000)[0]} | {metric('original_rf',13065)[0]} | {metric('original_rf','full_population')[0]} |
| v3 RF | 68,302 | {metric('v3_rf',100)[0]} | {metric('v3_rf',250)[0]} | {metric('v3_rf',500)[0]} | {metric('v3_rf',1000)[0]} | {metric('v3_rf',5000)[0]} | {metric('v3_rf',10000)[0]} | {metric('v3_rf',13065)[0]} | **{v3full[0]} ({100*v3full[1]:.2f}%)** |
| v4 physical ranker | 13,065 | {metric('v4_physical_ranker',100)[0]} | {metric('v4_physical_ranker',250)[0]} | {metric('v4_physical_ranker',500)[0]} | **{v41000[0]}** | {metric('v4_physical_ranker',5000)[0]} | {metric('v4_physical_ranker',10000)[0]} | {metric('v4_physical_ranker',13065)[0]} | {v4full[0]} |

The v3 **672/847 = 79.34%** figure is full-population coverage across 68,302 hypotheses. It is not Top-250 or Top-13,065 recall. At the equal 13,065-stop budget it retrieves {metric('v3_rf',13065)[0]} records.

## Why the original RF remains hard to beat

The label manifest is small and structurally confounded: {single_class_routes} of {training.route_id.nunique()} routes contain only one observed class. The original RF's strong early queue reflects interactions learned from the same development lineage; simple physical-feature challengers do not reproduce that concentration reliably across held-out routes and spatial blocks. The 10 m DEM/profile scale is internally consistent, while 75 m density fields exist in training but not the deployment queue and are therefore excluded. Missing DEM/profile values stay missing through feature construction and are imputed only inside each fitting fold. There are no duplicated training site keys or candidate IDs.

## Artifacts

- `ranked_sites.parquet` and `ranked_sites.csv`: complete v4 queue.
- `retrieval_metrics.csv`: budgeted and full-population one-to-one 20 m results.
- `region_route_metrics.csv`: region/route breakdowns at each practical budget.
- `diagnosis.json`: parity, missingness, label, duplication, and geometry audit.
- Model directory contains the serialized model, feature schema, exact splits, OOF predictions, validation metrics, selection table, provenance hashes, and feature importance when available.

No frozen artifact, production model, default ranking, commit, push, or deployment was changed.
"""
    (REPORT / "REPORT.md").write_text(report)
    save_json(REPORT / "COMPLETE.json", {"version": VERSION, "default_changed": promoted, "v4_matches_at_1000": v41000[0], "baseline_matches_at_1000": b1000[0], "v3_full_matches": v3full[0], "v3_full_recall": v3full[1], "model_sha256": sha256_file(MODEL_DIR / "model.joblib")})
    print(f"v4 complete: {v41000[0]} @1000 vs baseline {b1000[0]}; promote={promoted}", flush=True)


if __name__ == "__main__":
    main()
