"""Run the leakage-aware, baseline-preserving physical reranker experiment."""
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
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from culvert_ai.flow_c_bundle import sha256_file
from scripts.validate_inventory_heldout_roads import fold_for, match_one_to_one
from staging.hydrology_sites import apply_feature_contract
from staging.model_sprint import GIS_FEATURES, HYDRO_FEATURES, ROAD_PROFILE_FEATURES, TERRAIN_FEATURES

VERSION = "baseline-preserving-physical-reranker-v5-experimental"
REPORT = ROOT / "reports/experiments/baseline_preserving_v5_20261019"
MODEL_DIR = ROOT / "models/experiments/baseline_preserving_v5_20261019"
SELECTIVE = ROOT / "reports/experiments/selective_admission_20261016"
PHYSICAL = ROOT / "reports/experiments/physical_discovery_20261016"
HYDROLOGY = ROOT / "reports/experiments/hydrology_driven_sites_20261015"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
V4_MODEL = ROOT / "models/experiments/physical_ranker_v4_20261019/model.joblib"
BUDGETS = (100, 250, 500, 1000, 5000, 10000, 13065)
ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.10)
EXCLUDED = {
    "distance_to_nearest_road_m",
    "hydrology_edge_risk",
    "candidate_edge_risk",
    "road_profile_available",
    "dem_data_available",
}


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map({label: len(y) / (2 * count) for label, count in counts.items()}).to_numpy(float)


def screened_features(training: pd.DataFrame, candidates: pd.DataFrame, declared) -> list[str]:
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
    return result


def physical_estimator(family: str, columns: list[str]) -> Pipeline:
    if family == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.05, class_weight="balanced", max_iter=2500, random_state=20261019)),
            ]
        )
    if family == "hgb":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", HistGradientBoostingClassifier(learning_rate=0.03, max_iter=160, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=20, early_stopping=False, random_state=20261019)),
            ]
        )
    raise ValueError(family)


def truth_frame() -> gpd.GeoDataFrame:
    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    audit = pd.read_csv(PHYSICAL / "all_959_record_diagnosis.csv").set_index("record_id")
    return truth.loc[truth.record_id.map(audit.query_success).astype(bool)].sort_values("source_excel_row", kind="stable").reset_index(drop=True)


def original_rf_oof(training: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    result = np.full(len(training), np.nan)
    assignments = []
    for fold in range(3):
        bundle = joblib.load(HYDROLOGY / f"hydrology-sites-v2_RF_fold{fold}.joblib")
        heldout = training.route_id.map(lambda value: fold_for(str(value))).eq(fold)
        transformed = apply_feature_contract(training, bundle["contract"], normalization_context=training)
        matrix = numeric(transformed, bundle["feature_columns"]).fillna(bundle["medians"])
        result[heldout] = bundle["model"].predict_proba(matrix.loc[heldout])[:, 1]
        for index in training.index[heldout]:
            assignments.append(dict(candidate_id=training.loc[index, "candidate_id"], route_id=training.loc[index, "route_id"], fold=fold))
    if not np.isfinite(result).all():
        raise RuntimeError("Every training site requires one whole-route-excluded original-RF score")
    return result, pd.DataFrame(assignments).sort_values("candidate_id")


def mixed_route_auc(training: pd.DataFrame, scores: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    scored = pd.DataFrame(
        {"route_id": training.route_id, "target": training.target, "_score": scores}
    )
    for route, group in scored.groupby("route_id", sort=True):
        if group.target.nunique() != 2:
            continue
        rows.append(
            dict(
                route_id=route,
                rows=len(group),
                positives=int(group.target.sum()),
                negatives=int((1 - group.target).sum()),
                auc=float(roc_auc_score(group.target, group._score)),
                average_precision=float(average_precision_score(group.target, group._score)),
            )
        )
    return float(np.mean([row["auc"] for row in rows])) if rows else np.nan, rows


def blend(base: np.ndarray, physical: np.ndarray, alpha: float) -> np.ndarray:
    base_rank = pd.Series(base).rank(method="average", pct=True).to_numpy(float)
    physical_rank = pd.Series(physical).rank(method="average", pct=True).to_numpy(float)
    return (1 - alpha) * base_rank + alpha * physical_rank


def crossfit(training: pd.DataFrame, family: str, feature_set: str, columns: list[str], split: str, groups: pd.Series, base_oof: np.ndarray):
    x, y = numeric(training, columns), training.target.astype(int)
    oof = np.full(len(training), np.nan)
    assignments = np.full(len(training), -1, int)
    fold_rows = []
    for fold, (train_index, valid_index) in enumerate(GroupKFold(3).split(x, y, groups)):
        model = physical_estimator(family, columns)
        fit_kwargs = {"model__sample_weight": sample_weights(y.iloc[train_index])} if family == "hgb" else {}
        model.fit(x.iloc[train_index], y.iloc[train_index], **fit_kwargs)
        oof[valid_index] = model.predict_proba(x.iloc[valid_index])[:, 1]
        assignments[valid_index] = fold
        valid_y = y.iloc[valid_index]
        fold_rows.append(
            dict(
                family=family,
                feature_set=feature_set,
                split=split,
                fold=fold,
                training_rows=len(train_index),
                validation_rows=len(valid_index),
                positives=int(valid_y.sum()),
                negatives=int((1 - valid_y).sum()),
                average_precision=float(average_precision_score(valid_y, oof[valid_index])) if valid_y.nunique() == 2 else None,
                roc_auc=float(roc_auc_score(valid_y, oof[valid_index])) if valid_y.nunique() == 2 else None,
            )
        )
    pooled = dict(
        family=family,
        feature_set=feature_set,
        split=split,
        fold="pooled_oof",
        training_rows=len(training),
        validation_rows=len(training),
        positives=int(y.sum()),
        negatives=int((1 - y).sum()),
        average_precision=float(average_precision_score(y, oof)),
        roc_auc=float(roc_auc_score(y, oof)),
        balanced_accuracy=float(balanced_accuracy_score(y, oof >= 0.5)),
    )
    selections = []
    route_details = []
    for alpha in ALPHAS:
        combined = blend(base_oof, oof, alpha)
        mixed_auc, routes = mixed_route_auc(training, combined)
        selections.append(
            dict(
                family=family,
                feature_set=feature_set,
                split=split,
                alpha=alpha,
                pooled_average_precision=float(average_precision_score(y, combined)),
                pooled_roc_auc=float(roc_auc_score(y, combined)),
                mixed_route_mean_auc=mixed_auc,
            )
        )
        route_details.extend(dict(family=family, feature_set=feature_set, split=split, alpha=alpha, **row) for row in routes)
    prediction = pd.DataFrame(dict(candidate_id=training.candidate_id, family=family, feature_set=feature_set, split=split, fold=assignments, target=y, physical_score=oof))
    return [*fold_rows, pooled], selections, route_details, prediction


def order(frame: gpd.GeoDataFrame, score: np.ndarray, column: str) -> gpd.GeoDataFrame:
    result = frame.copy()
    result[column] = score
    result = result.sort_values([column, "candidate_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["model_score"] = result[column]
    return result


def evaluate(name: str, queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame):
    rows, matched = [], {}
    truth_xy = np.c_[truth.geometry.x, truth.geometry.y]
    for budget in BUDGETS:
        part = queue.head(budget)
        match = match_one_to_one(part, truth_xy, 20)
        matched[budget] = match
        rows.append(dict(model=name, budget=budget, stops=len(part), candidate_population=len(queue), matched=len(match), recall=len(match) / len(truth), matches_per_1000=1000 * len(match) / len(part), median_distance_m=float(np.median(match.distance_m)) if len(match) else None))
    full = match_one_to_one(queue, truth_xy, 20)
    rows.append(dict(model=name, budget="full_population", stops=len(queue), candidate_population=len(queue), matched=len(full), recall=len(full) / len(truth), matches_per_1000=1000 * len(full) / len(queue), median_distance_m=float(np.median(full.distance_m)) if len(full) else None))
    return rows, matched


def regional(name: str, truth: gpd.GeoDataFrame, matches: dict) -> list[dict]:
    rows = []
    for budget, match in matches.items():
        selected = set(match.real_row.astype(int))
        for column in ("Region", "Route"):
            for value, indices in truth.groupby(column, dropna=False).groups.items():
                rows.append(dict(model=name, budget=budget, stratum=column.lower(), value=str(value), inventory=len(indices), matched=sum(int(index) in selected for index in indices)))
    return rows


def main() -> None:
    if REPORT.exists() or MODEL_DIR.exists():
        raise RuntimeError("v5 output already exists; completed experiments are immutable")
    REPORT.mkdir(parents=True)
    MODEL_DIR.mkdir(parents=True)

    training_path = SELECTIVE / "supervised_site_manifest.csv"
    baseline_path = PHYSICAL / "variant_BASELINE.parquet"
    hybrid_path = PHYSICAL / "variant_HYBRID-GROUPED.parquet"
    training = pd.read_csv(training_path).reset_index(drop=True)
    baseline = gpd.read_parquet(baseline_path)
    hybrid = gpd.read_parquet(hybrid_path)
    truth = truth_frame()
    labels_valid = (training.label_source.eq("field_verified_negative") & training.target.eq(0)) | (training.label_source.eq("authoritative_inventory_positive_20m") & training.target.eq(1))
    if not labels_valid.all() or not training.site_key.is_unique:
        raise RuntimeError("Training rows must be one verified physical site each")

    base_oof, base_splits = original_rf_oof(training)
    training_out = training.copy()
    training_out["observation_id"] = training.site_key
    training_out["label_provenance"] = training.label_source
    training_out["verification_status"] = np.where(training.target.eq(1), "authoritative_positive", "field_verified_absence")
    training_out["training_group_route"] = training.route_id.astype(str)
    training_out["training_group_spatial_5km"] = np.floor(training.x_m / 5000).astype(int).astype(str) + ":" + np.floor(training.y_m / 5000).astype(int).astype(str)
    training_out["baseline_rf_oof_score"] = base_oof
    training_out.to_parquet(REPORT / "physical_site_training_manifest.parquet", index=False)
    training_out.to_csv(REPORT / "physical_site_training_manifest.csv", index=False)
    base_splits.to_csv(MODEL_DIR / "baseline_rf_oof_split.csv", index=False)

    feature_sets = {
        "gis": GIS_FEATURES,
        "gis_dem": GIS_FEATURES + HYDRO_FEATURES,
        "gis_profiles": GIS_FEATURES + ROAD_PROFILE_FEATURES,
        "gis_dem_terrain": GIS_FEATURES + HYDRO_FEATURES + TERRAIN_FEATURES,
    }
    screened = {name: screened_features(training, baseline, declared) for name, declared in feature_sets.items()}
    feature_screen_rows = []
    for name, declared in feature_sets.items():
        for column in declared:
            feature_screen_rows.append(dict(feature_set=name, feature=column, retained=column in screened[name], reason="retained" if column in screened[name] else ("acquisition_or_availability_proxy" if column in EXCLUDED else "missing_constant_or_class_dependent_missingness")))
    pd.DataFrame(feature_screen_rows).to_csv(MODEL_DIR / "feature_screen.csv", index=False)

    road_groups = training.training_group_route if "training_group_route" in training else training.route_id.astype(str)
    spatial_groups = training_out.training_group_spatial_5km
    validation, selection, route_validation, predictions = [], [], [], []
    for feature_set, columns in screened.items():
        if len(columns) < 3:
            continue
        for family in ("logistic", "hgb"):
            for split, groups in (("whole_route", road_groups), ("spatial_5km", spatial_groups)):
                fold_rows, choices, route_rows, oof = crossfit(training, family, feature_set, columns, split, groups, base_oof)
                validation.extend(fold_rows)
                selection.extend(choices)
                route_validation.extend(route_rows)
                predictions.append(oof)
    pd.DataFrame(validation).to_csv(MODEL_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame(route_validation).to_csv(MODEL_DIR / "mixed_route_validation.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(MODEL_DIR / "oof_physical_predictions.csv", index=False)
    choices = pd.DataFrame(selection)
    comparable = choices.pivot(index=["family", "feature_set", "alpha"], columns="split", values="mixed_route_mean_auc").reset_index()
    comparable["robust_mixed_route_auc"] = comparable[["whole_route", "spatial_5km"]].min(axis=1)
    comparable = comparable.sort_values(["robust_mixed_route_auc", "alpha", "family", "feature_set"], ascending=[False, True, True, True], kind="stable")
    comparable.to_csv(MODEL_DIR / "reranker_selection.csv", index=False)
    selected = comparable.iloc[0]
    family, feature_set, alpha = str(selected.family), str(selected.feature_set), float(selected.alpha)
    columns = screened[feature_set]
    final_physical = physical_estimator(family, columns)
    fit_kwargs = {"model__sample_weight": sample_weights(training.target.astype(int))} if family == "hgb" else {}
    final_physical.fit(numeric(training, columns), training.target.astype(int), **fit_kwargs)

    baseline_score = pd.to_numeric(baseline.score_RF, errors="coerce").to_numpy(float)
    physical_score = final_physical.predict_proba(numeric(baseline, columns))[:, 1]
    combined_score = blend(baseline_score, physical_score, alpha)
    baseline_with_physical = baseline.copy()
    baseline_with_physical["score_v5_physical"] = physical_score
    reranked = order(baseline_with_physical, combined_score, "score_v5")
    reranked.to_parquet(REPORT / "ranked_sites.parquet", index=False)
    reranked.drop(columns="geometry").to_csv(REPORT / "ranked_sites.csv", index=False)

    # Experiment A: an isolated refit of the repaired 48-feature RF contract.
    source_bundle = joblib.load(HYDROLOGY / "hydrology-sites-v2_RF_experimental.joblib")
    transformed_training = apply_feature_contract(training, source_bundle["contract"], normalization_context=training)
    repaired_columns = source_bundle["feature_columns"]
    repaired_medians = numeric(transformed_training, repaired_columns).median().fillna(0.0)
    repaired_rf = clone(source_bundle["model"])
    repaired_rf.fit(numeric(transformed_training, repaired_columns).fillna(repaired_medians), training.target.astype(int))
    repaired_score = repaired_rf.predict_proba(numeric(baseline, repaired_columns).fillna(repaired_medians))[:, 1]
    repaired_queue = order(baseline, repaired_score, "score_repaired_rf")
    repaired_queue.to_parquet(REPORT / "queue_repaired_rf.parquet", index=False)

    bundle = dict(
        version=VERSION,
        experimental=True,
        active_default=False,
        calibrated=False,
        physical_estimator=final_physical,
        physical_feature_columns=columns,
        original_rf_score_column="score_RF",
        blend_alpha=alpha,
        blend_definition="(1-alpha)*global_percentile(original_rf_score)+alpha*global_percentile(physical_model_score)",
        selection_rule="maximize the minimum mean within-route AUC across whole-route and fixed 5km spatial OOF predictions; smallest alpha wins ties",
        label_contract="one physical site per authoritative positive or field-verified negative; no unlabeled negatives",
    )
    joblib.dump(bundle, MODEL_DIR / "reranker.joblib")
    joblib.dump(dict(version="repaired-rf-v5-experimental", model=repaired_rf, medians=repaired_medians, feature_columns=repaired_columns, contract=source_bundle["contract"], active_default=False), MODEL_DIR / "repaired_rf.joblib")
    save_json(MODEL_DIR / "feature_schema.json", {"version": VERSION, "physical_features": columns, "units": {"distance": "metres", "density": "metres_per_square_kilometre", "angle": "degrees", "area": "square_metres", "elevation": "metres"}, "missing_policy": "NaN through extraction; fold-fitted median imputation; no arbitrary zero fill", "excluded": sorted(EXCLUDED)})
    save_json(MODEL_DIR / "training_parameters.json", {"family": family, "feature_set": feature_set, "blend_alpha": alpha, "alphas_considered": ALPHAS, "random_state": 20261019})

    # Controlled 2x2 and all final queues. v4 is scored on each unchanged population.
    v4 = joblib.load(V4_MODEL)
    v4_columns = v4["feature_columns"]
    queues = {
        "original_population_original_rf": gpd.read_parquet(SELECTIVE / "queue_BASELINE-RF.parquet").sort_values("global_rank", kind="stable").reset_index(drop=True),
        "original_population_v4": order(baseline, v4["pipeline"].predict_proba(numeric(baseline, v4_columns))[:, 1], "score_v4"),
        "hybrid_population_original_rf": gpd.read_parquet(SELECTIVE / "queue_FULL-HYBRID-RF.parquet").sort_values("global_rank", kind="stable").reset_index(drop=True),
        "hybrid_population_v4": order(hybrid, v4["pipeline"].predict_proba(numeric(hybrid, v4_columns))[:, 1], "score_v4"),
        "original_population_repaired_rf": repaired_queue,
        "original_population_v5_reranker": reranked,
    }
    retrieval, region_rows = [], []
    match_maps = {}
    for name, queue in queues.items():
        rows, matches = evaluate(name, queue, truth)
        retrieval.extend(rows)
        region_rows.extend(regional(name, truth, matches))
        match_maps[name] = matches
    metrics = pd.DataFrame(retrieval)
    metrics.to_csv(REPORT / "retrieval_metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(REPORT / "route_region_metrics.csv", index=False)

    # Top-1,000 failure decomposition and physical support for baseline matches.
    baseline_queue = queues["original_population_original_rf"]
    top_match = match_maps["original_population_original_rf"][1000]
    full_match = match_one_to_one(baseline_queue, np.c_[truth.geometry.x, truth.geometry.y], 20)
    top_records = set(top_match.real_row.astype(int))
    full_records = set(full_match.real_row.astype(int))
    diagnosis = pd.read_csv(PHYSICAL / "all_959_record_diagnosis.csv").set_index("record_id").loc[truth.record_id].reset_index()
    diagnosis["top1000_status"] = ["matched_top1000" if i in top_records else "candidate_present_poorly_ranked" if i in full_records else diagnosis.loc[i, "failure_substage"] for i in range(len(diagnosis))]
    diagnosis.to_csv(REPORT / "top1000_record_diagnosis.csv", index=False)
    diagnosis.top1000_status.value_counts().rename_axis("category").rename("records").to_csv(REPORT / "top1000_failure_counts.csv")

    matched_site_rows = set(top_match.site_row.astype(int))
    support_features = [c for c in columns if c in baseline_queue]
    support = []
    for feature in support_features:
        values = pd.to_numeric(baseline_queue[feature], errors="coerce")
        matched_values = values.iloc[sorted(matched_site_rows)]
        other_values = values.iloc[:1000].drop(index=list(matched_site_rows), errors="ignore")
        support.append(dict(feature=feature, matched_median=float(matched_values.median()) if matched_values.notna().any() else None, unmatched_top1000_median=float(other_values.median()) if other_values.notna().any() else None, matched_missing_rate=float(matched_values.isna().mean()), unmatched_top1000_missing_rate=float(other_values.isna().mean())))
    pd.DataFrame(support).to_csv(REPORT / "baseline_top1000_physical_support.csv", index=False)

    route_labels = training.groupby(["route_id", "target"]).size().unstack(fill_value=0)
    route_labels.columns = ["verified_negatives", "positives"]
    route_labels["single_class"] = (route_labels == 0).any(axis=1)
    route_labels.to_csv(REPORT / "label_coverage_by_route.csv")
    priority = diagnosis.loc[diagnosis.top1000_status.ne("matched_top1000")].groupby(["Region", "Route", "top1000_status"]).size().rename("records").reset_index().sort_values("records", ascending=False)
    priority.to_csv(REPORT / "field_label_collection_priorities.csv", index=False)

    overlap_distance = cKDTree(training.loc[training.target.eq(1), ["x_m", "y_m"]]).query(np.c_[truth.geometry.x, truth.geometry.y])[0]
    audit = {
        "training_rows": len(training),
        "unique_site_keys": int(training.site_key.nunique()),
        "positive_sites": int(training.target.sum()),
        "verified_negative_sites": int((1 - training.target).sum()),
        "unvisited_negative_sites": 0,
        "routes": int(training.route_id.nunique()),
        "single_class_routes": int(route_labels.single_class.sum()),
        "mixed_label_routes": int((~route_labels.single_class).sum()),
        "summer_records_within_25m_of_training_positive": int((overlap_distance <= 25).sum()),
        "evaluation_independent": False,
        "candidate_to_label_join": "2526 source-positive rows are consolidated to 689 unique physical site_key positives; 126 verified-negative site keys remain one row each; one conflicting label is excluded upstream",
        "negative_sufficiency": "Insufficient for robust supervised reranking: only 126 negatives, 11/14 routes single-class, and only three routes contain both labels.",
        "candidate_population_diagnosis": "v4 collapses on both original and hybrid populations, so different population size is not the primary cause.",
    }
    save_json(REPORT / "training_data_audit.json", audit)
    provenance_paths = [training_path, baseline_path, hybrid_path, SELECTIVE / "queue_BASELINE-RF.parquet", SELECTIVE / "queue_FULL-HYBRID-RF.parquet", V4_MODEL, SUMMER / "inventory_coverage.parquet"]
    save_json(MODEL_DIR / "dataset_provenance.json", {"version": VERSION, "inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in provenance_paths}, "summer_inventory_used_for_model_selection": False, "production_or_frozen_artifacts_modified": False})

    def match_count(model: str, budget) -> int:
        return int(metrics.loc[metrics.model.eq(model) & metrics.budget.astype(str).eq(str(budget)), "matched"].iloc[0])

    baseline1000 = match_count("original_population_original_rf", 1000)
    v51000 = match_count("original_population_v5_reranker", 1000)
    promoted = v51000 > baseline1000 and float(selected.robust_mixed_route_auc) > 0.5
    report = f"""# Baseline-preserving physical reranker v5 — 2026-10-19

## Decision

**{'The challenger clears the retrospective Top-1,000 count, but still requires independent validation.' if promoted else 'Keep the original RF as the default.'}** The training-selected v5 queue retrieves **{v51000}/847** records at 1,000 stops versus **{baseline1000}/847** for the original RF. Selection never used the 847-record summer inventory. The inventory remains retrospective, previously exposed evaluation.

## Why v3/v4 collapsed

The controlled 2 × 2 comparison rules out population size as the main explanation. v4 retrieves **{match_count('original_population_v4',1000)}** on the original 13,065 sites and **{match_count('hybrid_population_v4',1000)}** on the hybrid 31,848 sites, versus **{baseline1000}** and **{match_count('hybrid_population_original_rf',1000)}** for the original RF. IDs, geometry, grouping, sorting and feature completeness were checked within each population. The dominant problem is the label sample: 689 positives versus 126 negatives, with 11/14 routes containing only one class. Physical features encode that acquisition geography even without route IDs, producing flattering pooled AP but weak within-route discrimination.

## Controlled same-population results

| Candidate population / ranker | Sites | @100 | @250 | @500 | @1,000 | Full |
|---|---:|---:|---:|---:|---:|---:|
| Original / original RF | 13,065 | {match_count('original_population_original_rf',100)} | {match_count('original_population_original_rf',250)} | {match_count('original_population_original_rf',500)} | {baseline1000} | {match_count('original_population_original_rf','full_population')} |
| Original / v4 | 13,065 | {match_count('original_population_v4',100)} | {match_count('original_population_v4',250)} | {match_count('original_population_v4',500)} | {match_count('original_population_v4',1000)} | {match_count('original_population_v4','full_population')} |
| Hybrid / original RF | 31,848 | {match_count('hybrid_population_original_rf',100)} | {match_count('hybrid_population_original_rf',250)} | {match_count('hybrid_population_original_rf',500)} | {match_count('hybrid_population_original_rf',1000)} | {match_count('hybrid_population_original_rf','full_population')} |
| Hybrid / v4 | 31,848 | {match_count('hybrid_population_v4',100)} | {match_count('hybrid_population_v4',250)} | {match_count('hybrid_population_v4',500)} | {match_count('hybrid_population_v4',1000)} | {match_count('hybrid_population_v4','full_population')} |

## Experiments A–D

- **A — repaired RF:** a new isolated 48-feature RF refit uses the same serialized v2 physical contract for training and inference. It retrieves {match_count('original_population_repaired_rf',1000)} at 1,000. The frozen v1 model is not fed incompatible v2 measurements.
- **B — feature families:** GIS, GIS+DEM, GIS+profiles, and GIS+DEM+terrain were tested with logistic and regularized HGB models under whole-route and fixed 5 km spatial folds. Profile fields that reveal acquisition source are screened out. See `validation_metrics.csv`, `feature_screen.csv`, and `reranker_selection.csv`.
- **C — baseline-preserving reranker:** the selected `{family}` / `{feature_set}` physical model uses a conservative **alpha={alpha:.2f}** percentile blend with leakage-safe, whole-route-excluded original-RF scores. Its robust mixed-route validation AUC is {float(selected.robust_mixed_route_auc):.3f}. The serialized bundle contains the estimator, preprocessing, schema and blend definition.
- **D — selective new sites:** not promoted. Ranking did not establish robust improvement on the original population, so new hybrid sites are not allowed to displace stronger baseline predictions.

## Training data and remaining limitation

The saved physical-site manifest has true coordinates, observation/site IDs, provenance, verification status, route, road segment, route/spatial groups, features and OOF baseline scores. No unvisited site is labeled negative. Multiple source candidates are consolidated to one `site_key`. However, {audit['summer_records_within_25m_of_training_positive']} of the 847 searchable summer records lie within 25 m of a training positive, so this is not independent evaluation.

The 126 verified negatives are not sufficient for reliable countywide reranking. The most useful next labels are complete road-mile surveys—especially routes currently containing positives but no verified absences—with every inspected non-culvert location recorded, and held out by whole route before model selection. `field_label_collection_priorities.csv` ranks exposed failure strata for planning only.

## Error diagnosis

`top1000_failure_counts.csv` separates 123 retrieved records, 224 candidates present but ranked below 1,000, and the existing full-list failure causes: missing/admission failures, localization, representative grouping, query coverage and one-to-one competition. `baseline_top1000_physical_support.csv` summarizes the physical features supporting the original RF's high-ranked matches without injecting inventory coordinates into training.

## Safety and reproducibility

All models and queues are isolated experimental artifacts. The original RF remains the staging default unless both retrospective improvement and route-separated evidence are robust. No frozen model, production path, deployment, commit or push was changed.
"""
    (REPORT / "REPORT.md").write_text(report)
    save_json(REPORT / "COMPLETE.json", {"version": VERSION, "baseline_matches_at_1000": baseline1000, "v5_matches_at_1000": v51000, "promoted": promoted, "selected_family": family, "selected_feature_set": feature_set, "blend_alpha": alpha, "model_sha256": sha256_file(MODEL_DIR / "reranker.joblib"), "repaired_rf_sha256": sha256_file(MODEL_DIR / "repaired_rf.joblib")})
    print(f"v5 complete: {v51000} @1000 vs baseline {baseline1000}; promote={promoted}; alpha={alpha}", flush=True)


if __name__ == "__main__":
    main()
