"""Reconcile historical research rankings with the frozen deployed RF pipeline.

This is a read-only evaluation. It loads preserved OOF predictions and the
frozen bundle, but never fits an estimator, builds a feature, or changes the
production application.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle
from culvert_ai.inference import RouteZConfig, rank_flow_c_candidates


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/experiments"
OUT = ROOT / "reports/research/model_lineage_reconciliation_20261014"
FEATURES = ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg"
POOL = EXP / "nysdot_ulster_consensus_precision_20261013/candidate_consensus_features.parquet"
MODEL = ROOT / "models/flow_c_frozen_v1.joblib"
V8 = EXP / "nysdot_ulster_dem_hydrology_v8_20261005"
V18 = EXP / "nysdot_ulster_ranking_v18_20261006"
V19 = EXP / "nysdot_ulster_score_comparability_v19_20261006"
BENCHMARK = EXP / "nysdot_ulster_real_inventory_benchmark_v1_20261006"
RF_OOF = V8 / "hydro_c_plus_flow_c__whole_route__oof_predictions.parquet"
V18_RF_COPY = V18 / "out_of_fold_predictions/flow_c.parquet"
POINT_OOF = V18 / "out_of_fold_predictions/pointwise_gb__authoritative_inventory.parquet"
POINT_ROUTE_Z = V19 / "out_of_fold_scores/route-z.parquet"
EVALUATOR = V18 / "run_experiment.py"
KS = (25, 50, 100, 250)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_evaluator():
    spec = importlib.util.spec_from_file_location("lineage_v18_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_rows(system: str, sites: pd.DataFrame, inventory_xy: np.ndarray, evaluator) -> list[dict]:
    rows = []
    for k in KS:
        matches, _ = evaluator.match_sites(sites, inventory_xy, top_k=k)
        top = sites.head(k)
        route_counts = top["route"].value_counts()
        rows.append(
            {
                "system": system,
                "k": k,
                "recommendations": len(top),
                "distinct_confirmed": len(matches),
                "precision": len(matches) / len(top),
                "site_recall": len(matches) / len(inventory_xy),
                "routes_represented": int(len(route_counts)),
                "largest_route": str(route_counts.index[0]),
                "largest_route_share": float(route_counts.iloc[0] / len(top)),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [
        MODEL,
        MODEL.with_suffix(MODEL.suffix + ".manifest.json"),
        MODEL.with_suffix(MODEL.suffix + ".schema.json"),
        MODEL.with_suffix(MODEL.suffix + ".training_records.csv"),
        FEATURES,
        POOL,
        RF_OOF,
        V18_RF_COPY,
        POINT_OOF,
        POINT_ROUTE_Z,
        EVALUATOR,
        BENCHMARK / "benchmark_metrics.csv",
        ROOT / "src/culvert_ai/inference.py",
    ]
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}

    bundle = load_frozen_flow_c_bundle(MODEL)
    exact_features = json.loads((V8 / "exact_feature_lists.json").read_text(encoding="utf-8"))[
        "hydro_c_plus_flow_c"
    ]
    if list(bundle["feature_columns"]) != list(exact_features):
        raise AssertionError("Frozen bundle feature schema differs from V8 FLOW-C.")

    rf_v8 = pd.read_parquet(RF_OOF).set_index("candidate_id")["probability"].sort_index()
    rf_v18 = (
        pd.read_parquet(V18_RF_COPY).set_index("candidate_id")["outer_oof_score"].sort_index()
    )
    if not rf_v8.index.equals(rf_v18.index) or not rf_v8.equals(rf_v18):
        raise AssertionError("The V18 FLOW-C copy is not the exact V8 deployed-spec OOF vector.")
    if digest(RF_OOF) != bundle["route_z_global"]["source_sha256"]:
        raise AssertionError("The frozen bundle names a different RF OOF source.")

    oof_frame = pd.read_parquet(RF_OOF, columns=["candidate_id", "experiment_route", "probability"])
    oof_frame["candidate_id"] = oof_frame["candidate_id"].astype(str)
    if len(oof_frame) != 51_417 or not oof_frame["candidate_id"].is_unique:
        raise AssertionError("Saved RF OOF artifact is not the 51,417-row evaluation pool.")
    evaluation_pool = pd.read_parquet(
        POOL, columns=["candidate_id", "route", "x_m", "y_m"]
    )
    evaluation_pool["candidate_id"] = evaluation_pool["candidate_id"].astype(str)
    if not evaluation_pool["candidate_id"].is_unique or set(oof_frame["candidate_id"]) != set(
        evaluation_pool["candidate_id"]
    ):
        raise AssertionError("Saved RF OOF candidates cannot be mapped to the frozen evaluation pool.")
    candidates = evaluation_pool.set_index("candidate_id", drop=False).loc[
        oof_frame["candidate_id"]
    ].copy().reset_index(drop=True)
    candidates = gpd.GeoDataFrame(
        candidates,
        geometry=gpd.points_from_xy(candidates["x_m"], candidates["y_m"]),
        crs="EPSG:26918",
    )
    candidates["route_id"] = oof_frame["experiment_route"].fillna("<unassigned>").astype(str).to_numpy()
    candidates["route"] = candidates["route_id"]
    candidates["flow_c_score"] = oof_frame["probability"].to_numpy(float)

    route_z = bundle["route_z_global"]
    config = RouteZConfig(
        global_center=float(route_z["center"]), global_scale=float(route_z["scale"])
    )
    deployed_sites = rank_flow_c_candidates(
        candidates, route_column="route_id", route_z_config=config
    ).rename(columns={"global_rank": "rank", "physical_site_id": "site_id"})
    deployed_sites["route"] = deployed_sites["route_id"]
    deployed_sites["x_m"] = deployed_sites.geometry.x
    deployed_sites["y_m"] = deployed_sites.geometry.y

    point_scores = pd.read_parquet(POINT_OOF).set_index("candidate_id")["outer_oof_score"]
    archived_z = pd.read_parquet(POINT_ROUTE_Z).set_index("candidate_id")["outer_oof_score"]
    point_frame = candidates.copy()
    point_frame["pointwise_score"] = point_frame["candidate_id"].map(point_scores).to_numpy(float)
    point_frame["route_z_score"] = point_frame["candidate_id"].map(archived_z).to_numpy(float)

    # The historical evaluator's suppression is the archived POINTWISE rule:
    # ROUTE-Z descending, candidate ID ascending, then greedy 20 m suppression.
    evaluator = load_evaluator()
    historical_sites = evaluator.nms_sites(point_frame, "route_z_score", "POINTWISE_GB_ROUTE_Z")

    real = evaluator.load_real()
    inventory_xy = real[["x", "y"]].to_numpy(float)
    metrics = pd.DataFrame(
        metric_rows("Historical POINTWISE-GB + ROUTE-Z", historical_sites, inventory_xy, evaluator)
        + metric_rows("Frozen RF FLOW-C + dynamic ROUTE-Z", deployed_sites, inventory_xy, evaluator)
    )
    expected_point = metrics.loc[
        metrics["system"].eq("Historical POINTWISE-GB + ROUTE-Z") & metrics["k"].eq(250),
        "distinct_confirmed",
    ].item()
    if expected_point != 41:
        raise AssertionError(f"Historical POINTWISE 41/250 did not reproduce: {expected_point}")
    metrics.to_csv(OUT / "heldout_ranking_comparison.csv", index=False)

    route_rows = []
    for system, sites in (
        ("Historical POINTWISE-GB + ROUTE-Z", historical_sites),
        ("Frozen RF FLOW-C + dynamic ROUTE-Z", deployed_sites),
    ):
        top = sites.head(250)
        matches, _ = evaluator.match_sites(sites, inventory_xy, top_k=250)
        matched = set(matches["site_row"].astype(int))
        top = top.assign(distinct_match=[index in matched for index in range(len(top))])
        for route, part in top.groupby("route"):
            route_rows.append(
                {
                    "system": system,
                    "route": route,
                    "recommendations": len(part),
                    "distinct_confirmed": int(part["distinct_match"].sum()),
                    "precision": float(part["distinct_match"].mean()),
                    "share": len(part) / len(top),
                }
            )
    pd.DataFrame(route_rows).to_csv(OUT / "top250_route_distribution.csv", index=False)

    benchmark = pd.read_csv(BENCHMARK / "benchmark_metrics.csv")
    coverage = benchmark.loc[benchmark["system"].eq("FLOW-C")].iloc[0]
    if int(coverage["predictions_distinct_20m_sites"]) != 15_650 or int(
        coverage["distinct_real_matches_at_20m"]
    ) != 650:
        raise AssertionError("Historical 650/714 benchmark artifact changed.")

    rf250 = metrics.loc[
        metrics["system"].eq("Frozen RF FLOW-C + dynamic ROUTE-Z") & metrics["k"].eq(250)
    ].iloc[0]
    decision = "RESEARCH / DEPLOYMENT MISMATCH UNRESOLVED"
    audit = {
        "decision": decision,
        "new_models_fit": 0,
        "new_features_built": 0,
        "production_changed": False,
        "decision_reason": (
            "The deployed RF held-out ranking is materially weaker. The historical POINTWISE result is stronger, "
            "but V18 preserved outer-fold-specific, inner-selected estimators rather than one audited/frozen "
            "unseen-road production scorer, so promotion conditions are not yet established."
        ),
        "exact_rf_oof_already_existed": True,
        "rf_oof_source": str(RF_OOF.relative_to(ROOT)),
        "rf_oof_copy": str(V18_RF_COPY.relative_to(ROOT)),
        "rf_oof_rows": len(rf_v8),
        "rf_oof_max_abs_copy_difference": float(np.max(np.abs(rf_v8 - rf_v18))),
        "frozen_model": str(MODEL.relative_to(ROOT)),
        "model_class": bundle["model_class"],
        "model_parameters": bundle["model_parameters"],
        "feature_count": len(bundle["feature_columns"]),
        "feature_columns": list(bundle["feature_columns"]),
        "training_data_provenance": bundle["training_data_provenance"],
        "candidate_generation": bundle["candidate_generation"],
        "physical_site_definition": bundle["physical_site_definition"],
        "dynamic_route_z": {
            "standard": "per-route mean/population standard deviation for routes with >=8 finite scores",
            "small_route": "per-route median and IQR/1.349 for routes with >=4 finite scores",
            "fallback": bundle["route_z_global"],
        },
        "historical_coverage": {
            "raw_candidates": int(coverage["raw_prediction_rows"]),
            "score_ordered_physical_sites": int(coverage["predictions_distinct_20m_sites"]),
            "distinct_one_to_one_matches": int(coverage["distinct_real_matches_at_20m"]),
            "inventory_records": 714,
            "recall": float(coverage["real_culvert_recall_at_20m"]),
            "classification": "post-ranking/post-representative site coverage",
            "artifact": str((BENCHMARK / "benchmark_metrics.csv").relative_to(ROOT)),
        },
        "historical_41_lineage": {
            "scorer": "sklearn.ensemble.HistGradientBoostingClassifier, POINTWISE-GB AUTHORITATIVE-INVENTORY, outer-fold-specific inner-selected configuration",
            "base_score_artifact": str(POINT_OOF.relative_to(ROOT)),
            "normalized_score_artifact": str(POINT_ROUTE_Z.relative_to(ROOT)),
            "final_score_column": "route_z_score",
            "normalization": "per held-out route mean and population standard deviation of unlabeled POINTWISE-GB OOF scores",
            "consolidation": "descending ROUTE-Z, candidate_id ascending tie-break, greedy 20 m suppression",
            "folds": "11 leave-one-complete-route-out outer folds; two inner grouped-route splits select between two predeclared configurations",
            "labels": "authoritative candidates <=20 m as site-balanced positives; >20–40 m excluded; >40 m sampled route-diverse inventory negatives plus independent field-verified negatives",
            "features": "V18 D_ALL_VALID label-free feature set; exact ordered list in experiment_config.json",
        },
    }
    (OUT / "lineage_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    comparison = metrics.loc[metrics["k"].isin([100, 250])].copy()
    lines = [
        "# Research/deployment ranking reconciliation — 2026-10-14",
        "",
        f"**Decision: {decision}.**",
        "",
        "No estimator was fitted, no feature was built, and no production or UI file was changed. Exact saved whole-route OOF predictions already existed for the deployed Random Forest specification.",
        "",
        "## Historical 650/714 lineage",
        "",
        "The 650/714 (91.04%) value is **post-ranking/post-representative site coverage**, not raw candidate-generation coverage and not accuracy. The real-inventory benchmark loaded 51,417 frozen candidates and the V14 `FLOW-C` score column, sorted by score descending with candidate ID ascending, greedily retained one representative per 20 m neighborhood, and produced 15,650 physical prediction sites. Maximum-cardinality one-to-one matching within 20 m linked those representatives to 650 of 714 authoritative records. Because the score order chooses representatives, ranking can affect this coverage value.",
        "",
        "Artifacts: `benchmark_metrics.csv` and `run_experiment.py` in `nysdot_ulster_real_inventory_benchmark_v1_20261006`.",
        "",
        "## Historical 41/250 lineage",
        "",
        "The 41/250 result uses POINTWISE-GB AUTHORITATIVE-INVENTORY OOF scores, not RF FLOW-C scores. Its scorer is `sklearn.ensemble.HistGradientBoostingClassifier`; each of 11 complete routes is held out once, while two inner grouped-route splits choose between the predeclared PW-COMPACT and PW-WIDE settings. Training uses site-balanced authoritative positives within 20 m, excludes the 20–40 m buffer, and uses sampled route-diverse inventory negatives beyond 40 m plus independent field-verified negatives. Features are the exact V18 `D_ALL_VALID` label-free set. The held-out route's unlabeled score distribution is normalized by mean and population standard deviation. Global order is ROUTE-Z descending with candidate ID as the tie-break, followed by greedy 20 m physical-site suppression.",
        "",
        "Artifacts: `pointwise_gb__authoritative_inventory.parquet`, V19 `route-z.parquet`, V18 `experiment_config.json`, `nested_validation.csv`, and per-route model artifacts.",
        "",
        "## Exact deployed RF pipeline",
        "",
        f"The app uses `{MODEL.relative_to(ROOT)}`: `{bundle['model_class']}`, 800 trees, balanced-subsample class weighting, minimum leaf size 2, seed 42, and {len(bundle['feature_columns'])} ordered numeric FLOW-C features. It uses full-training medians for missing values. New roads are sampled at 37.5 m then every 75 m with no lateral offsets; route identifiers only define unlabeled normalization groups. Dynamic ROUTE-Z uses route mean/population standard deviation for normal routes, median/IQR for small routes, and the frozen global OOF center/scale only as fallback. Candidates are ordered by ROUTE-Z descending, RF score descending, then candidate ID ascending; greedy 20 m suppression retains physical-site representatives. Final priority is ROUTE-Z.",
        "",
        "## Held-out ranking comparison",
        "",
        "| System | Confirmed@100 | Confirmed@250 | Precision@100 | Precision@250 | Site recall@250 | Routes@250 | Largest-route share@250 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, part in comparison.groupby("system", sort=False):
        row100 = part.loc[part["k"].eq(100)].iloc[0]
        row250 = part.loc[part["k"].eq(250)].iloc[0]
        lines.append(
            f"| {system} | {row100['distinct_confirmed']} | {row250['distinct_confirmed']} | {row100['precision']:.2%} | {row250['precision']:.2%} | {row250['site_recall']:.2%} | {row250['routes_represented']} | {row250['largest_route_share']:.1%} |"
        )
    lines += [
        "",
        "The RF values above are the first direct reconstruction of the app's scorer, dynamic ROUTE-Z, and 20 m ordering from exact saved held-out predictions. They are not training metrics. Evaluation uses the preserved EPSG:26918 benchmark coordinates; production performs the same metre-based 20 m rule in EPSG:32618.",
        "",
        "## Correct weekly-report language",
        "",
        "**650/714:** “In the historical real-inventory benchmark, FLOW-C score-ordered 20 m consolidation reduced 51,417 candidates to 15,650 physical-site representatives; one-to-one matching within 20 m covered 650 of 714 authoritative culverts (91.04% post-representative site recall). This is not raw candidate-generation coverage or accuracy, and the score-ordered representative choice affects it.”",
        "",
        "**41/250:** “The archived best normalized global ranking found 41 distinct authoritative culverts in its first 250 physical-site recommendations (16.4% precision; 5.74% site recall). Its base scorer was the V18 POINTWISE-GB AUTHORITATIVE-INVENTORY model, normalized per held-out route; it was not the deployed Random Forest FLOW-C scorer.”",
        "",
        f"**Current app:** “Using exact saved whole-route OOF predictions for the frozen RF FLOW-C specification, followed by the app's dynamic ROUTE-Z and greedy 20 m consolidation, the deployed ranking produced {int(rf250['distinct_confirmed'])}/250 ({rf250['precision']:.1%}) with {rf250['site_recall']:.2%} site recall across {int(rf250['routes_represented'])} routes; the largest route supplied {rf250['largest_route_share']:.1%} of Top250 recommendations. Recommendations require field verification.”",
        "",
        "## Unresolved provenance limits",
        "",
        "The historical 650/714 benchmark used the older V14 FLOW-C score vector and its score-ordered representatives, so that coverage is not automatically transferable to the deployed RF + dynamic ROUTE-Z representatives. POINTWISE-GB is preserved as outer-fold models and OOF scores, but its inner-selected configuration varies by held-out route; there is no single audited, frozen POINTWISE bundle for arbitrary unseen roads. This task did not test or authorize fitting one. Therefore the production decision is RESEARCH / DEPLOYMENT MISMATCH UNRESOLVED, not automatic POINTWISE promotion. The comparison is confined to the shared 51,417-candidate Ulster evaluation pool and authoritative inventory.",
        "",
        "No tuning, model selection, feature experiment, retraining, or UI change occurred. The field message remains: **Recommendations require field verification.**",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    if before != after:
        raise AssertionError("A protected model, input, evaluation, or production file changed.")
    (OUT / "input_integrity.json").write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": True}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(json.dumps({"decision": decision, "exact_rf_oof_already_existed": True}, indent=2))


if __name__ == "__main__":
    main()
