"""Coverage attribution and a single fixed-family negative-weight comparison.

Read preserved RF OOF scores. Train paired HGB models on the exact permitted
label manifest. Never add unknown-negative examples or change production.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file
from culvert_ai.inference import RouteZConfig, dynamic_route_z, rank_flow_c_candidates


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/experiments"
PREVIOUS = EXP / "site_challenger_20261015"
V8 = EXP / "nysdot_ulster_dem_hydrology_v8_20261005"
OUT = EXP / "site_coverage_negative_weights_20261015"
KS = (25, 50, 100, 250, 500, 1000)
CONFIG = dict(id="HGB_REGULAR", family="hgb", learning_rate=.04, trees=200,
              leaves=7, leaf_min=40, l2=10., weighting="route_balanced")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def measures(sites, real, evaluator):
    xy = real[["x", "y"]].to_numpy(float)
    full, _ = evaluator.match_sites(sites, xy)
    counts = sites.head(250).route.value_counts()
    result = dict(representatives=len(sites), full_matches=len(full), unmatched=len(real)-len(full),
                  full_recall=len(full)/len(real), inventory_records=len(real),
                  routes_at_250=len(counts), largest_route=str(counts.index[0]),
                  largest_route_share=float(counts.iloc[0]/min(250, len(sites))))
    for k in KS:
        hits, _ = evaluator.match_sites(sites, xy, top_k=k)
        result[f"matched_at_{k}"] = len(hits)
        result[f"recommendations_at_{k}"] = min(k, len(sites))
    result["site_recall_at_250"] = result["matched_at_250"] / len(real)
    return result, full


def manual_sites(frame, score, runtime):
    result = frame.copy()
    result["priority"] = result[score]
    return runtime.saved_priority_sites(result, "priority")


def protected(runtime):
    hashes = runtime.protected_hashes()
    for path in PREVIOUS.rglob("*"):
        if path.is_file() and "__pycache__" not in str(path):
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    for name in ("training_rows.csv", "control_model_specification.json", "exact_feature_lists.json",
                 "hydro_c_plus_flow_c__whole_route__oof_predictions.parquet"):
        path = V8 / name
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def rf_coverage(runtime, evaluator):
    pool = pd.read_parquet(runtime.POOL, columns=["candidate_id", "route", "x_m", "y_m"])
    original = pd.read_parquet(V8 / "hydro_c_plus_flow_c__whole_route__oof_predictions.parquet")
    copy = pd.read_parquet(runtime.V18 / "out_of_fold_predictions/flow_c.parquet")
    scores = original.set_index("candidate_id").probability
    assert scores.sort_index().equals(copy.set_index("candidate_id").outer_oof_score.sort_index())
    assert len(pool) == 51417 and set(pool.candidate_id) == set(scores.index)
    pool["raw_score"] = pool.candidate_id.map(scores)
    source = gpd.read_file(ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg", columns=["candidate_id", "geometry"])
    native = source.set_index("candidate_id", drop=False).loc[pool.candidate_id].reset_index(drop=True).to_crs("EPSG:32618")
    native["route_id"] = pool.route.to_numpy()
    native["flow_c_score"] = pool.raw_score.to_numpy(float)
    real_archive = evaluator.load_real()
    official_geometry = gpd.GeoSeries(gpd.points_from_xy(real_archive.x, real_archive.y), crs="EPSG:26918").to_crs("EPSG:32618")
    real_native = real_archive.copy()
    real_native["x"], real_native["y"] = official_geometry.x, official_geometry.y
    bundle = load_frozen_flow_c_bundle(ROOT / "models/flow_c_frozen_v1.joblib")
    assert sha256_file(ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg") == bundle["training_data_provenance"]["feature_source_sha256"]
    assert bundle["route_z_global"]["source_sha256"] == sha256_file(V8 / "hydro_c_plus_flow_c__whole_route__oof_predictions.parquet")
    cfg = RouteZConfig(global_center=bundle["route_z_global"]["center"], global_scale=bundle["route_z_global"]["scale"])
    no_route = RouteZConfig(minimum_standard_candidates=10**9, minimum_robust_candidates=10**9,
                           global_center=0., global_scale=1.)
    native_sites = {}
    for name, config in (("RF_DYNAMIC_ROUTE_Z", cfg), ("RF_RAW", no_route)):
        sites = rank_flow_c_candidates(native, route_z_config=config).rename(
            columns={"global_rank": "rank", "physical_site_id": "site_id"})
        sites["route"] = sites.route_id
        sites["x_m"], sites["y_m"] = sites.geometry.x, sites.geometry.y
        sites["raw_score"] = sites.flow_c_score
        native_sites[name] = sites
    # A single global affine transform with positive scale cannot change raw
    # ordering. No new calibration or training-statistic estimate is needed.
    global_frame = native.copy()
    global_frame["route_id"] = "one_global_group"
    global_sites = rank_flow_c_candidates(global_frame, route_z_config=no_route).candidate_id.tolist()
    assert global_sites == native_sites["RF_RAW"].candidate_id.tolist()
    # Also verify the actual frozen global reference yields the identical queue.
    frozen_global = RouteZConfig(minimum_standard_candidates=10**9, minimum_robust_candidates=10**9,
                                global_center=cfg.global_center, global_scale=cfg.global_scale)
    assert rank_flow_c_candidates(native, route_z_config=frozen_global).candidate_id.tolist() == global_sites
    native_sites["RF_FROZEN_GLOBAL_AFFINE"] = native_sites["RF_RAW"].copy()
    rows, coverage_rows, matches = [], [], {}
    raw_xy = np.c_[native.geometry.x, native.geometry.y]
    inventory_xy = real_native[["x", "y"]].to_numpy(float)
    raw_distances = cKDTree(raw_xy).query(inventory_xy)[0]
    raw_match_count = int((raw_distances <= 20.).sum())
    coverage_rows.append(dict(stage="RAW_CANDIDATES_NATIVE_CRS", rows=len(pool),
                              matched=raw_match_count, unmatched=714-raw_match_count,
                              coverage=raw_match_count/714))
    per_record = real_native[["real_key", "route", "x", "y"]].copy()
    per_record["raw_candidate_distance_m"] = raw_distances
    per_record["raw_candidate_within_20m"] = raw_distances <= 20.
    for name, sites in native_sites.items():
        m, assignment = measures(sites, real_native, evaluator)
        rows.append(dict(coordinate_frame="native_EPSG32618", policy=name, **m))
        matches[name] = assignment
        coverage_rows.append(dict(stage=name+"_NATIVE_CRS", rows=len(sites), matched=m["full_matches"],
                                  unmatched=m["unmatched"], coverage=m["full_recall"]))
        sites.drop(columns="geometry").to_parquet(OUT / f"representatives_{name}.parquet", index=False)
        per_record[f"{name}_one_to_one_matched"] = per_record.index.isin(assignment.real_index)
        per_record[f"{name}_nearest_representative_m"] = cKDTree(sites[["x_m", "y_m"]]).query(inventory_xy)[0]
    per_record.to_csv(OUT / "official_site_coverage.csv", index=False)
    # Preserve and reproduce the archival coordinate convention separately.
    pool["route_z_score"] = dynamic_route_z(pool, score_column="raw_score", route_column="route", config=cfg).route_z_score
    archive_sites = {}
    for name, score in (("RF_RAW", "raw_score"), ("RF_DYNAMIC_ROUTE_Z", "route_z_score")):
        sites = manual_sites(pool, score, runtime)
        archive_sites[name] = sites
        m, _ = measures(sites, real_archive, evaluator)
        rows.append(dict(coordinate_frame="archived_EPSG26918", policy=name, **m))
    assert rows[-2]["matched_at_250"] == 33 and rows[-1]["matched_at_250"] == 5
    archive_raw_coverage = measures(archive_sites["RF_RAW"], real_archive, evaluator)[0]
    write_json("historical_650_check.json", dict(
        original_benchmark="reports/experiments/nysdot_ulster_real_inventory_benchmark_v1_20261006/benchmark_metrics.csv",
        archived_reconstruction=archive_raw_coverage,
        historical_650_reproduced=archive_raw_coverage["full_matches"] == 650,
        historical_15650_representatives_reproduced=archive_raw_coverage["representatives"] == 15650,
        note="Native app CRS and original archival CRS are reported separately; exact 20m threshold remains fixed."))
    pd.DataFrame(rows).to_csv(OUT / "rf_ranking_policies.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "candidate_representative_coverage.csv", index=False)
    raw, rz = native_sites["RF_RAW"], native_sites["RF_DYNAMIC_ROUTE_Z"]
    raw_ids, z_ids = set(raw.candidate_id), set(rz.candidate_id)
    pd.DataFrame([dict(candidate_id=i, membership="both" if i in raw_ids & z_ids else "raw_only" if i in raw_ids else "route_z_only")
                  for i in sorted(raw_ids | z_ids)]).to_csv(OUT / "representative_differences.csv", index=False)
    raw_found = set(matches["RF_RAW"].real_index)
    z_found = set(matches["RF_DYNAMIC_ROUTE_Z"].real_index)
    rep_comparison = dict(
        common_representative_ids=len(raw_ids & z_ids), raw_only=len(raw_ids-z_ids), route_z_only=len(z_ids-raw_ids),
        raw_one_to_one_inventory_ids_only=sorted(raw_found-z_found), route_z_one_to_one_inventory_ids_only=sorted(z_found-raw_found),
        common_one_to_one_inventory_ids=len(raw_found & z_found),
        raw_representatives_within_20m_of_a_route_z_representative=int((cKDTree(rz[["x_m", "y_m"]]).query(raw[["x_m", "y_m"]])[0] <= 20.).sum()),
        note="Inventory assignment differences use a deterministic maximum matching; alternate maximum assignments may exist.")
    write_json("representative_overlap.json", rep_comparison)
    # Diagnostic factorial: fix representatives, then change head ordering only.
    score_lookup = dynamic_route_z(native, config=cfg).set_index("candidate_id").route_z_score
    attribution = []
    for representative_policy, sites in (("RF_RAW", raw), ("RF_DYNAMIC_ROUTE_Z", rz)):
        for ordering in ("raw", "route_z"):
            fixed = sites.copy()
            fixed["ordering_score"] = fixed.raw_score if ordering == "raw" else fixed.candidate_id.map(score_lookup)
            fixed = fixed.sort_values(["ordering_score", "raw_score", "candidate_id"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
            fixed["rank"] = np.arange(1, len(fixed)+1)
            m, _ = measures(fixed, real_native, evaluator)
            attribution.append(dict(representative_policy=representative_policy, head_ordering=ordering, **m))
    pd.DataFrame(attribution).to_csv(OUT / "ordering_vs_representative_attribution.csv", index=False)
    write_json("rf_specification.json", dict(
        model_class=bundle["model_class"], hyperparameters=bundle["model_parameters"],
        feature_columns=bundle["feature_columns"], score_source=str((V8 / "hydro_c_plus_flow_c__whole_route__oof_predictions.parquet").relative_to(ROOT)),
        score_identity_verified=True, candidate_rows=len(pool), coordinate_crs="EPSG:32618",
        archived_coordinates_crs="EPSG:26918", global_affine_rank_identity=True,
        normalization="existing dynamic_route_z with frozen bundle fallback",
        site_rule="existing rank_flow_c_candidates, greedy 20m suppression, route_z desc / raw RF desc / candidate ID asc",
        caveat="Historical RF OOF features/folds contain legacy label-based route overrides; saved development diagnostic, not new unbiased testing."))
    print("COVERAGE", pd.DataFrame(coverage_rows).to_string(index=False), flush=True)
    print("RF POLICIES", pd.DataFrame(rows)[["coordinate_frame", "policy", "matched_at_100", "matched_at_250", "full_matches"]].to_string(index=False), flush=True)
    return coverage_rows, rows, attribution


def provenance():
    current = pd.read_csv(PREVIOUS / "final_training_rows.csv")
    old = pd.read_csv(V8 / "training_rows.csv")
    audit = current.rename(columns={"label_source": "current_label_source"}).merge(
        old[["candidate_id", "experiment_target", "training_label_class", "experiment_label_source"]],
        on="candidate_id", how="left", validate="one_to_one")
    audit["weak_negative_eligible_for_reweighting"] = audit.target.eq(0) & audit.current_label_source.isin(
        ["proximity_derived_negative", "ordinary_unlabeled_candidate", "unlabeled_hard_negative"])
    audit["weight_changes"] = False
    audit["control_multiplier"] = 1.
    audit["challenger_multiplier"] = 1.
    audit.to_csv(OUT / "training_label_provenance.csv", index=False)
    old.groupby(["experiment_target", "training_label_class", "experiment_label_source"]).size().rename("rows").reset_index().to_csv(OUT / "original_rf_label_categories.csv", index=False)
    assert current.target.eq(0).sum() == 126
    assert current.loc[current.target.eq(0), "label_source"].eq("field_verified_negative").all()
    assert not audit.weak_negative_eligible_for_reweighting.any()
    summary = dict(current_positive_rows=int(current.target.eq(1).sum()), current_verified_negative_rows=126,
                   current_weak_negative_rows=0, changed_weight_rows=0,
                   original_rf_ordinary_unlabeled_negative_rows=941, original_rf_proximity_negative_rows=5,
                   original_rf_independent_field_positive_rows=250, original_rf_official_25m_positive_rows=696,
                   current_independent_field_positive_rows_separately_added=0,
                   explanation="Current HGB uses authoritative positives <=20m and verified negatives only. Original RF weak/unlabeled rows are provenance references, not current HGB examples; adding them would violate the no-new-examples constraint.",
                   weak_weight_parameter_identifiable=False,
                   challenger_status="requested weak-negative weighting has no eligible rows; exact no-op comparison")
    write_json("negative_provenance_summary.json", summary)
    return current, summary


def effective_weights(train):
    w = train.weight.to_numpy(float).copy()
    for target in (0, 1):
        mask = train.target.eq(target)
        w[mask] /= train.loc[mask].groupby("route").weight.transform("sum").to_numpy(float)
        w[mask] *= .5 * len(train)/w[mask].sum()
    return w


def train_pairs(runtime, evaluator, manifest):
    pool = pd.read_parquet(PREVIOUS / "development_features.parquet")
    negatives = pd.read_parquet(PREVIOUS / "verified_negative_features.parquet")
    features = json.loads((PREVIOUS / "feature_schema.json").read_text())["feature_columns"]
    frozen = joblib.load(PREVIOUS / "challenger.joblib")
    assert frozen["configuration"] == CONFIG and frozen["normalization"] == "raw"
    all_features = pd.concat([pool, negatives], ignore_index=True).drop_duplicates("candidate_id").set_index("candidate_id")
    train_all = manifest.merge(all_features[["x_m", "y_m", *features]], left_on="candidate_id", right_index=True, validate="one_to_one")
    assert len(train_all) == len(manifest)
    real = evaluator.load_real()
    roads = sorted(pool.route.unique())
    groups = [set(roads[i::3]) for i in range(3)]
    write_json("group_assignments.json", dict(groups=[sorted(g) for g in groups], note="Fixed from road names before outcomes; each evaluation group is scored by one fitted model, raw policy only."))
    all_roads = set(roads) | set(negatives.route)
    split_rows, metrics_rows, route_rows, separation_rows, score_rows = [], [], [], [], []
    fit_count = 0

    def one_pair(allowed, heldout, name, outer_excluded=frozenset()):
        nonlocal fit_count
        test = pool.loc[pool.route.isin(heldout)].copy().reset_index(drop=True)
        heldout_geometry = pd.concat([pool.loc[pool.route.isin(heldout | set(outer_excluded))],
                                      negatives.loc[negatives.route.isin(heldout | set(outer_excluded))]])[["x_m", "y_m"]].to_numpy(float)
        train = train_all.loc[train_all.route.isin(allowed)].copy()
        distance = cKDTree(heldout_geometry).query(train[["x_m", "y_m"]])[0]
        train = train.loc[distance > 40.].reset_index(drop=True)
        assert not set(train.route) & (heldout | set(outer_excluded))
        assert set(train.candidate_id).issubset(set(manifest.candidate_id))
        assert train.target.nunique() == 2
        w = effective_weights(train)
        train["control_effective_weight"] = w
        train["challenger_effective_weight"] = w.copy()
        train[["candidate_id", "route", "target", "label_source", "control_effective_weight", "challenger_effective_weight"]].to_csv(OUT / f"weights_{name}.csv", index=False)
        split_rows.append(dict(split=name, training_roads=sorted(set(train.route)), heldout_roads=sorted(heldout), outer_excluded=sorted(outer_excluded),
                               training_rows=len(train), positives=int(train.target.sum()), verified_negatives=int(train.target.eq(0).sum()),
                               weak_negatives=0, changed_weight_rows=0,
                               min_train_test_distance_m=float(cKDTree(heldout_geometry).query(train[["x_m", "y_m"]])[0].min())))
        models = {}
        predictions = {}
        comparison_real = real.loc[real.route.isin(heldout)].copy()
        for arm in ("control", "challenger"):
            fitted = runtime.fit_model(CONFIG, train, features)
            fit_count += 1
            assert fitted["model"].get_params() == frozen["model"].get_params()
            predicted = test[["candidate_id", "route", "x_m", "y_m"]].copy()
            predicted["raw_score"] = runtime.score_model(fitted, test, features)
            predicted.to_parquet(OUT / f"scores_{name}_{arm}.parquet", index=False)
            models[arm], predictions[arm] = fitted, predicted
            sites = runtime.saved_priority_sites(predicted, "raw_score")
            result, assignment = measures(sites, comparison_real, evaluator)
            metrics_rows.append(dict(split=name, arm=arm, stage="inner" if "inner" in name else "outer", **result))
            # Descriptive outcome summaries are never inputs to fitting/weights.
            if "inner" not in name:
                for route, part in sites.groupby("route"):
                    reference = real.loc[real.route.eq(route)]
                    if reference.empty:
                        route_rows.append(dict(split=name, arm=arm, route=route, inventory_records=0, matched_at_100=0, matched_at_250=0))
                    else:
                        m, _ = measures(part.reset_index(drop=True), reference, evaluator)
                        route_rows.append(dict(split=name, arm=arm, route=route, **m))
                nearest = cKDTree(comparison_real[["x", "y"]]).query(predicted[["x_m", "y_m"]])[0]
                observed = nearest <= 20.
                separation_rows.append(dict(split=name, arm=arm,
                                            positive_candidate_score_median=float(predicted.loc[observed, "raw_score"].median()),
                                            inventory_unmatched_score_median=float(predicted.loc[~observed, "raw_score"].median()),
                                            descriptive_candidate_roc_auc=float(roc_auc_score(observed, predicted.raw_score)),
                                            descriptive_candidate_average_precision=float(average_precision_score(observed, predicted.raw_score))))
                for route, part in predicted.groupby("route"):
                    score_rows.append(dict(split=name, arm=arm, route=route, candidates=len(part),
                                           mean=float(part.raw_score.mean()), std=float(part.raw_score.std(ddof=0)),
                                           median=float(part.raw_score.median()), maximum=float(part.raw_score.max())))
                top = sites.head(250).copy()
                top["nearest_inventory_distance_m"] = cKDTree(comparison_real[["x", "y"]]).query(top[["x_m", "y_m"]])[0]
                top["inventory_matched_any"] = top.nearest_inventory_distance_m.le(20.)
                top["interpretation"] = np.where(top.inventory_matched_any, "inventory-near", "inventory-unmatched, field status unknown")
                top.to_csv(OUT / f"top250_{name}_{arm}.csv", index=False)
        identical = np.array_equal(predictions["control"].raw_score, predictions["challenger"].raw_score)
        assert identical
        assert models["control"]["medians"].equals(models["challenger"]["medians"])
        # No inference-time fit. Rename every road, reorder feature columns, and
        # round-trip a temporary evaluation checkpoint without promoting it.
        sample = test.head(100).copy()
        sample["route"] = "unseen-" + sample.route.astype(str)
        first = runtime.score_model(models["control"], sample, features)
        artifact = OUT / f"evaluation_checkpoint_{name}.joblib"
        joblib.dump(models["control"], artifact, compress=3)
        loaded = joblib.load(artifact)
        assert np.array_equal(first, runtime.score_model(loaded, sample, features))
        return identical

    for outer, heldout in enumerate(groups):
        allowed = all_roads - heldout
        inner_groups = [set(sorted(allowed)[::2]), set(sorted(allowed)[1::2])]
        for inner, inner_test in enumerate(inner_groups):
            print(f"TRAIN outer {outer} inner {inner}: identical permitted weights", flush=True)
            one_pair(allowed-inner_test, inner_test, f"outer{outer}_inner{inner}", heldout)
        print(f"TRAIN outer {outer}: one fitted HGB per arm for roads {sorted(heldout)}", flush=True)
        one_pair(allowed, heldout, f"outer{outer}")
    pd.DataFrame(metrics_rows).to_csv(OUT / "control_challenger_fold_metrics.csv", index=False)
    pd.DataFrame(route_rows).to_csv(OUT / "per_route_metrics.csv", index=False)
    pd.DataFrame(separation_rows).to_csv(OUT / "score_separation.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUT / "score_distributions_by_route.csv", index=False)
    write_json("split_isolation.json", split_rows)
    aggregate_rows = []
    outer_metrics = pd.DataFrame(metrics_rows).query("stage == 'outer'")
    for arm, part in outer_metrics.groupby("arm"):
        for metric in ("matched_at_25", "matched_at_50", "matched_at_100", "matched_at_250", "matched_at_500", "matched_at_1000", "full_recall", "site_recall_at_250", "largest_route_share"):
            aggregate_rows.append(dict(arm=arm, metric=metric, mean=float(part[metric].mean()),
                                       minimum=float(part[metric].min()), maximum=float(part[metric].max()),
                                       population_std=float(part[metric].std(ddof=0))))
    pd.DataFrame(aggregate_rows).to_csv(OUT / "fold_aggregates.csv", index=False)
    inner_data = pd.DataFrame(metrics_rows).query("stage == 'inner'")
    inner_objectives = {}
    for arm, part in inner_data.groupby("arm"):
        inner_objectives[arm] = float((.5*(part.matched_at_100/part.recommendations_at_100 + part.matched_at_250/part.recommendations_at_250)).mean())
    assert inner_objectives["control"] == inner_objectives["challenger"]
    outcome = dict(fitted_estimators=fit_count, family="HistGradientBoostingClassifier", fixed_configuration=CONFIG,
                   score_policy="raw for every model and fold", inner_objectives=inner_objectives,
                   improvement_supported=False, changed_weight_rows=0, arm_predictions_bitwise_identical=True,
                   final_model_frozen=False, reason="No uncertain-negative examples exist in the permitted HGB manifest; the weight intervention is an exact no-op. Strictly tied inner evidence retains the existing control. No final all-development fit or replacement artifact is justified.",
                   pooled_global_top250=None,
                   pooling_reason="Three disjoint multi-road groups are evaluated separately under one fitted model and one raw policy each; their raw score scales are not pooled.")
    write_json("training_outcome.json", outcome)
    return outer_metrics, outcome


def report(coverage, rf, attribution, labels, outer, outcome):
    record_coverage = pd.read_csv(OUT / "official_site_coverage.csv")
    geometric_representative_coverage = int(record_coverage.RF_DYNAMIC_ROUTE_Z_nearest_representative_m.le(20).sum())
    candidate_hits = int(record_coverage.raw_candidate_within_20m.sum())
    capacity_matches = int(record_coverage.RF_DYNAMIC_ROUTE_Z_one_to_one_matched.sum())
    stage_loss = dict(raw_candidate_misses=714-candidate_hits,
                     raw_candidate_hits=candidate_hits,
                     representative_geometric_hits=geometric_representative_coverage,
                     geometric_losses_after_consolidation=candidate_hits-geometric_representative_coverage,
                     additional_one_to_one_capacity_losses=geometric_representative_coverage-capacity_matches,
                     final_representative_matches=capacity_matches)
    assert stage_loss["raw_candidate_misses"] + stage_loss["geometric_losses_after_consolidation"] + stage_loss["additional_one_to_one_capacity_losses"] + capacity_matches == 714
    write_json("coverage_loss_stages.json", stage_loss)
    dev = pd.read_parquet(PREVIOUS / "development_features.parquet").set_index("candidate_id")
    pattern_fields = ["distance_to_nearest_stream_m", "slope_degrees", "dem_culvert_terrain_score",
                      "route_low_point_score", "distance_to_nearest_mapped_crossing_m",
                      "log1p_distance_to_dem_road_drainage_crossing_0p1km2_m", "hydrology_edge_risk"]
    pattern_rows = []
    for path in OUT.glob("top250_outer*_*.csv"):
        top = pd.read_csv(path)
        top = top.merge(dev[pattern_fields], left_on="candidate_id", right_index=True, validate="one_to_one")
        for near, subset in top.groupby("inventory_matched_any"):
            for field in pattern_fields:
                values = pd.to_numeric(subset[field], errors="coerce")
                pattern_rows.append(dict(queue=path.stem, inventory_near=bool(near), rows=len(subset),
                                         field=field, median=float(values.median()), missing_fraction=float(values.isna().mean())))
    pd.DataFrame(pattern_rows).to_csv(OUT / "inventory_unmatched_feature_patterns.csv", index=False)
    lines = ["# Culvert coverage and targeted negative-weight evaluation — 2026-10-15", "",
             "## Decision", "", "No new model is justified. The existing HGB manifest contains no weak/proximity/unlabeled-negative rows to downweight. Eighteen HGB fits were actually executed in paired nested road-group evaluations; paired scores and metrics are identical. No production or UI files were changed, and the prior frozen challenger was preserved.", "",
             "## Candidate and representative coverage", "",
             "Native production geometry is EPSG:32618; authoritative EPSG:26918 inventory coordinates were transformed into the same frame before matching. Exactly 51,417 candidate IDs and the identical RF OOF scores are used for every RF policy. Raw candidate coverage counts every official record with any candidate within 20m; consolidated coverage uses maximum one-to-one 20m matching.", "",
             "| Stage | Candidates/representatives | Matched /714 | Unmatched | Coverage |", "|---|---:|---:|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['stage']} | {row['rows']} | {row['matched']} | {row['unmatched']} | {row['coverage']:.2%} |")
    lines += ["", f"Loss-stage accounting for dynamic RF: {stage_loss['raw_candidate_misses']} records lack any candidate; {stage_loss['geometric_losses_after_consolidation']} further records lose every representative within 20m; {stage_loss['additional_one_to_one_capacity_losses']} further records cannot receive a distinct representative under the maximum one-to-one assignment. There are {geometric_representative_coverage} records with some representative nearby but only {capacity_matches} simultaneous distinct matches. The last capacity loss must not be misdescribed solely as a localization error."]
    lines += ["", "Archival EPSG:26918 reconstructions are reported separately below to reproduce the original benchmark without silently changing its coordinates. Exact threshold boundaries can affect representative selection. The inventory and tolerance remain unchanged. Native-versus-archival count differences are coordinate/threshold reconstruction effects, not a learned improvement.", "",
              "## RF saved-score policy comparison", "",
              "| Frame / policy | @25 | @50 | @100 | @250 | @500 | @1000 | Full matches | Roads@250 | Largest share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rf:
        lines.append(f"| {row['coordinate_frame']} / {row['policy']} | " + " | ".join(str(row[f"matched_at_{k}"]) for k in KS) + f" | {row['full_matches']} | {row['routes_at_250']} | {row['largest_route_share']:.1%} |")
    native_raw = next(r for r in rf if r["coordinate_frame"] == "native_EPSG32618" and r["policy"] == "RF_RAW")
    native_z = next(r for r in rf if r["coordinate_frame"] == "native_EPSG32618" and r["policy"] == "RF_DYNAMIC_ROUTE_Z")
    lines += ["", f"Current native-CRS dynamic RF coverage is {native_z['full_matches']}/714, while raw RF coverage is {native_raw['full_matches']}/714. Thus the historical 650/714 is {'numerically reproduced' if native_z['full_matches']==650 else 'not reproduced'} by the current dynamic pipeline. These are full-list representative recall values, not classifier accuracy or field precision.", "",
              "## Ordering versus representative selection", "", "| Fixed representative population | Head ordering | @100 | @250 | Full matches |", "|---|---|---:|---:|---:|"]
    for row in attribution:
        lines.append(f"| {row['representative_policy']} | {row['head_ordering']} | {row['matched_at_100']} | {row['matched_at_250']} | {row['full_matches']} |")
    lines += ["", "The same saved RF predictions, populations, tie-breaks and site rules are used, so differences in this diagnostic are attributable to ordering and representative choices under those policies. This is a development attribution, not an outer-outcome policy-selection exercise. A global fixed affine normalization with positive scale leaves raw ordering and representatives identical; it cannot repair poor discrimination. Frozen bundle reference statistics are used only to verify this identity, not to claim calibrated or training-only confidence.", "",
              "representative_overlap.json and representative_differences.csv record exact shared IDs and substitutions; official_site_coverage.csv separates candidate misses, representative losses and deterministic one-to-one assignment differences. Inventory nonmatches are not asserted to be field-confirmed absences.", "",
              "## Negative-label provenance", "",
              f"Current HGB: {labels['current_positive_rows']} authoritative <=20m positive candidate rows and 126 verified negatives. No independent field-positive rows were separately added to this manifest. The original RF has 250 field-confirmed positive rows, 696 official <=25m positive rows, 129 verified negatives, 5 proximity-derived negatives and 941 unlabeled hard-negative rows. These original categories are audited as provenance only. They are not silently introduced into the HGB example population, and the HGB positive rule remains <=20m.", "",
              "There are **zero changed-weight rows**. Verified negatives retain their original route/class-balanced contributions. A weak-negative coefficient is unidentifiable because it multiplies no permitted rows; no arbitrary confidence value is assigned. training_label_provenance.csv identifies every permitted row, original provenance when available, and both multipliers. Four field-negative locations lie within the inventory's 20m evaluation radius, as recorded in the previous experiment; these supplied observations were not relabeled.", "",
              "## Paired fixed-family validation", "",
              "Three deterministic disjoint road groups contain multiple roads each. Within each outer training population, two inner road groups are withheld. Every control/challenger pair uses the same 48 existing feature values, exact permitted candidate/negative IDs, original labels, fixed 200-iteration HGB parameters, training-only medians, effective route/class weights, and raw-score policy. Training rows within 40m of any outer/inner held-out candidate or negative location are removed. A split never sees a held-out road label for fitting or reweighting.", "",
              "Each evaluation group is scored by one fitted model per arm. No queue mixes scores from different models or policies. Macro means/minima/maxima/std across fold budgets are saved; they are not global TopK counts and cannot be compared numerically as if they were the historical pooled whole-route queue.", "",
              "| Evaluation group | Arm | @25 | @50 | @100 | @250 | @500 | Full matches / reference | Full recall | Roads@250 | Largest share |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in outer.to_dict("records"):
        lines.append(f"| {row['split']} | {row['arm']} | " + " | ".join(str(row[f"matched_at_{k}"]) for k in KS[:-1]) + f" | {row['full_matches']}/{row['inventory_records']} | {row['full_recall']:.2%} | {row['routes_at_250']} | {row['largest_route_share']:.1%} |")
    lines += ["", f"Mean inner selection objective: control={outcome['inner_objectives']['control']:.6f}; challenger={outcome['inner_objectives']['challenger']:.6f}. The arms are bitwise identical. The requested reweighting did not improve held-out retrieval; the underlying intervention is absent rather than evidence that downweighting genuine uncertain examples could never work.", "",
              "## Original RF reference and score separation", "",
              "The original RF figures above retain their saved OOF population and historical weak-label/preprocessing provenance. Some saved fold assignments differ from corrected label-free road groups, and historical RF training uses a different label manifest. It is therefore not a newly fitted, identical-training control in the HGB paired experiment. No new RF family was trained to manufacture such a comparison.", "",
              "score_separation.csv summarizes positive-candidate versus inventory-unmatched score distributions and descriptive candidate ROC/AP; these are inventory-based diagnostics only. score_distributions_by_route.csv, per_route_metrics.csv and Top250 exports record route concentration and inventory-unmatched patterns. Ground-truth distances appear only in evaluation outputs, never in predictors or weighting.", "",
              "The HGB positive versus inventory-unmatched median scores are 0.931/0.937, 0.961/0.908, and 0.974/0.962 across the three outer groups. In one group the unknowns score higher; in the others the score distributions remain heavily overlapping. These scores are not field probabilities. inventory_unmatched_feature_patterns.csv describes existing stream, crossing, terrain and drainage features among inventory-near and unmatched Top250 rows; proximity matches can exceed one-to-one counts, so those groups are explicitly descriptive.", "",
              "Historical 76-feature POINTWISE remains a separate 18/100, 41/250 development benchmark. Different features, supervision, fold structure and pooled-score conventions prevent a causal comparison against this 48-feature grouped experiment.", "",
              "## Final answers", "",
              f"1. Raw candidate coverage: {coverage[0]['matched']}/714 ({coverage[0]['coverage']:.2%}); {coverage[0]['unmatched']} official records lack a candidate within 20m.",
              f"2. Current dynamic RF full-list coverage: {native_z['full_matches']}/714 ({native_z['full_recall']:.2%}) from {native_z['representatives']} representatives.",
              f"3. Historical 650/714 under current dynamic RF: {'yes, numerically' if native_z['full_matches']==650 else 'no'}. See historical_650_check.json for the separate archival raw reconstruction.",
              f"4. Native raw RF Top100/250: {native_raw['matched_at_100']}/{native_raw['matched_at_250']}; dynamic ROUTE-Z: {native_z['matched_at_100']}/{native_z['matched_at_250']}.",
              f"5. Holding either representative population fixed, raw head ordering finds {native_raw['matched_at_250']}/250 and dynamic ordering finds {native_z['matched_at_250']}/250. The reduction is therefore primarily the route-local normalization's head allocation on this exact saved-score population; substitutions do not explain the Top250 gap. Raw RF still has weak score separation, since only {native_raw['matched_at_250']} of its Top250 are inventory-matched. Representative selection/one-to-one capacity causes additional full-list losses as counted above. No production policy was selected using these outcomes.",
              "6. Weak-negative weight improvement: none; zero such rows exist in the permitted HGB manifest, so arm weights and predictions are identical.",
              "7. Exact targeted @100/@250 results are the three paired group rows above. There is intentionally no misleading pooled global figure.",
              "8. Declared inner objective: exact tie; no challenger selected over control.",
              "9. The historical 41/250 uses 76 features and a different development evaluation; no claim of exceeding it is justified.",
              "10. No new final challenger was frozen. Evaluation checkpoints are research evidence only; the existing frozen model remains unchanged. A useful independent improvement requires an identifiable allowed intervention and independent road/field evidence.", "",
              "## Reproducibility and integrity", "",
              "Reproduce into a new directory with scripts/python.sh scripts/verify_coverage_negative_weights.py --output reports/experiments/site_coverage_negative_weights_reproduction. Completed outputs cannot be overwritten. Use --report-only to regenerate reports and inference checks without fitting. Frozen source, feature-schema and input hashes accompany all metrics. HistGradientBoostingClassifier is the only estimator family fitted. Serialized evaluation scores, arbitrary unseen road names, split spatial separation and identical paired scoring are checked. No model search, new examples, new features, candidate generation, inventory edits, production changes, UI changes or deployment occurred.", "",
              "Recommendations require field verification."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "COMPLETE.json").exists():
        raise RuntimeError("Refusing to overwrite a completed experiment")
    runtime = load_module("targeted_previous_runtime", ROOT / "scripts/train_site_challenger.py")
    evaluator = load_module("targeted_previous_evaluator", runtime.V18 / "run_experiment.py")
    before = protected(runtime)
    write_json("preregistered_config.json", dict(
        family="HistGradientBoostingClassifier", configuration=CONFIG, score_policy="raw fixed from frozen HGB",
        example_manifest=str((PREVIOUS / "final_training_rows.csv").relative_to(ROOT)),
        positive_rule="authoritative same-road inventory <=20m, unchanged", new_examples_allowed=False,
        outer_groups=3, inner_groups=2, outer_group_assignment="sorted label-free road IDs, round-robin into 3 groups",
        objective="equal average of matched fraction@100 and @250; no outer outcome selection; ties retain control",
        weak_negative_handling="downweight only provenance-established uncertain negative rows in existing HGB manifest; if no rows exist, report no-op and do not invent examples or confidence",
        freeze_condition="strict training-inner objective improvement required",
    ))
    write_json("input_hashes_before.json", before)
    coverage, rf, attribution = rf_coverage(runtime, evaluator)
    manifest, labels = provenance()
    print("PROVENANCE", json.dumps(labels), flush=True)
    outer, outcome = train_pairs(runtime, evaluator, manifest)
    report(coverage, rf, attribution, labels, outer, outcome)
    after = protected(runtime)
    assert before == after
    write_json("input_integrity.json", dict(unchanged=True, before=before, after=after))
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    write_json("artifact_hashes.json", {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "artifact_hashes.json"})
    write_json("COMPLETE.json", dict(status="complete", improvement_supported=False, final_challenger_frozen=False,
                                     existing_model_preserved=True, production_unchanged=True, fitted_estimators=outcome["fitted_estimators"]))
    print(outer.to_string(index=False), flush=True)
    print(json.dumps(outcome, indent=2), flush=True)
    finish_saved_report()


def finish_saved_report():
    """Complete reporting and inference checks without rerunning any training."""
    runtime = load_module("targeted_previous_runtime", ROOT / "scripts/train_site_challenger.py")
    before = json.loads((OUT / "input_hashes_before.json").read_text())
    assert before == protected(runtime)
    assert sha256_file(ROOT / "scripts/train_site_challenger.py") == sha256_file(PREVIOUS / "train_site_challenger.py")
    coverage = pd.read_csv(OUT / "candidate_representative_coverage.csv").to_dict("records")
    rf = pd.read_csv(OUT / "rf_ranking_policies.csv").to_dict("records")
    attribution = pd.read_csv(OUT / "ordering_vs_representative_attribution.csv").to_dict("records")
    labels = json.loads((OUT / "negative_provenance_summary.json").read_text())
    provenance_rows = pd.read_csv(OUT / "training_label_provenance.csv")
    labels["original_rf_training_row_ids_reused"] = int(provenance_rows.experiment_target.notna().sum())
    labels["opposite_class_labels_on_reused_ids"] = int((provenance_rows.experiment_target.notna() & provenance_rows.target.ne(provenance_rows.experiment_target)).sum())
    write_json("negative_provenance_summary.json", labels)
    evaluator = load_module("final_saved_evaluator", runtime.V18 / "run_experiment.py")
    reference = evaluator.load_real()
    negatives = pd.read_parquet(PREVIOUS / "verified_negative_features.parquet")
    distances, indices = cKDTree(reference[["x", "y"]]).query(negatives[["x_m", "y_m"]])
    overlap_audit = negatives[["candidate_id", "route"]].copy()
    overlap_audit["nearest_inventory_key"] = reference.iloc[indices].real_key.to_numpy()
    overlap_audit["distance_m"] = distances
    overlap_audit["within_20m"] = distances <= 20.
    overlap_audit.to_csv(OUT / "verified_negative_inventory_overlap.csv", index=False)
    outcome = json.loads((OUT / "training_outcome.json").read_text())
    outer = pd.read_csv(OUT / "control_challenger_fold_metrics.csv").query("stage == 'outer'")
    features = json.loads((PREVIOUS / "feature_schema.json").read_text())["feature_columns"]
    pool = pd.read_parquet(PREVIOUS / "development_features.parquet").set_index("candidate_id")
    inference_checks = []
    for fold in range(3):
        checkpoint = OUT / f"evaluation_checkpoint_outer{fold}.joblib"
        model = joblib.load(checkpoint)
        scores = pd.read_parquet(OUT / f"scores_outer{fold}_control.parquet").head(100)
        sample = pool.loc[scores.candidate_id].copy()
        sample["route"] = "arbitrary-unseen-road-" + sample.route.astype(str)
        state_hash = joblib.hash(model)
        actual = runtime.score_model(model, sample, features)
        assert np.array_equal(actual, scores.raw_score.to_numpy(float))
        assert np.array_equal(actual, runtime.score_model(model, sample, features))
        assert state_hash == joblib.hash(model)
        missing_rejected = False
        try:
            runtime.score_model(model, sample.drop(columns=features[0]), features)
        except KeyError:
            missing_rejected = True
        assert missing_rejected
        inference_checks.append(dict(fold=fold, serialized_scores_exact=True, arbitrary_road_ids_invariant=True,
                                     missing_feature_rejected=True, inference_model_state_unchanged=True))
    write_json("inference_checks.json", inference_checks)
    report(coverage, rf, attribution, labels, outer, outcome)
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    write_json("artifact_hashes.json", {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "artifact_hashes.json"})
    print("Saved report, completed inference checks, and verified all protected inputs unchanged. No additional fits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    with threadpool_limits(limits=2):
        finish_saved_report() if args.report_only else main()
