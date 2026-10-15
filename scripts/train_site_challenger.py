"""Execute a bounded nested-route experiment using the frozen 48-feature schema.

Research outputs only. Unmatched inventory candidates never become negatives.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from culvert_ai.flow_c_bundle import sha256_file
from culvert_ai.inference import RouteZConfig, dynamic_route_z
from culvert_ai.inference import add_route_local_density_normalization


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/experiments"
OUT = EXP / "site_challenger_20261015"
V8 = EXP / "nysdot_ulster_dem_hydrology_v8_20261005"
V18 = EXP / "nysdot_ulster_ranking_v18_20261006"
POOL = EXP / "nysdot_ulster_consensus_precision_20261013/candidate_consensus_features.parquet"
POLICIES = ("raw", "dynamic_route_z", "robust_route")
CONFIGS = [
    dict(id="HGB_COMPACT", family="hgb", learning_rate=.05, trees=100, leaves=15, leaf_min=20, l2=1., weighting="site_balanced"),
    dict(id="HGB_WIDE", family="hgb", learning_rate=.05, trees=160, leaves=31, leaf_min=20, l2=1., weighting="site_balanced"),
    dict(id="HGB_REGULAR", family="hgb", learning_rate=.04, trees=200, leaves=7, leaf_min=40, l2=10., weighting="route_balanced"),
    dict(id="HGB_SHALLOW", family="hgb", learning_rate=.03, trees=240, leaves=4, leaf_min=60, l2=20., weighting="route_balanced"),
    dict(id="GB_REGULAR", family="gb", learning_rate=.04, trees=120, leaves=7, leaf_min=30, l2=None, weighting="site_balanced"),
    dict(id="GB_SHALLOW", family="gb", learning_rate=.03, trees=180, leaves=4, leaf_min=40, l2=None, weighting="route_balanced"),
    dict(id="PAIR_C01", family="pairwise", C=.1, weighting="site_balanced"),
    dict(id="PAIR_C1", family="pairwise", C=1., weighting="route_balanced"),
    dict(id="RF_CONTROL", family="rf", trees=800, leaf_min=2, weighting="site_balanced"),
]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def score_model(fitted, frame, features):
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(fitted["medians"])
    if fitted["family"] == "pairwise":
        x = (x - fitted["medians"]) / fitted["scales"]
        return fitted["model"].decision_function(x.to_numpy(float))
    return fitted["model"].predict_proba(x)[:, 1]


def fit_model(config, train, features):
    x = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x.median().fillna(0.)
    x = x.fillna(medians)
    weight = train["weight"].to_numpy(float).copy()
    if config["weighting"] == "route_balanced":
        for label in (0, 1):
            idx = train["target"].eq(label)
            totals = train.loc[idx].groupby("route")["weight"].transform("sum")
            weight[idx] /= totals.to_numpy(float)
    # Equal total mass for observed positive and verified negative classes.
    for label in (0, 1):
        idx = train["target"].eq(label).to_numpy()
        if not idx.any():
            raise RuntimeError("Fold has no observed examples of one class.")
        weight[idx] *= .5 * len(train) / weight[idx].sum()
    result = dict(family=config["family"], medians=medians)
    if config["family"] == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=config["learning_rate"], max_iter=config["trees"],
            max_leaf_nodes=config["leaves"], min_samples_leaf=config["leaf_min"],
            l2_regularization=config["l2"], early_stopping=False, random_state=42,
        )
        model.fit(x, train["target"], sample_weight=weight)
    elif config["family"] == "gb":
        model = GradientBoostingClassifier(
            learning_rate=config["learning_rate"], n_estimators=config["trees"],
            max_leaf_nodes=config["leaves"], max_depth=None,
            min_samples_leaf=config["leaf_min"], subsample=.8, random_state=42,
        )
        model.fit(x, train["target"], sample_weight=weight)
    elif config["family"] == "rf":
        model = RandomForestClassifier(n_estimators=800, min_samples_leaf=2,
                                       class_weight="balanced_subsample", random_state=42, n_jobs=2)
        model.fit(x, train["target"], sample_weight=weight)
    else:
        scales = (x.quantile(.75) - x.quantile(.25)).where(lambda s: s > 1e-6, 1.)
        scaled = ((x - medians) / scales).to_numpy(float)
        # At most one positive representative per authoritative site. Pairs use
        # only observed positives and FIELD VERIFIED negatives, never unknowns.
        reps = train.loc[train.target.eq(1)].drop_duplicates("site_key").index
        negatives = np.flatnonzero(train.target.eq(0).to_numpy())
        rng = np.random.default_rng(42)
        left, right = [], []
        for pos in reps:
            chosen = rng.choice(negatives, size=min(12, len(negatives)), replace=False)
            left.extend([pos] * len(chosen))
            right.extend(chosen)
        delta = scaled[np.asarray(left)] - scaled[np.asarray(right)]
        pair_weight = np.sqrt(weight[np.asarray(left)] * weight[np.asarray(right)])
        model = LogisticRegression(C=config["C"], fit_intercept=False, max_iter=2000, random_state=42)
        model.fit(np.concatenate([delta, -delta]), np.r_[np.ones(len(delta)), np.zeros(len(delta))],
                  sample_weight=np.r_[pair_weight, pair_weight])
        result["scales"] = scales
    result["model"] = model
    return result


def normalize(frame, policy, fallback):
    if policy == "raw":
        return frame["raw_score"].to_numpy(float)
    if policy == "dynamic_route_z":
        return dynamic_route_z(
            frame, score_column="raw_score", route_column="route",
            config=RouteZConfig(global_center=fallback[0], global_scale=fallback[1]),
        )["route_z_score"].to_numpy(float)
    result = np.zeros(len(frame))
    for indices in frame.groupby("route").indices.values():
        values = frame.iloc[indices]["raw_score"].to_numpy(float)
        center = float(np.median(values))
        scale = float((np.quantile(values, .75) - np.quantile(values, .25)) / 1.349)
        if len(values) < 4 or scale < 1e-6:
            center, scale = fallback
        result[indices] = (values - center) / max(scale, 1e-6)
    return result


def sites_for(frame, policy, fallback, evaluator):
    ranked = frame.copy().reset_index(drop=True)
    ranked["priority"] = normalize(ranked, policy, fallback)
    # Production-compatible tie ordering for all controlled comparisons.
    ordered = ranked.sort_values(["priority", "raw_score", "candidate_id"],
                                 ascending=[False, False, True], kind="stable").reset_index(drop=True)
    xy = ordered[["x_m", "y_m"]].to_numpy(float)
    tree = cKDTree(xy)
    suppressed = np.zeros(len(ordered), bool)
    selected = []
    for i in range(len(ordered)):
        if not suppressed[i]:
            selected.append(i)
            suppressed[tree.query_ball_point(xy[i], 20.)] = True
    sites = ordered.iloc[selected].copy().reset_index(drop=True)
    sites["rank"] = np.arange(1, len(sites) + 1)
    sites["site_id"] = "challenger_" + sites.candidate_id
    return sites


def metrics(sites, real, evaluator):
    xy = real[["x", "y"]].to_numpy(float)
    result = {}
    for k in (25, 50, 100, 250):
        matches, _ = evaluator.match_sites(sites, xy, top_k=k)
        result[f"confirmed_at_{k}"] = len(matches)
        result[f"inventory_matched_fraction_at_{k}"] = len(matches) / min(k, len(sites))
    result["site_recall_at_250"] = result["confirmed_at_250"] / len(real) if len(real) else None
    counts = sites.head(250).route.value_counts()
    result["routes_at_250"] = len(counts)
    result["largest_route"] = str(counts.index[0])
    result["largest_route_share"] = float(counts.iloc[0] / min(250, len(sites)))
    return result


def objective(m):
    # Predeclared, equal emphasis on inventory-matched fraction at 100 and 250.
    return .5 * (m["inventory_matched_fraction_at_100"] + m["inventory_matched_fraction_at_250"])


def training_rows(pool, negatives, real, train_routes, heldout_xy):
    parts = []
    for route in sorted(train_routes):
        part = pool.loc[pool.route.eq(route)].copy()
        official = real.loc[real.route.eq(route)]
        if part.empty or official.empty:
            continue
        distance, nearest = cKDTree(official[["x", "y"]]).query(part[["x_m", "y_m"]])
        positive = distance <= 20.
        part = part.loc[positive].copy()
        part["site_key"] = official.iloc[nearest[positive]].real_key.to_numpy()
        part["target"] = 1
        part["label_source"] = "authoritative_inventory_positive_20m"
        parts.append(part)
    neg = negatives.loc[negatives.route.isin(train_routes)].copy()
    neg["target"] = 0
    neg["site_key"] = "field_negative_" + neg.candidate_id
    neg["label_source"] = "field_verified_negative"
    train = pd.concat([*parts, neg], ignore_index=True)
    if len(heldout_xy):
        distance = cKDTree(heldout_xy).query(train[["x_m", "y_m"]])[0]
        # Unlabeled geometry only: purge candidates within 40 m of held-out road
        # candidates/negative locations, including junction duplicates.
        train = train.loc[distance > 40.].copy()
    train = train.drop_duplicates("candidate_id").reset_index(drop=True)
    train["weight"] = 1. / train.groupby(["target", "site_key"]).candidate_id.transform("size")
    return train


def inner_groups(routes):
    # Deterministic route-only grouping. Auxiliary routes with field negatives
    # remain development training groups even when absent from inspection pool.
    values = sorted(routes)
    return [set(values[::2]), set(values[1::2])]


def choose(rows):
    return sorted(rows, key=lambda r: (-r["objective"], -r["confirmed_at_250"],
                                      -r["confirmed_at_100"], r["config"], POLICIES.index(r["policy"])))[0]


def protected_hashes():
    paths = [ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg",
             ROOT / "data/processed/hydrology/usgs_3dep_10m/candidate_hydrology_features.parquet",
             POOL, V18 / "out_of_fold_predictions/flow_c.parquet",
             V18 / "out_of_fold_predictions/pointwise_gb__authoritative_inventory.parquet",
             EXP / "nysdot_ulster_score_comparability_v19_20261006/out_of_fold_scores/route-z.parquet",
             EXP / "nysdot_ulster_real_inventory_benchmark_v1_20261006/source_extract/small_culverts_r8_source.csv"]
    for directory in ("web", "api", "server", "models", "src/culvert_ai"):
        paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in str(p)]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}


def prepare():
    evaluator = module("challenger_v18", V18 / "run_experiment.py")
    runner = module("challenger_v7", EXP / "nysdot_ulster_dem_hydrology_v7_completed_20260924/run_experiment.py")
    print("Loading existing feature artifacts and frozen transformations", flush=True)
    # Read frozen physical features directly. The old _prepare_experiment()
    # assigns some routes from authoritative matches; do not call it here.
    frame = gpd.read_file(ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg")
    frame["candidate_id"] = frame.candidate_id.astype(str)
    raw_routes = frame.matched_route.map(evaluator.norm_route)
    samples = frame.loc[frame.source.eq("route_interval_sample") & raw_routes.ne("<unassigned>")]
    distance, nearest = cKDTree(np.c_[samples.geometry.x, samples.geometry.y]).query(
        np.c_[frame.geometry.x, frame.geometry.y])
    inferred = samples.iloc[nearest].matched_route.map(evaluator.norm_route).to_numpy()
    missing = raw_routes.eq("<unassigned>") & (distance <= 100.)
    raw_routes.loc[missing] = inferred[missing]
    frame["road_group"] = raw_routes
    # Existing 38-feature HYDRO-C artifacts, no recomputed GIS measurements.
    hydro_path = runner.V6.HYDRO_FEATURES_PATH
    hydro = pd.read_parquet(hydro_path).drop(columns="geometry")
    for column in hydro.columns.difference(["candidate_id"]):
        if column in frame:
            frame.drop(columns=column, inplace=True)
    frame = frame.merge(hydro, on="candidate_id", validate="one_to_one")
    frame = runner._add_dem_features(frame)
    frame = add_route_local_density_normalization(frame, route_column="road_group")
    features = json.loads((V8 / "exact_feature_lists.json").read_text())["hydro_c_plus_flow_c"]
    # Keep only the declared numeric features and source identifiers/geometry.
    frozen_pool = pd.read_parquet(POOL, columns=["candidate_id", "route", "x_m", "y_m"])
    pool = frozen_pool.rename(columns={"route": "historical_route"}).merge(
        pd.DataFrame(frame[["candidate_id", "road_group", *features]]), on="candidate_id", validate="one_to_one")
    pool["route"] = pool.road_group
    negative_ids = pd.read_csv(V8 / "training_rows.csv").query("training_label_class == 'verified_negative'")
    neg = frame.loc[frame.candidate_id.isin(negative_ids.candidate_id)].copy()
    metric = gpd.GeoDataFrame(neg, geometry="geometry", crs=frame.crs).to_crs("EPSG:26918")
    neg["x_m"], neg["y_m"] = metric.geometry.x, metric.geometry.y
    neg["route"] = neg.road_group
    neg = pd.DataFrame(neg[["candidate_id", "route", "x_m", "y_m", *features]])
    neg = neg.loc[neg.route.ne("<unassigned>")].copy().reset_index(drop=True)
    assert len(pool) == 51417 and pool.candidate_id.is_unique
    assert len(features) == 48
    assert all("label" not in f and "target" not in f and "known" not in f for f in features)
    save_json(OUT / "road_group_audit.json", dict(
        source="matched_route or nearest existing interval road sample within 100m; no inventory route overrides",
        changed_from_historical=int(pool.route.ne(pool.historical_route).sum()),
        label_free_feature_preprocessing=True,
        negatives_by_route=neg.route.value_counts().to_dict(),
        interrupted_legacy_pass="initial inner fits interrupted before any selection; old route-derived preprocessing discarded",
        hydrology_feature_artifact=str(hydro_path.relative_to(ROOT))))
    save_json(OUT / "feature_schema.json", {"feature_columns": features, "numeric_cast": "float64", "feature_crs": "EPSG:32618", "evaluation_crs": "EPSG:26918"})
    pool.to_parquet(OUT / "development_features.parquet", index=False)
    neg.to_parquet(OUT / "verified_negative_features.parquet", index=False)
    return evaluator, pool, neg, evaluator.load_real(), features


def run_inner(pool, neg, real, features, allowed, scope, evaluator):
    score_parts = {c["id"]: [] for c in CONFIGS}
    audit = []
    for fold, heldout in enumerate(inner_groups(allowed)):
        train_routes = allowed - heldout
        test = pool.loc[pool.route.isin(heldout)].copy()
        if test.empty:
            continue
        heldout_xy = pd.concat([test, neg.loc[neg.route.isin(heldout)]])[["x_m", "y_m"]].to_numpy(float)
        train = training_rows(pool, neg, real, train_routes, heldout_xy)
        assert not set(train.route) & heldout
        audit.append(dict(scope=scope, inner_fold=fold, training_routes=sorted(train_routes),
                          validation_routes=sorted(heldout), rows=len(train),
                          positives=int(train.target.sum()), negatives=int(train.target.eq(0).sum()),
                          min_train_validation_distance_m=float(cKDTree(heldout_xy).query(train[["x_m", "y_m"]])[0].min())))
        for config in CONFIGS:
            fitted = fit_model(config, train, features)
            part = test[["candidate_id", "route", "x_m", "y_m"]].copy()
            part["raw_score"] = score_model(fitted, test, features)
            score_parts[config["id"]].append(part)
            print(f"{scope} inner={fold} {config['id']} trained rows={len(train)}", flush=True)
    rows = []
    for config in CONFIGS:
        predicted = pd.concat(score_parts[config["id"]], ignore_index=True)
        fallback = (float(predicted.raw_score.mean()), max(float(predicted.raw_score.std(ddof=0)), 1e-6))
        for policy in POLICIES:
            sites = sites_for(predicted, policy, fallback, evaluator)
            m = metrics(sites, real.loc[real.route.isin(allowed)], evaluator)
            rows.append(dict(scope=scope, config=config["id"], policy=policy, objective=objective(m), **m))
    return choose(rows), rows, audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "COMPLETE.json").exists():
        raise RuntimeError("Completed experiment already exists; refusing to overwrite it.")
    if (OUT / "preregistered_config.json").exists():
        previous = json.loads((OUT / "preregistered_config.json").read_text())
        if previous["configurations"] != CONFIGS:
            assert not list(OUT.glob("fold_*.joblib")), "Cannot change configurations after outer fold completion"
            save_json(OUT / "interrupted_initial_declaration.json", previous)
    before = protected_hashes()
    save_json(OUT / "protected_before.json", before)
    save_json(OUT / "preregistered_config.json", dict(
        configurations=CONFIGS, candidate_configurations=len(CONFIGS), policies=POLICIES,
        primary_objective="0.5 * (inventory-matched fraction@100 + inventory-matched fraction@250)",
        tie_break="confirmed@250, confirmed@100, configuration ID, declared policy order",
        validation="leave-one-complete-label-free-road-group-out outer folds; two deterministic route groups inner; separate all-development two-fold selection for final fit",
        positives="training-route authoritative inventory <=20m; one unit weight per authoritative site",
        negatives="field-verified negatives only; unknown candidates never target=0",
        purge="40m training exclusion around all validation/held-out candidate and negative geometry",
        no_external_boosting="LightGBM/XGBoost/CatBoost unavailable; sklearn regularized GradientBoostingClassifier used",
        final_fit_is_independent_validation=False,
    ))
    evaluator, pool, neg, real, features = prepare()
    routes = sorted(r for r in pool.route.unique() if r != "<unassigned>")
    all_routes = set(routes) | set(neg.route.unique())
    outer_parts, selections, inner_rows, leakage = [], [], [], []
    for route in routes:
        checkpoint = OUT / f"fold_{route}.joblib"
        if checkpoint.exists():
            stored = joblib.load(checkpoint)
            print(f"Resuming completed outer route {route}", flush=True)
        else:
            allowed = all_routes - {route}
            # Remove outer validation geometry BEFORE inner training, including
            # nearby cross-route duplicate sites. This depends on no outcomes.
            outer_test = pool.loc[pool.route.eq(route)].copy()
            heldout_xy = pd.concat([outer_test, neg.loc[neg.route.eq(route)]])[["x_m", "y_m"]].to_numpy(float)
            safe_pool = pool.loc[pool.route.ne(route)].copy()
            safe_neg = neg.loc[neg.route.ne(route)].copy()
            for frame in (safe_pool, safe_neg):
                distances = cKDTree(heldout_xy).query(frame[["x_m", "y_m"]])[0]
                frame.drop(frame.index[distances <= 40.], inplace=True)
            selected, rows, audit = run_inner(safe_pool, safe_neg, real.loc[real.route.ne(route)], features, allowed, f"outer_{route}", evaluator)
            train = training_rows(pool, neg, real, allowed, heldout_xy)
            outer_predictions = []
            for config in CONFIGS:
                fitted = fit_model(config, train, features)
                scored = outer_test[["candidate_id", "route", "x_m", "y_m"]].copy()
                scored["raw_score"] = score_model(fitted, outer_test, features)
                ref = score_model(fitted, pool.loc[pool.route.isin(allowed)], features)
                fallback = (float(np.mean(ref)), max(float(np.std(ref)), 1e-6))
                for policy in POLICIES:
                    scored[policy] = normalize(scored, policy, fallback)
                scored["config"] = config["id"]
                scored["outer_route"] = route
                scored["selected"] = config["id"] == selected["config"]
                scored["selected_policy"] = selected["policy"]
                outer_predictions.append(scored)
            # Persist exactly which labels trained each fold for provenance.
            train[["candidate_id", "route", "target", "label_source", "site_key", "weight"]].to_csv(OUT / f"training_rows_heldout_{route}.csv", index=False)
            stored = dict(predictions=pd.concat(outer_predictions, ignore_index=True), selected=selected, rows=rows,
                          leakage=audit + [dict(scope=f"outer_{route}", training_routes=sorted(allowed), validation_routes=[route],
                                               positives=int(train.target.sum()), negatives=int(train.target.eq(0).sum()), rows=len(train),
                                               min_train_validation_distance_m=float(cKDTree(heldout_xy).query(train[["x_m", "y_m"]])[0].min()))])
            joblib.dump(stored, checkpoint)
        outer_parts.append(stored["predictions"])
        selections.append(stored["selected"])
        inner_rows.extend(stored["rows"])
        leakage.extend(stored["leakage"])
        print(f"FINISHED outer={route} inner selected={stored['selected']['config']} / {stored['selected']['policy']}", flush=True)

    oof = pd.concat(outer_parts, ignore_index=True)
    oof.to_parquet(OUT / "outer_oof_predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(OUT / "outer_fold_selections.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "inner_selection_metrics.csv", index=False)
    save_json(OUT / "leakage_checks.json", leakage)
    final_selection, final_inner, final_audit = run_inner(pool, neg, real, features, all_routes, "final_development_selection", evaluator)
    save_json(OUT / "final_selection.json", final_selection)
    pd.DataFrame(final_inner).to_csv(OUT / "final_development_selection_metrics.csv", index=False)
    save_json(OUT / "final_selection_leakage.json", final_audit)
    final_config = next(c for c in CONFIGS if c["id"] == final_selection["config"])
    train = training_rows(pool, neg, real, all_routes, np.empty((0, 2)))
    fitted = fit_model(final_config, train, features)
    reference = score_model(fitted, pool, features)
    fallback = (float(np.mean(reference)), max(float(np.std(reference)), 1e-6))
    bundle = dict(**fitted, feature_columns=features, configuration=final_config,
                  normalization=final_selection["policy"], normalization_fallback=fallback,
                  radius_m=20., feature_crs="EPSG:32618", evaluation_crs="EPSG:26918",
                  status="experimental_not_deployed", message="Recommendations require field verification.",
                  preprocessing="numeric float64; infinities to NaN; training-only feature medians; pairwise only: training IQR scaling",
                  training_rows=len(train), independent_validation=False)
    joblib.dump(bundle, OUT / "challenger.joblib", compress=3)
    train[["candidate_id", "route", "target", "label_source", "site_key", "weight"]].to_csv(OUT / "final_training_rows.csv", index=False)
    # A uniform bijection of arbitrary route identifiers must preserve scores,
    # normalization groups, and output ranks, including after serialization.
    sample = pool.head(1000).copy().reset_index(drop=True)
    original = score_model(bundle, sample, features)
    loaded = joblib.load(OUT / "challenger.joblib")
    sample["route"] = "unseen-road-" + sample.route.astype(str)
    repeated = score_model(loaded, sample, features)
    assert np.array_equal(original, repeated)
    first = pool.head(1000)[["candidate_id", "route", "x_m", "y_m"]].copy().reset_index(drop=True)
    first["raw_score"] = original
    renamed = first.copy()
    renamed["route"] = "arbitrary-" + renamed.route.astype(str)
    assert np.array_equal(normalize(first, bundle["normalization"], fallback), normalize(renamed, bundle["normalization"], fallback))
    results = []
    def evaluate_vector(frame, priority, name):
        # Already normalized OOF scores: preserve per-fold fallback, then use
        # precisely the same score/secondary-score/ID physical-site rule.
        part = frame.copy()
        part["raw_score_backup"] = part.raw_score
        # sites_for(raw) only needs the final priority. Pre-sort by secondary raw
        # score for tie handling via a tiny helper implemented independently below.
        sites = saved_priority_sites(part, priority)
        results.append(dict(system=name, **metrics(sites, real, evaluator)))
        sites.head(250).to_csv(OUT / f"top250_{name}.csv", index=False)
        return sites
    all_sites = {}
    for config in CONFIGS:
        subset = oof.loc[oof.config.eq(config["id"])].copy()
        for policy in POLICIES:
            all_sites[f"{config['id']}__{policy}"] = evaluate_vector(subset, policy, f"{config['id']}__{policy}")
    nested = oof.loc[oof.selected].copy()
    nested["nested_priority"] = [row[row.selected_policy] for _, row in nested.iterrows()]
    nested_sites = evaluate_vector(nested, "nested_priority", "NESTED_SELECTION_PROCEDURE")
    pd.DataFrame(results).to_csv(OUT / "outer_oof_metrics.csv", index=False)
    fold_results = []
    for route in routes:
        part = nested.loc[nested.route.eq(route)]
        sites = saved_priority_sites(part, "nested_priority")
        fold_results.append(dict(route=route, **metrics(sites, real.loc[real.route.eq(route)], evaluator)))
    pd.DataFrame(fold_results).to_csv(OUT / "selected_procedure_route_metrics.csv", index=False)
    top = nested_sites.head(250)
    distribution = top.route.value_counts().rename_axis("route").reset_index(name="recommendations")
    distribution["share"] = distribution.recommendations / len(top)
    distribution.to_csv(OUT / "nested_top250_route_distribution.csv", index=False)
    stability = []
    for name, sites in all_sites.items():
        ids = set(sites.head(250).candidate_id)
        other = set(nested_sites.head(250).candidate_id)
        stability.append(dict(system=name, top250_candidate_jaccard_with_nested=len(ids & other)/len(ids | other)))
    pd.DataFrame(stability).to_csv(OUT / "ranking_stability.csv", index=False)
    # Match historical artifacts with the original definitions, plus controlled
    # RF comparison using the same current secondary-score tie-break.
    baseline_results = []
    base = pool[["candidate_id", "historical_route", "x_m", "y_m"]].rename(columns={"historical_route": "route"}).copy()
    rf = pd.read_parquet(V18 / "out_of_fold_predictions/flow_c.parquet").set_index("candidate_id").outer_oof_score
    z = pd.read_parquet(EXP / "nysdot_ulster_score_comparability_v19_20261006/out_of_fold_scores/route-z.parquet").set_index("candidate_id").outer_oof_score
    base["raw_score"] = base.candidate_id.map(rf)
    frozen = joblib.load(ROOT / "models/flow_c_frozen_v1.joblib")
    f = frozen["route_z_global"]
    for policy in POLICIES:
        sites = sites_for(base, policy, (f["center"], f["scale"]), evaluator)
        baseline_results.append(dict(system=f"SAVED_RF__{policy}", **metrics(sites, real, evaluator)))
    hist = base.copy()
    hist["historical_z"] = hist.candidate_id.map(z)
    sites = evaluator.nms_sites(hist, "historical_z", "HISTORICAL_POINTWISE_76_FEATURES")
    baseline_results.append(dict(system="HISTORICAL_POINTWISE_76_FEATURES", **metrics(sites, real, evaluator)))
    assert baseline_results[-1]["confirmed_at_250"] == 41
    assert baseline_results[1]["confirmed_at_250"] == 5
    pd.DataFrame(baseline_results).to_csv(OUT / "baseline_reproduction.csv", index=False)
    # Existing cross-fitted RF features inside the historical 76-field set can
    # carry outer-route labels indirectly: document rather than reuse them.
    historical = json.loads((V18 / "experiment_config.json").read_text())["feature_sets"]["D_ALL_VALID"]
    save_json(OUT / "historical_compatibility.json", dict(historical_features=historical,
        historical_only_features=[f for f in historical if f not in features], current_features=features,
        historical_configuration_by_fold=pd.read_csv(V18 / "nested_validation.csv").query(
            "method == 'POINTWISE-GB' and regime == 'AUTHORITATIVE-INVENTORY' and selected_config == True"
        )[["outer_heldout_route", "config"]].drop_duplicates().to_dict("records"),
        limitation="Historical 76-feature model contains saved RF OOF score-derived predictors, whose cross-fitting was not nested inside POINTWISE outer folds. No historical score-derived predictor enters the new 48-field experiment."))
    after = protected_hashes()
    assert before == after
    save_json(OUT / "protected_integrity.json", dict(unchanged=True, before=before, after=after))
    save_json(OUT / "manifest.json", dict(
        files={p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"},
        production_changed=False, deployed=False, total_candidate_configurations=len(CONFIGS),
        serialized_scoring_exact=True, arbitrary_route_rename_invariant=True))
    save_json(OUT / "COMPLETE.json", dict(status="complete", final_selection=final_selection,
                                         nested_procedure=results[-1], final_model=str(OUT / "challenger.joblib")))
    print(json.dumps(dict(final_selection=final_selection, nested_procedure=results[-1], results=results), indent=2), flush=True)


def saved_priority_sites(frame, priority):
    ordered = frame.sort_values([priority, "raw_score", "candidate_id"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    tree = cKDTree(ordered[["x_m", "y_m"]])
    suppressed = np.zeros(len(ordered), bool)
    selected = []
    for index, xy in enumerate(ordered[["x_m", "y_m"]].to_numpy(float)):
        if not suppressed[index]:
            selected.append(index)
            suppressed[tree.query_ball_point(xy, 20.)] = True
    sites = ordered.iloc[selected].copy().reset_index(drop=True)
    sites["rank"] = np.arange(1, len(sites)+1)
    sites["site_id"] = "site_" + sites.candidate_id
    return sites


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="New or interrupted experiment directory; completed runs cannot be overwritten")
    args = parser.parse_args()
    OUT = args.output.resolve()
    with threadpool_limits(limits=2):
        main()
