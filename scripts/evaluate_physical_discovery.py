"""Fixed physical-discovery variants on identical saved summer search cells.

Zero fitting. BASELINE retains exact current recommendations. DENSE is an
explicit computational ceiling, never a displayable detection configuration.
"""

# ruff: noqa: E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import hashlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point

from scripts.capture_summer_discovery import DEFAULT as OUT, PRIOR
from scripts.diagnose_summer_discovery import xy, own, nearest, TOLERANCES
from scripts.evaluate_hydrology_sites import protected
from scripts.validate_inventory_heldout_roads import match_one_to_one, fold_for
from staging.discovery_service import prepare, compatible_scores
from staging.discovery_sites import small_drainage, physical_groups, ordered
from staging.hydrology_service import datasets, HYDRO, load_model
from staging.hydrology_sites import physical_measurements, search_candidates, apply_feature_contract
from staging.hybrid_sites import attach_road_topology
from culvert_ai.flow_c_bundle import sha256_file

VARIANTS = ("BASELINE", "DENSE", "STREAM", "DEM", "HYBRID", "HYBRID-GROUPED")
KS = (25, 50, 100, 250, 500, 1000)


def generation_signature():
    sources = [
        inspect.getsource(cell),
        inspect.getsource(score_existing_road_holdouts),
        inspect.getsource(road_heldout_model),
    ]
    sources += [
        (ROOT / p).read_text()
        for p in ["staging/discovery_sites.py", "staging/discovery_service.py"]
    ]
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


@lru_cache(maxsize=6)
def road_heldout_model(family, fold):
    return joblib.load(
        ROOT
        / f"reports/experiments/hydrology_driven_sites_20261015/hydrology-sites-v2_{family}_fold{fold}.joblib"
    )


def score_existing_road_holdouts(raw, measured, center):
    prior = ROOT / "reports/experiments/real_inventory_heldout_roads_20261015"
    roads = gpd.read_parquet(prior / "recovered_road_segments.parquet")
    route = roads.set_index(roads.segment_id.astype(str)).corridor
    raw["heldout_corridor"] = raw.road_id.astype(str).map(route)
    raw["heldout_fold"] = raw.heldout_corridor.map(
        lambda r: fold_for(str(r)) if pd.notna(r) else -1
    )
    context = measured.loc[
        measured.geometry.distance(center).le(1500)
        & measured.proposal_kind.eq("internal_road_probe")
    ]
    for family in ["RF", "HGB"]:
        raw[f"score_{family}_roadheldout"] = np.nan
        for fold in range(3):
            mask = raw.heldout_fold.eq(fold)
            if not mask.any():
                continue
            manifest = pd.read_csv(prior / f"training_manifest_{family}_fold{fold}.csv")
            assert set(raw.loc[mask, "heldout_corridor"].astype(str)).isdisjoint(
                set(manifest.corridor.astype(str))
            )
            _, scores, _ = compatible_scores(
                raw.loc[mask], context, road_heldout_model(family, fold)
            )
            raw.loc[mask, f"score_{family}_roadheldout"] = scores
    return raw


def cell(task):
    ix, iy, lat, lon = task
    source = OUT / "cells" / f"{ix}_{iy}"
    folder = source / "challenger"
    done = folder / "COMPLETE.json"
    if done.exists():
        saved = json.loads(done.read_text())
        assert saved["generation_signature"] == generation_signature(), (
            "Changed generator: use a new experimental output directory"
        )
        return saved
    folder.mkdir(exist_ok=True)
    if not json.loads((source / "COMPLETE.json").read_text())["success"]:
        return dict(cell_x=ix, cell_y=iy, success=False)
    center = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32618).iloc[0]
    roads = gpd.read_parquet(source / "roads.parquet")
    roads["route_id"] = roads.name.fillna(roads.id.astype(str)).astype(str)
    # Road geometries saved through GeoJSON differ by subnanometre roundoff.
    # Use original native geometries/attributes for all new generation instead.
    all_roads, streams, _ = datasets()
    roads = (
        all_roads.loc[all_roads.id.astype(str).isin(roads.id.astype(str))]
        .copy()
        .reset_index(drop=True)
    )
    roads["route_id"] = roads.name.fillna(roads.id.astype(str)).astype(str)
    measured = gpd.read_parquet(source / "measurements.parquet")
    small, links = small_drainage(roads, HYDRO, center.buffer(1580).bounds)
    small = small.loc[small.geometry.distance(center).le(1560)].reset_index(drop=True)
    small_measured = physical_measurements(small, all_roads, streams, HYDRO) if len(small) else None
    raw, transformed = prepare(measured, center, 1500, small_measured)
    raw = score_existing_road_holdouts(raw, measured, center)
    previous = gpd.read_parquet(source / "raw_features.parquet")
    check = raw.set_index("candidate_id").loc[previous.candidate_id]
    np.testing.assert_allclose(
        check.loc[previous.inside_query.to_numpy(), "score_RF"],
        previous.loc[previous.inside_query, "model_score"],
        atol=1e-12,
        rtol=0,
    )
    columns = load_model("RF")["feature_columns"]
    np.testing.assert_allclose(
        check[columns], previous[columns], equal_nan=True, atol=1e-10, rtol=1e-12
    )
    inner = raw.loc[raw.inside_query].copy()
    baseline = gpd.read_parquet(source / "sites.parquet")
    score_lookup = inner.set_index("candidate_id")
    for col in [
        "score_RF",
        "score_HGB",
        "score_RF_roadheldout",
        "score_HGB_roadheldout",
        "heldout_corridor",
        "heldout_fold",
        *columns,
    ]:
        baseline[col] = baseline.candidate_id.map(score_lookup[col])
    hybrid = inner.loc[inner.discovery_admitted].copy()
    variants = dict(
        BASELINE=baseline,
        DENSE=inner.loc[inner.proposal_kind.eq("internal_road_probe")].copy(),
        STREAM=inner.loc[inner.proposal_kind.eq("mapped_crossing")].copy(),
        DEM=inner.loc[inner.proposal_kind.isin(["dem_crossing", "small_dem_crossing"])].copy(),
        HYBRID=hybrid,
        **{"HYBRID-GROUPED": physical_groups(hybrid)},
    )
    for name, frame in variants.items():
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    # Retain full measurements and exact features for distribution/parity audits.
    raw.to_parquet(folder / "raw.parquet", index=False)
    for family, frame in transformed.items():
        frame.to_parquet(folder / f"features_{family}.parquet", index=False)
    links.to_parquet(folder / "small_drainage.parquet", index=False)
    controls = {}
    for spacing in [75, 50, 25, 10]:
        samples = attach_road_topology(search_candidates(roads, interval_m=spacing), roads)
        samples = samples.loc[samples.geometry.distance(center).le(1500)].copy()
        samples.to_parquet(folder / f"sampling_{spacing}m.parquet", index=False)
        controls[spacing] = len(samples)
    result = dict(
        cell_x=ix,
        cell_y=iy,
        success=True,
        counts={k: len(v) for k, v in variants.items()},
        raw=len(inner),
        small_crossings=len(small),
        controls=controls,
        baseline_score_and_feature_parity=True,
        generation_signature=generation_signature(),
    )
    done.write_text(json.dumps(result, indent=2) + "\n")
    return result


def pools(tasks):
    grouped = {
        v: []
        for v in [
            *VARIANTS,
            "BASELINE-RAW",
            "sampling_75m",
            "sampling_50m",
            "sampling_25m",
            "sampling_10m",
        ]
    }
    for ix, iy, _, _ in tasks:
        source = OUT / "cells" / f"{ix}_{iy}"
        if not json.loads((source / "COMPLETE.json").read_text())["success"]:
            continue
        for name in grouped:
            frame = gpd.read_parquet(
                source
                / (
                    "raw_features.parquet"
                    if name == "BASELINE-RAW"
                    else f"challenger/{name}.parquet"
                )
            )
            if name == "BASELINE-RAW":
                frame = frame.loc[frame.inside_query].copy()
            if name in VARIANTS:
                for family in ["RF", "HGB"]:
                    # Rank BEFORE owner clipping for an exact per-query budget.
                    ranked = ordered(
                        frame.assign(model_score=frame[f"score_{family}"]), policy="raw_rf"
                    )
                    ranks = ranked.set_index("candidate_id").global_rank
                    frame[f"query_rank_{family}"] = frame.candidate_id.map(ranks)
            frame = own(frame, ix, iy)
            frame["cell_x"], frame["cell_y"] = ix, iy
            grouped[name].append(frame)
    return {
        name: gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=32618)
        .sort_values(["cell_x", "cell_y", "candidate_id"], kind="stable")
        .reset_index(drop=True)
        for name, parts in grouped.items()
    }


def summarize(frames, truth):
    supported = (
        pd.read_csv(PRIOR / "regional_record_audit.csv").set_index("record_id").query_success
    )
    target = truth.loc[truth.record_id.map(supported)].reset_index(drop=True)
    rows, ranks, routes, records = [], [], [], []
    concentrations, holdouts = [], []
    tx = xy(target)
    for name, frame in frames.items():
        assert frame.candidate_id.is_unique
        frame.to_parquet(OUT / f"variant_{name}.parquet", index=False)
        distance, positions = nearest(frame, target)
        for tolerance in TOLERANCES:
            matches = match_one_to_one(frame, tx, tolerance)
            rows.append(
                dict(
                    variant=name,
                    stage="raw_candidates"
                    if name in ["BASELINE-RAW", "DENSE"] or name.startswith("sampling")
                    else "site_hypotheses",
                    tolerance_m=tolerance,
                    candidates=len(frame),
                    unique_coordinates=len(np.unique(xy(frame), axis=0)),
                    inventory=len(target),
                    covered_any=int((distance <= tolerance).sum()),
                    matched=len(matches),
                    missed=len(target) - len(matches),
                    recall=len(matches) / len(target),
                    nearest_median_m=float(np.median(distance)),
                    nearest_p90_m=float(np.quantile(distance, 0.9)),
                    matched_median_m=float(matches.distance_m.median()) if len(matches) else None,
                )
            )
        matches = match_one_to_one(frame, tx, 20)
        matched_records = set(target.iloc[matches.real_row].record_id)
        for i, record in target.iterrows():
            records.append(
                dict(
                    variant=name,
                    record_id=record.record_id,
                    source_excel_row=int(record.source_excel_row),
                    region=record.Region,
                    route=record.Route,
                    nearest_candidate_id=frame.iloc[positions[i]].candidate_id
                    if len(frame)
                    else None,
                    nearest_distance_m=distance[i],
                    matched20=record.record_id in matched_records,
                )
            )
        for (region, route), actual in target.groupby(["Region", "Route"]):
            found = int(actual.record_id.isin(matched_records).sum())
            routes.append(
                dict(
                    variant=name,
                    region=region,
                    route=route,
                    development_road_fold=fold_for(str(route)),
                    records=len(actual),
                    matched20=found,
                    recall=found / len(actual),
                    independent=False,
                )
            )
        if name not in VARIANTS:
            continue
        for family in ["RF", "HGB"]:
            for policy in ["raw_rf", "dynamic_route_z"]:
                # Global policies recomputed on the SAME owned representatives.
                # These are not pooled per-query z-scores or the app's Top25.
                queue = ordered(frame.assign(model_score=frame[f"score_{family}"]), policy=policy)
                for k in KS:
                    selected = queue.iloc[:k].reset_index(drop=True)
                    match = match_one_to_one(selected, tx, 20)
                    counts = selected.groupby("route_id").size()
                    ranks.append(
                        dict(
                            variant=name,
                            family=family,
                            policy=policy,
                            budget_scope="global_fixed_sites",
                            requested_budget=k,
                            recommendations=len(selected),
                            matched=len(match),
                            inventory=len(target),
                            recall=len(match) / len(target),
                            represented_routes=len(counts),
                            maximum_route_share=float(counts.max() / len(selected))
                            if len(selected)
                            else None,
                            routes_with_no_recommendation=int(
                                frame.route_id.nunique() - len(counts)
                            ),
                        )
                    )
                    for route, group in frame.groupby("route_id", dropna=False):
                        selected_count = int(selected.route_id.eq(route).sum())
                        concentrations.append(
                            dict(
                                variant=name,
                                family=family,
                                policy=policy,
                                budget=k,
                                route=str(route),
                                available_sites=len(group),
                                selected_sites=selected_count,
                                neglected=selected_count == 0,
                            )
                        )
            for k in KS:
                selected = frame.loc[frame[f"query_rank_{family}"].le(k)].reset_index(drop=True)
                match = match_one_to_one(selected, tx, 20)
                ranks.append(
                    dict(
                        variant=name,
                        family=family,
                        policy="raw_rf",
                        budget_scope="per_query_before_clipping",
                        requested_budget=k,
                        recommendations=len(selected),
                        matched=len(match),
                        inventory=len(target),
                        recall=len(match) / len(target),
                    )
                )
        # One geometry-only duplicate burden statistic; proximity is not identity.
        counts = (
            np.asarray(
                [
                    len(v)
                    for v in __import__("scipy").spatial.cKDTree(xy(frame)).query_ball_point(tx, 20)
                ]
            )
            if len(frame)
            else np.zeros(len(target), int)
        )
        pd.DataFrame(
            dict(
                record_id=target.record_id,
                candidates_within20m=counts,
                excess_nearby_recommendations=np.maximum(counts - 1, 0),
            )
        ).to_csv(OUT / f"duplicate_burden_{name}.csv", index=False)
        # Whole-corridor exclusions are inherited from existing saved models.
        # Records/methods were historically exposed: NOT independent validation.
        for family in ["RF", "HGB"]:
            subset = frame.loc[frame[f"score_{family}_roadheldout"].notna()].copy()
            recovered = gpd.read_parquet(
                ROOT
                / "reports/experiments/real_inventory_heldout_roads_20261015/recovered_road_segments.parquet"
            )
            (left, right), distances = recovered.sindex.nearest(
                target.geometry, return_all=False, return_distance=True
            )
            eligible = np.zeros(len(target), bool)
            eligible[left] = distances <= 100
            actual = target.loc[eligible].reset_index(drop=True)
            for policy in ["raw_rf", "dynamic_route_z"]:
                queue = ordered(
                    subset.assign(model_score=subset[f"score_{family}_roadheldout"]), policy=policy
                )
                for k in [*KS, len(queue)]:
                    match = match_one_to_one(queue.iloc[:k].reset_index(drop=True), xy(actual), 20)
                    holdouts.append(
                        dict(
                            variant=name,
                            family=family,
                            policy=policy,
                            requested_budget=k,
                            recommendations=min(k, len(queue)),
                            records=len(actual),
                            matched=len(match),
                            independently_unexposed=False,
                            whole_corridor_model_exclusion=True,
                        )
                    )
    pd.DataFrame(rows).to_csv(OUT / "candidate_and_site_coverage.csv", index=False)
    pd.DataFrame(ranks).to_csv(OUT / "fixed_budget_retrieval.csv", index=False)
    pd.DataFrame(routes).to_csv(OUT / "region_route_performance.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "variant_record_distances.csv", index=False)
    pd.DataFrame(concentrations).to_csv(OUT / "route_budget_concentration.csv", index=False)
    pd.DataFrame(holdouts).drop_duplicates().to_csv(
        OUT / "existing_complete_road_holdout_comparison.csv", index=False
    )
    return rows


def distribution(tasks, frames):
    """Descriptive shifts only; never select thresholds or recalibrate models."""
    folder = ROOT / "reports/experiments/hydrology_driven_sites_20261015"
    measured = gpd.read_parquet(folder / "physical_measurements_v2.parquet")
    context = measured.loc[measured.proposal_kind.eq("internal_road_probe")]
    rows = []
    for family, manifest_path in [
        ("RF", ROOT / "models/flow_c_frozen_v1.joblib.training_records.csv"),
        ("HGB", ROOT / "reports/experiments/site_challenger_20261015/final_training_rows.csv"),
    ]:
        fitted = load_model(family)
        transformed = apply_feature_contract(
            measured, fitted["contract"], normalization_context=context
        ).set_index("candidate_id")
        manifest = pd.read_csv(manifest_path)
        train = transformed.loc[manifest.candidate_id]
        candidate_parts = []
        for ix, iy, _, _ in tasks:
            path = OUT / "cells" / f"{ix}_{iy}" / "challenger" / f"features_{family}.parquet"
            if path.exists():
                frame = gpd.read_parquet(path)
                candidate_parts.append(own(frame, ix, iy))
        candidates = pd.concat(candidate_parts).set_index("candidate_id")
        for variant in VARIANTS:
            selected = candidates.loc[frames[variant].candidate_id]
            for col in fitted["feature_columns"]:
                t = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                v = pd.to_numeric(selected[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                lo, hi = t.quantile([0.01, 0.99])
                rows.append(
                    dict(
                        family=family,
                        variant=variant,
                        feature=col,
                        train_rows=len(t),
                        candidate_rows=len(v),
                        train_missing_fraction=float(t.isna().mean()),
                        candidate_missing_fraction=float(v.isna().mean()),
                        train_p01=lo,
                        train_p99=hi,
                        candidate_median=v.median(),
                        outside_training_p01_p99_fraction=float(
                            ((v < lo) | (v > hi)).sum() / v.notna().sum()
                        )
                        if v.notna().any()
                        else None,
                        exact_saved_transform=True,
                    )
                )
    pd.DataFrame(rows).to_csv(OUT / "feature_compatibility_distribution.csv", index=False)


def main():
    assert (OUT / "DIAGNOSIS_COMPLETE.json").exists(), "Diagnose baseline first"
    assert (OUT / "FIXED_METHOD.md").exists(), "Fix challenger parameters before evaluating"
    assert not (OUT / "COMPUTATIONS_COMPLETE.json").exists()
    before = protected()
    hashes = json.loads((OUT / "baseline_input_hashes.json").read_text())
    extra = [
        ROOT / "staging/discovery_sites.py",
        ROOT / "staging/discovery_service.py",
        Path(__file__),
        OUT / "FIXED_METHOD.md",
    ]
    extra += list(
        (ROOT / "reports/experiments/hydrology_driven_sites_20261015").glob("*experimental.joblib")
    )
    extra += list(
        (ROOT / "reports/experiments/hydrology_driven_sites_20261015").glob("*fold*.joblib")
    )
    guard = {str(p): sha256_file(p) for p in extra}
    (OUT / "experiment_input_hashes.json").write_text(json.dumps(guard, indent=2) + "\n")
    tasks = json.loads((PRIOR / "regional_fixed_method.json").read_text())["tasks"]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i, result in enumerate(as_completed([pool.submit(cell, task) for task in tasks]), 1):
            print(i, len(tasks), result.result(), flush=True)
    frames = pools(tasks)
    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
    rows = summarize(frames, truth)
    distribution(tasks, frames)
    assert protected() == before
    assert all(sha256_file(Path(p)) == digest for p, digest in (hashes | guard).items())
    (OUT / "COMPUTATIONS_COMPLETE.json").write_text(
        json.dumps(
            dict(
                cells=len(tasks),
                fits=0,
                production_changed=False,
                frozen_models_unchanged=True,
                baseline_inputs_unchanged=True,
                metric_rows=len(rows),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
