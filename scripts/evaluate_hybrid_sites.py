"""Fixed high-recall generator comparison. Zero model fits; staging outputs only."""

# ruff: noqa: E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.features import add_route_profile_dem_features
from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hydrology_sites import PRIOR, HYDRO, protected, save_json, score
from scripts.validate_inventory_heldout_roads import coverage, evaluate, match_one_to_one
from staging.hydrology_sites import (
    THRESHOLDS,
    apply_feature_contract,
    detect_events,
    physical_measurements,
)
from staging.hybrid_sites import (
    VERSION,
    attach_road_topology,
    classify_evidence,
    cluster_sites,
    historical_candidates,
    overlapping_proposals,
    rank_sites,
)

PREVIOUS = ROOT / "reports/experiments/hydrology_driven_sites_20261015"
DEFAULT_OUT = ROOT / "reports/experiments/high_recall_hybrid_20261015"


def inventory():
    rows = pd.read_csv(PRIOR / "inventory_split_manifest.csv")
    return (
        gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy(rows.source_x, rows.source_y), crs=26918)
        .to_crs(32618)
        .sort_values("record_key")
        .reset_index(drop=True)
    )


def coverage_row(method, phase, candidates, sites, truth, km, scope, route="all"):
    xy = np.c_[truth.geometry.x, truth.geometry.y]
    raw = len(match_one_to_one(candidates, xy))
    full = len(match_one_to_one(sites, xy))
    return dict(
        method=method,
        phase=phase,
        scope=scope,
        route=route,
        official_culverts=len(truth),
        candidates=len(candidates),
        physical_sites=len(sites),
        candidate_covered_any=int((coverage(candidates, truth) <= 20).sum()),
        candidate_matches_one_to_one=raw,
        consolidated_covered_any=int((coverage(sites, truth) <= 20).sum()),
        consolidated_matches_one_to_one=full,
        matches_lost_consolidation=raw - full,
        road_km=km,
        sites_per_road_km=len(sites) / km if km else None,
    )


def prepare(out):
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    roads["id"], roads["route_id"] = roads.segment_id, roads.corridor
    roads = roads.sort_values("id").reset_index(drop=True)
    streams = gpd.read_file(ROOT / "data/raw/streams.gpkg").to_crs(32618)
    print(
        "Regenerating historical label-free10m+mapped recipe on121 identical road segments",
        flush=True,
    )
    historical, probes, mapped = historical_candidates(roads, streams)
    old_probes = gpd.read_parquet(PREVIOUS / "dense_search_probes.parquet").set_index(
        "candidate_id"
    )
    assert set(probes.candidate_id) == set(old_probes.index)
    assert (
        np.max(
            probes.geometry.distance(
                old_probes.loc[probes.candidate_id].reset_index().geometry, align=False
            )
        )
        < 1e-6
    )
    networks = {key: gpd.read_parquet(HYDRO / f"drainage_{key}.parquet") for key in THRESHOLDS}
    events, rejected = detect_events(roads, streams, networks)
    events = attach_road_topology(events, roads)
    old_events = gpd.read_parquet(PREVIOUS / "crossing_events.parquet")
    assert set(events.candidate_id) == set(old_events.candidate_id)
    rejected.to_csv(out / "rejected_contacts.csv", index=False)
    for name, table in [
        ("historical", historical),
        ("probes", probes),
        ("mapped", mapped),
        ("crossing", events),
    ]:
        table.to_parquet(out / f"{name}_proposals.parquet", index=False)
    # Existing measurements are shared, not regenerated/relabelled. Only genuinely
    # new mapped proposal geometry requires fresh GIS extraction.
    cached = gpd.read_parquet(PREVIOUS / "physical_measurements_v2.parquet")
    new = mapped.loc[~mapped.candidate_id.isin(cached.candidate_id)].copy()
    if len(new):
        new["source"] = (
            "route_interval_sample"  # Same v2 builder contract; no source indicator is a model input.
        )
        all_roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618)
        print(
            f"Measuring {len(new)} new mapped proposals; reusing existing dense/event measurements",
            flush=True,
        )
        measured_new = physical_measurements(new, all_roads, streams, HYDRO)
        measured_new.to_parquet(out / "new_mapped_measurements.parquet", index=False)
        cached = gpd.GeoDataFrame(pd.concat([cached, measured_new], ignore_index=True), crs=32618)
    raw = gpd.GeoDataFrame(pd.concat([probes, mapped, events], ignore_index=True), crs=32618)
    assert raw.candidate_id.is_unique
    feature_columns = [
        "distance_to_nearest_stream_m",
        "route_elevation_dip_60m",
        "route_low_point_score",
        "road_profile_available",
        "flow_accumulation_cells",
        "upstream_area_m2",
        "slope_degrees",
        "hydrology_edge_risk",
    ]
    index = cached.set_index("candidate_id")
    for column in feature_columns:
        raw[column] = index.loc[raw.candidate_id, column].to_numpy()
    raw = classify_evidence(raw)
    hybrid = overlapping_proposals(raw)
    historical = (
        raw.set_index("candidate_id", drop=False)
        .loc[historical.candidate_id]
        .reset_index(drop=True)
    )
    crossing = raw.loc[raw.candidate_id.isin(events.candidate_id)].reset_index(drop=True)
    cohorts = {"historical": historical, "crossing_only": crossing, "hybrid": hybrid}
    raw.to_parquet(out / "all_raw_proposals.parquet", index=False)
    hybrid.to_parquet(out / "hybrid_proposals.parquet", index=False)
    return roads, cached, raw, cohorts


def feature_audit(out, cached):
    fitted = joblib.load(PREVIOUS / "hydrology-sites-v2_RF_fold0.joblib")
    context = cached.loc[cached.proposal_kind.eq("internal_road_probe")]
    repaired = apply_feature_contract(
        cached, fitted["contract"], normalization_context=context
    ).set_index("candidate_id")
    original = gpd.read_parquet(PRIOR / "historical10_features.parquet").set_index("candidate_id")
    common = sorted(set(original.index) & set(repaired.index))
    assert len(common) > 100
    assert (
        original.loc[common].geometry.distance(repaired.loc[common].geometry, align=False).max()
        < 1e-8
    )
    rows = []
    for column in fitted["feature_columns"]:
        a, b = (
            original.loc[common, column].to_numpy(float),
            repaired.loc[common, column].to_numpy(float),
        )
        finite = np.isfinite(a) & np.isfinite(b)
        equal = np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
        delta = np.abs(a[finite] - b[finite])
        rows.append(
            dict(
                feature=column,
                paired_rows=len(common),
                equal_values=int(equal.sum()),
                differing_values=int((~equal).sum()),
                missing_v1=int((~np.isfinite(a)).sum()),
                missing_v2=int((~np.isfinite(b)).sum()),
                mean_absolute_difference=float(delta.mean()) if len(delta) else None,
                max_absolute_difference=float(delta.max()) if len(delta) else None,
            )
        )
    pd.DataFrame(rows).to_csv(out / "feature_parity_48.csv", index=False)
    probes = cached.loc[cached.proposal_kind.eq("internal_road_probe")].reset_index(drop=True)
    dense_old = add_route_profile_dem_features(probes)
    profile = []
    for column in [
        "route_elevation_dip_60m",
        "route_elevation_dip_rank_pct",
        "route_low_point_score",
    ]:
        a, b = dense_old[column].to_numpy(float), probes[column].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        profile.append(
            dict(
                feature=column,
                rows=len(probes),
                finite_pairs=int(both.sum()),
                different_finite_values=int(
                    (~np.isclose(a[both], b[both], rtol=1e-6, atol=1e-6)).sum()
                ),
                unavailable_v2=int((~np.isfinite(b)).sum()),
                valid_zero_v2=int((b == 0).sum()),
                nonzero_v2=int((np.isfinite(b) & (b != 0)).sum()),
            )
        )
    pd.DataFrame(profile).to_csv(out / "road_profile_parity.csv", index=False)
    save_json(
        out / "feature_compatibility.json",
        dict(
            frozen_rf_compatible=False,
            original_frozen_model_scored=False,
            columns=48,
            compared_rows=len(common),
            different_fields=[r["feature"] for r in rows if r["differing_values"]],
            reasons=[
                "v2 profiles use complete independent10m probes, unavailable shoulders stayNaN instead of0",
                "v2 positive-dip empirical ranks use upperCDF rather than pandas average-tie ranks",
                "DEM composite references are training-only rather than proposal-batch quantiles",
                "real DEM edge flags replace legacy default0",
                "road density normalization uses unlabeled dense context and training fallback",
            ],
            evaluation="Reuse already fitted fixed800tree v2 RF copies and original held-out manifests; no new fit, tuning or original model mutation",
        ),
    )


def reconciliation(out, roads, cohorts):
    saved = gpd.read_file(ROOT / "data/interim/actual_ulster_candidates_with_route_samples.gpkg")
    selected = saved.loc[saved.road_id.astype(str).isin(roads.id.astype(str))].copy()
    fresh = cohorts["historical"]

    def distances(a, b):
        return cKDTree(np.c_[b.geometry.x, b.geometry.y]).query(np.c_[a.geometry.x, a.geometry.y])[
            0
        ]

    a, b = distances(fresh, selected), distances(selected, fresh)
    score_ids = pd.read_parquet(
        ROOT
        / "reports/experiments/nysdot_ulster_ranking_v14_20261006/oof_candidate_scores.parquet",
        columns=["candidate_id"],
    )
    pool = saved.loc[saved.candidate_id.isin(score_ids.candidate_id)].copy()
    truth = inventory()
    notes = dict(
        saved_all_generated=len(saved),
        saved_selected_road_ids=len(selected),
        historical_regenerated_same_roads=len(fresh),
        saved_score_pool=len(score_ids),
        saved_score_pool_coordinates_recovered=len(pool),
        saved_score_pool_native_covered=int((coverage(pool, truth) <= 20).sum()),
        fresh_within1mm_of_saved=int((a <= 0.001).sum()),
        saved_within1mm_of_fresh=int((b <= 0.001).sum()),
        selection="121 whole TIGER segments recovered from historical interval metadata,22 geometric corridor groups; no inventory proximity determines selection",
        saved_difference="Full legacy mapped generation globally deduplicated across all county roads before merging121 sampled-road portions; restricted121-road rerun can retain contacts lost to other roads. Concurrent road IDs also preserve more probes than historical merge. Do not claim exact archived51417 pool on arbitrary new roads.",
        recipe="generate_candidates(snap20m,min_spacing20m)+generate_road_route_candidates(interval10m,offset0)+merge_candidate_layers(min_spacing5m); original optional field-report route selection replaced by all supplied roads, never by official coordinates",
        historical_coordinate_convention="Legacy benchmark used26918 x/y after converting candidate geometry; this experiment transforms official26918 inventory to32618 and preserves native candidate32618. No reinterpretation of coordinates.",
    )
    save_json(out / "generator_reconciliation.json", notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    before = protected()
    save_json(out / "protected_before.json", before)
    save_json(
        out / "preregistered_method.json",
        dict(
            version=VERSION,
            fits=0,
            cohorts=["historical", "crossing_only", "hybrid"],
            interval_m=10,
            overlap_radius_m=1,
            site_radius_m=20,
            clustering="same road and part,Euclidean AND chainage<=radius,nontransitive anchor groups,separate known channels; fixed priority/raw/ID representative",
            policies=["raw_rf", "dynamic_route_z"],
            ranking="Representative raw800treeRFv2 or default dynamicROUTEZ, no evidence boosts, no route quotas, no outcomes choose parameters",
            model_reuse="Three existing hydrology-sites-v2_RF_fold{0,1,2}.joblib copies; no new model fits",
            evaluation="same121 full road segments,22 corridor groups,prior3 fixedSHA256 folds and100m road purges",
            matching_m=20,
            denominators=[714, 705],
            independent=False,
            warning="Recommendations require field verification.",
        ),
    )
    roads, cached, raw, cohorts = prepare(out)
    reconciliation(out, roads, cohorts)
    feature_audit(out, cached)
    truth = inventory()
    eligible = truth.loc[truth.fold.ge(0)].copy()
    km = float(roads.geometry.length.sum() / 1000)
    geometry_rows, route_rows = [], []
    for method, candidates in cohorts.items():
        sites = cluster_sites(candidates)
        sites.to_parquet(out / f"geometry_sites_{method}.parquet", index=False)
        for scope, real in [("full714", truth), ("eligible705", eligible)]:
            geometry_rows.append(
                coverage_row(method, "score_free20m", candidates, sites, real, km, scope)
            )
        for route in sorted(roads.corridor.unique()):
            selected = candidates.loc[candidates.route_id.eq(route)]
            selected_sites = sites.loc[sites.route_id.eq(route)]
            real = eligible.loc[eligible.corridor.eq(route)]
            length = float(roads.loc[roads.corridor.eq(route)].geometry.length.sum() / 1000)
            route_rows.append(
                coverage_row(
                    method,
                    "score_free20m",
                    selected,
                    selected_sites,
                    real,
                    length,
                    "eligible705",
                    route,
                )
            )
    pd.DataFrame(geometry_rows).to_csv(out / "candidate_coverage.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "coverage_by_route.csv", index=False)
    contributions = []
    for column in ["proposal_kind", "representative_evidence_type"]:
        for source, group in raw.groupby(column, sort=True):
            covered = coverage(group, eligible) <= 20
            other = raw.loc[~raw.candidate_id.isin(group.candidate_id)]
            unique = covered & (coverage(other, eligible) > 20)
            contributions.append(
                dict(
                    grouping=column,
                    source=source,
                    proposals=len(group),
                    covered_any=int(covered.sum()),
                    unique_inventory_contribution=int(unique.sum()),
                    matches_one_to_one=len(
                        match_one_to_one(group, np.c_[eligible.geometry.x, eligible.geometry.y])
                    ),
                )
            )
    pd.DataFrame(contributions).to_csv(out / "candidate_source_contributions.csv", index=False)
    context = cached.loc[cached.proposal_kind.eq("internal_road_probe")]
    metrics, per_route, scored_coverage, retention = [], [], [], []
    input_models = {}
    for fold in range(3):
        model_path = PREVIOUS / f"hydrology-sites-v2_RF_fold{fold}.joblib"
        fitted = joblib.load(model_path)
        assert (
            fitted["experimental"]
            and len(fitted["feature_columns"]) == 48
            and fitted["model"].n_estimators == 800
        )
        input_models[str(model_path.relative_to(ROOT))] = sha256_file(model_path)
        transformed = apply_feature_contract(
            cached, fitted["contract"], normalization_context=context
        ).set_index("candidate_id")
        held = roads.loc[roads.fold.eq(fold), "corridor"].tolist()
        real = eligible.loc[eligible.fold.eq(fold)].reset_index(drop=True)
        test = raw.loc[raw.route_id.isin(held)].reset_index(drop=True)
        values = transformed.loc[test.candidate_id]
        test["model_score"] = score(fitted, values, fitted["feature_columns"])
        # Verify exact repeated scoring against previous run, not only same model name.
        prior_scores = pd.read_parquet(PREVIOUS / f"scores_RF_fold{fold}.parquet").set_index(
            "candidate_id"
        )
        common = test.loc[test.candidate_id.isin(prior_scores.index)]
        assert np.allclose(
            common.model_score,
            prior_scores.loc[common.candidate_id].model_score,
            rtol=0,
            atol=1e-12,
        )
        for method, candidates in cohorts.items():
            selected = candidates.loc[candidates.route_id.isin(held)].copy().reset_index(drop=True)
            selected["model_score"] = (
                test.set_index("candidate_id").loc[selected.candidate_id, "model_score"].to_numpy()
            )
            for policy in ("raw_rf", "dynamic_route_z"):
                sites = rank_sites(selected, policy)
                ctx = dict(cohort=method, fold=fold, model="RF_v2_reused", policy=policy)
                metrics.extend(
                    evaluate(sites.rename(columns={"route_id": "corridor"}), selected, real, ctx)
                )
                sites.to_parquet(out / f"sites_{method}_{policy}_fold{fold}.parquet", index=False)
                fold_km = float(roads.loc[roads.fold.eq(fold)].geometry.length.sum() / 1000)
                row = coverage_row(method, policy, selected, sites, real, fold_km, "heldout_group")
                row["fold"] = fold
                scored_coverage.append(row)
                for route in sorted(set(held)):
                    heads = sites.loc[sites.route_id.eq(route)]
                    per_route.extend(
                        evaluate(
                            heads.rename(columns={"route_id": "corridor"}),
                            selected.loc[selected.route_id.eq(route)],
                            real.loc[real.corridor.eq(route)],
                            dict(**ctx, route=route),
                        )
                    )
                    for k in (25, 50, 100, 250):
                        top = sites.head(k)
                        retention.append(
                            dict(
                                **ctx,
                                route=route,
                                k=k,
                                sites_in_global_budget=int(top.route_id.eq(route).sum()),
                                eligible_inventory=int(real.corridor.eq(route).sum()),
                            )
                        )
        print(f"Completed fixed RF fold{fold}; no model fit", flush=True)
    pd.DataFrame(metrics).to_csv(out / "ranking_metrics.csv", index=False)
    pd.DataFrame(per_route).to_csv(out / "ranking_by_route.csv", index=False)
    pd.DataFrame(retention).to_csv(out / "global_budget_route_allocation.csv", index=False)
    pd.DataFrame(scored_coverage).to_csv(out / "score_ordered_coverage.csv", index=False)
    save_json(out / "reused_model_hashes.json", input_models)
    assert before == protected()
    save_json(
        out / "COMPLETE.json",
        dict(
            version=VERSION,
            fits=0,
            production_changed=False,
            protected_unchanged=291,
            independent_test=False,
        ),
    )
    print("Hybrid evaluation complete; no production changes or model training", flush=True)


if __name__ == "__main__":
    main()
