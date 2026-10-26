"""Fixed feature-location policy + saved held-out RF predictions;zero fits.

Outputs are immutable new research directories. Production and historical
artifacts are SHA guarded. No policy is selected from resulting inventory matches.
"""

# ruff:noqa:E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from staging.feature_sites import VERSION, EVIDENCE_COLUMNS, proposals, ranked, legacy_terrain_at
from staging.hybrid_sites import cluster_sites
from scripts.validate_density_controls import (
    HYBRID,
    PREVIOUS,
    PRIOR,
    inventory,
    xy,
    adjacency,
    prefix_matches,
    conditional_order,
    stream_seed,
    summary,
    coverage_record,
    uniform,
    provenance,
)
from scripts.evaluate_hydrology_sites import protected, save_json
from scripts.validate_inventory_heldout_roads import match_one_to_one

OUT = ROOT / "reports/experiments/feature_localization_20261015"
KS = (10, 25, 50, 100, 250)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    before = protected()
    sources = [
        p
        for folder in ["staging", "web", "api", "models"]
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in str(p)
        and not str(p.relative_to(ROOT)).startswith("staging/feature")
    ]
    sources += list(HYBRID.glob("*.parquet")) + list(PREVIOUS.glob("scores_RF_fold*.parquet"))
    sources += [
        PREVIOUS / "physical_measurements_v2.parquet",
        PRIOR / "recovered_road_segments.parquet",
        PRIOR / "inventory_split_manifest.csv",
        HYBRID / "feature_parity_48.csv",
    ]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(sources))}
    save_json(out / "input_hashes.json", hashes)
    (out / "proposal_code_as_run.txt").write_bytes((ROOT / "staging/feature_sites.py").read_bytes())
    save_json(
        out / "fixed_method.json",
        dict(
            version=VERSION,
            fits=0,
            prediction_calls=0,
            independent=False,
            policy="Admit strict mapped/DEMcrossing OR local road-profile depression max>=0.15m and existing low-point score>0 OR local upstream-area max>0 with negativeTPI9px>0.",
            windows_m=60,
            extrema="same road/part,full two-sided60m,finite values,lower values on both sides;plateau ties byID",
            ml_peak="compatible savedRF peaks computed as diagnostic only;not sufficient alone for admission",
            parameter_origin="existing20–60m profile definition and prior0.15m floor;zero-valued sign tests;no inventory/outerfold threshold selection",
            consolidation_m=20,
            rank_policies=["raw_rf", "dynamic_route_z"],
            random_per_comparison=1000,
            seed=20261015,
            controls=["county", "route", "block2km"],
            density_budgets=[100, 250, 500, 1000, 2000, 5000],
            subsets=100,
            denominators=[705, 714],
            warning="Recommendations require field verification.",
        ),
    )
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    roads["id"] = roads.segment_id.astype(str)
    truth = inventory()
    real = truth.loc[truth.fold.ge(0)].copy().reset_index(drop=True)
    provenance(out, roads, real)
    km = float(roads.length.sum() / 1000)
    points = gpd.read_parquet(HYBRID / "all_raw_proposals.parquet")
    measured = gpd.read_parquet(PREVIOUS / "physical_measurements_v2.parquet")
    measured = gpd.GeoDataFrame(
        pd.concat(
            [measured, gpd.read_parquet(HYBRID / "new_mapped_measurements.parquet")],
            ignore_index=True,
        ),
        crs=32618,
    ).set_index("candidate_id")
    for column in EVIDENCE_COLUMNS:
        points[column] = measured.loc[points.candidate_id, column].to_numpy()
    cache = pd.concat(
        [gpd.read_parquet(PREVIOUS / f"scores_RF_fold{f}.parquet") for f in range(3)],
        ignore_index=True,
    )
    cache = cache.set_index("candidate_id")
    points["model_score"] = cache.model_score.reindex(points.candidate_id).to_numpy()
    assert np.isfinite(
        points.loc[
            points.proposal_kind.isin(["internal_road_probe", "mapped_crossing", "dem_crossing"]),
            "model_score",
        ]
    ).all()
    marked = proposals(points)
    selected = marked.loc[marked.evidence_admitted].copy().reset_index(drop=True)
    selected.to_parquet(out / "feature_proposals.parquet", index=False)
    marked[
        [
            "candidate_id",
            "road_id",
            "route_id",
            "road_part_index",
            "road_chainage_m",
            "proposal_kind",
            "evidence_admitted",
            "profile_local_peak",
            "flow_valley_local_peak",
            "ml_local_peak",
            "physical_crossing",
            "geometry",
        ]
    ].to_parquet(out / "admission_audit.parquet", index=False)
    physical = cluster_sites(selected)
    physical.to_parquet(out / "scorefree_sites.parquet", index=False)
    coverage = []
    routes = []
    for method, frame in [
        ("all_prior_candidates", points),
        ("feature_candidates", selected),
        ("feature_scorefree_sites", physical),
    ]:
        for scope, t in [("eligible705", real), ("all714", truth)]:
            coverage.append(coverage_record(method, frame, t, km, scope=scope))
        for route in sorted(roads.corridor.unique()):
            t = real.loc[real.corridor.eq(route)]
            length = float(roads.loc[roads.corridor.eq(route)].length.sum() / 1000)
            routes.append(
                coverage_record(method, frame.loc[frame.route_id.eq(route)], t, length, route=route)
            )
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(routes).to_csv(out / "coverage_by_route.csv", index=False)
    components = []
    for source in [
        "physical_crossing",
        "profile_local_peak",
        "flow_valley_local_peak",
        "ml_local_peak",
    ]:
        part = marked.loc[marked[source]]
        row = coverage_record(source, part, real, km)
        row["admitted_alone"] = source != "ml_local_peak"
        components.append(row)
    pd.DataFrame(components).to_csv(out / "source_diagnostics.csv", index=False)
    uniform10 = gpd.read_parquet(
        ROOT / "reports/experiments/density_controls_20261015/uniform_10m.parquet"
    )
    uniform_sites = gpd.read_parquet(
        ROOT / "reports/experiments/density_controls_20261015/scorefree_sites_uniform10.parquet"
    )
    density = []
    for stage, frame, control in [
        ("candidate", selected, uniform10),
        ("physical_site", physical, uniform_sites),
    ]:
        budgets = sorted(
            set([b for b in (100, 250, 500, 1000, 2000, 5000) if b <= len(frame)] + [len(frame)])
        )
        fe, ce = adjacency(frame, real), adjacency(control, real)
        draws = {b: [] for b in budgets}
        control_draws = {b: [] for b in budgets}
        for rep in range(100):
            a = prefix_matches(
                fe, stream_seed("feature-density", stage, rep).permutation(len(frame)), budgets
            )
            b = prefix_matches(
                ce, stream_seed("uniform-density", stage, rep).permutation(len(control)), budgets
            )
            for k, x, y in zip(budgets, a, b):
                draws[k].append(int(x))
                control_draws[k].append(int(y))
        for budget in budgets:
            exact = uniform(roads, budget=budget)
            exactmatch = len(match_one_to_one(exact, xy(real)))
            arr = np.asarray(draws[budget])
            null = np.asarray(control_draws[budget])
            density.append(
                dict(
                    stage=stage,
                    budget=budget,
                    density_per_km=budget / km,
                    feature_mean=float(arr.mean()),
                    feature_p025=float(np.quantile(arr, 0.025)),
                    feature_p975=float(np.quantile(arr, 0.975)),
                    uniform_subset_mean=float(null.mean()),
                    uniform_subset_p025=float(np.quantile(null, 0.025)),
                    uniform_subset_p975=float(np.quantile(null, 0.975)),
                    exact_uniform_matches=exactmatch,
                    excess_over_exact=float(arr.mean() - exactmatch),
                )
            )
    pd.DataFrame(density).to_csv(out / "density_matched.csv", index=False)
    rows = []
    draw_rows = []
    route_ranking = []
    cost = []
    for fold in range(3):
        held = set(roads.loc[roads.fold.eq(fold), "corridor"])
        test = selected.loc[selected.route_id.isin(held)].copy().reset_index(drop=True)
        t = real.loc[real.fold.eq(fold)].sort_values("record_key").reset_index(drop=True)
        for policy in ["raw_rf", "dynamic_route_z"]:
            sites = ranked(test, policy)
            sites.to_parquet(out / f"sites_{policy}_fold{fold}.parquet", index=False)
            sites = sites.sort_values("global_rank").reset_index(drop=True)
            edges = adjacency(sites, t)
            order = np.arange(len(sites))
            obs = prefix_matches(edges, order, KS)
            labels = sites.route_id.astype(str).to_numpy()
            blocks = (
                sites.route_id.astype(str)
                + "|"
                + sites.road_id.astype(str)
                + "|"
                + sites.road_part_index.astype(str)
                + "|"
                + (sites.road_chainage_m // 2000).astype(int).astype(str)
            ).to_numpy()
            cost.append(
                dict(
                    fold=fold,
                    policy=policy,
                    sites=len(sites),
                    full_matches=len(match_one_to_one(sites, xy(t))),
                )
            )
            for control in ["county", "route", "block2km"]:
                values = np.zeros((1000, len(KS)), int)
                for rep in range(1000):
                    rng = stream_seed("feature-ranking", fold, policy, control, rep)
                    perm = (
                        rng.permutation(len(sites))
                        if control == "county"
                        else conditional_order(order, labels if control == "route" else blocks, rng)
                    )
                    values[rep] = prefix_matches(edges, perm, KS)
                    draw_rows.extend(
                        dict(
                            fold=fold,
                            policy=policy,
                            control=control,
                            replicate=rep,
                            k=k,
                            matched=int(v),
                        )
                        for k, v in zip(KS, values[rep])
                    )
                for c, k in enumerate(KS):
                    assert obs[c] == len(match_one_to_one(sites.head(k), xy(t)))
                    top = sites.head(k)
                    allocation = top.route_id.value_counts()
                    rows.append(
                        dict(
                            fold=fold,
                            policy=policy,
                            control=control,
                            k=k,
                            actual_slots=len(top),
                            inventory=len(t),
                            site_recall=obs[c] / len(t),
                            inventory_match_fraction=obs[c] / len(top) if len(top) else None,
                            largest_route=str(allocation.index[0]) if len(top) else None,
                            largest_share=float(allocation.iloc[0] / len(top))
                            if len(top)
                            else None,
                            **summary(values[:, c], obs[c]),
                        )
                    )
            for route in sorted(held):
                own = sites.loc[sites.route_id.eq(route)]
                owntruth = t.loc[t.corridor.eq(route)]
                for k in KS:
                    top = sites.head(k)
                    top = top.loc[top.route_id.eq(route)]
                    pairs = match_one_to_one(top, xy(owntruth))
                    route_ranking.append(
                        dict(
                            fold=fold,
                            policy=policy,
                            route=route,
                            k=k,
                            inventory=len(owntruth),
                            population_sites=len(own),
                            slots=len(top),
                            matched=len(pairs),
                            localization_mean_m=float(pairs.distance_m.mean())
                            if len(pairs)
                            else None,
                        )
                    )
            print(f"Feature fixed ranking fold{fold} {policy}:{obs.tolist()}", flush=True)
    pd.DataFrame(rows).to_csv(out / "ranking_by_fold.csv", index=False)
    pd.DataFrame(cost).to_csv(out / "scoreordered_coverage.csv", index=False)
    pd.DataFrame(route_ranking).to_csv(out / "ranking_by_route.csv", index=False)
    draws = pd.DataFrame(draw_rows)
    draws.to_parquet(out / "random_replicates.parquet", index=False)
    combined = []
    for (policy, control, k), group in draws.groupby(["policy", "control", "k"]):
        arr = group.groupby("replicate").matched.sum().to_numpy()
        part = pd.DataFrame(rows)
        part = part.loc[part.policy.eq(policy) & part.control.eq(control) & part.k.eq(k)]
        slots = int(part.actual_slots.sum())
        observed = int(part.observed.sum())
        combined.append(
            dict(
                method="feature",
                policy=policy,
                control=control,
                k=k,
                actual_slots=slots,
                inventory_match_fraction=observed / slots if slots else None,
                site_recall=observed / 705,
                **summary(arr, observed),
            )
        )
    combined = pd.DataFrame(combined)
    combined.to_csv(out / "ranking_combined.csv", index=False)
    prior = pd.read_csv(
        ROOT / "reports/experiments/density_controls_20261015/ranking_random_combined.csv"
    )
    prior = prior.loc[prior.k.isin(KS)]
    pd.concat([prior, combined], ignore_index=True).to_csv(
        out / "ranking_comparison.csv", index=False
    )
    # Complete48-field status and fresh original15-field semantic recovery.
    parity = pd.read_csv(HYBRID / "feature_parity_48.csv")
    parity.to_csv(out / "feature_parity_48.csv", index=False)
    original = gpd.read_parquet(PRIOR / "historical10_features.parquet").set_index("candidate_id")
    ids = sorted(set(original.index) & set(measured.index))
    legacy = legacy_terrain_at(original.loc[ids], ROOT / "data/raw/dem.tif")
    replay = []
    for col in legacy.columns.drop("geometry"):
        equal = np.isclose(
            original.loc[ids, col], legacy[col], rtol=1e-6, atol=1e-6, equal_nan=True
        )
        replay.append(
            dict(
                feature=col,
                rows=len(ids),
                equal_values=int(equal.sum()),
                full_parity=bool(equal.all()),
            )
        )
    pd.DataFrame(replay).to_csv(out / "legacy_terrain_recovery.csv", index=False)
    assert all(row["full_parity"] for row in replay)
    save_json(
        out / "compatibility.json",
        dict(
            frozen_rf_used=False,
            v2_saved_scores_used=True,
            ordered_fields=48,
            semantic_mismatches=int(parity.differing_values.gt(0).sum()),
            original_terrain_fields_recovered=len(replay),
            same_coordinate_rows=len(ids),
            unresolved=[
                "10m versus1arcsecond DEM footprints",
                "profile missingness/tie ranks",
                "batch composites/density context",
                "crossing associations",
                "raw edge-path values versus binary quality flag",
            ],
            preprocessing=json.loads(
                (ROOT / "models/flow_c_frozen_v1.joblib.schema.json").read_text()
            )["preprocessing"],
        ),
    )
    assert before == protected()
    assert all(sha256_file(ROOT / p) == v for p, v in hashes.items())
    save_json(
        out / "COMPUTATIONS_COMPLETE.json",
        dict(
            version=VERSION,
            fits=0,
            prediction_calls=0,
            protected_unchanged=291,
            existing_files_unchanged=len(hashes),
            independent=False,
            production_changed=False,
        ),
    )
    print(
        "Feature localization evaluation complete;no model fits,predict calls or protected changes",
        flush=True,
    )


if __name__ == "__main__":
    main()
