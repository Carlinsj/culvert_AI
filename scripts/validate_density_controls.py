"""Retrospective density and ranking-null audit. No fitting, scoring, or UI writes.

Run: scripts/python.sh scripts/validate_density_controls.py
Outputs are new research artifacts; existing inputs are SHA256 guarded.
"""

# ruff: noqa: E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hybrid_sites import DEFAULT_OUT as HYBRID, PREVIOUS, inventory
from scripts.evaluate_hydrology_sites import PRIOR, protected, save_json
from staging.hydrology_sites import line_parts
from staging.hybrid_sites import cluster_sites

OUT = ROOT / "reports/experiments/density_controls_20261015"
KS = (10, 25, 50, 100, 250, 500)
INTERVALS = (10, 20, 40, 75, 150, 300)
TOLERANCES = (5, 10, 20, 25)
BUDGETS = (100, 250, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000)
SEED = 20261015
N_RANDOM = 1000
N_THIN = 100


def xy(frame):
    return np.c_[frame.geometry.x, frame.geometry.y]


def uniform(roads, spacing=None, budget=None):
    """Label-free half-interval samples; exact-budget global chainage alternative.

    No endpoints are added. A part with no half-interval sample gets its midpoint.
    Exact-budget samples use a half-offset grid on concatenated stable road parts.
    """
    if (spacing is None) == (budget is None):
        raise ValueError("Specify spacing OR budget")
    parts = [
        (r, j, line)
        for _, r in roads.sort_values("id").iterrows()
        for j, line in enumerate(line_parts(r.geometry))
        if line.length > 0
    ]
    rows, offset = [], 0.0
    step = sum(line.length for _, _, line in parts) / budget if budget else spacing
    for road, j, line in parts:
        if budget:
            first = max(0, int(np.ceil(offset / step - 0.5)))
            last = min(budget, int(np.ceil((offset + line.length) / step - 0.5)))
            positions = (np.arange(first, last) + 0.5) * step - offset
        else:
            positions = np.arange(step / 2, line.length, step)
            if not len(positions):
                positions = [line.length / 2]
        for position in positions:
            rows.append(
                dict(
                    candidate_id=f"uniform_{spacing or 'B' + str(budget)}_{road.id}_{j}_{position:.9f}",
                    road_id=str(road.id),
                    route_id=str(road.corridor),
                    proposal_kind="uniform_road_sample",
                    road_part_index=j,
                    road_chainage_m=float(position),
                    geometry=line.interpolate(float(position)),
                )
            )
        offset += line.length
    result = gpd.GeoDataFrame(rows, crs=32618)
    if budget:
        assert len(result) == budget
    return result


def adjacency(points, truth, tolerance=20):
    # Adjacency only is label-dependent; proposal generation and budgets are not.
    return cKDTree(xy(truth)).query_ball_point(xy(points), tolerance)


def cardinality(edges, selected, n_truth):
    chosen = [edges[i] for i in selected]
    rows, cols = [], []
    for i, links in enumerate(chosen):
        rows.extend([i] * len(links))
        cols.extend(sorted(links))
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(chosen), n_truth)
    )
    graph.sort_indices()
    return int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())


def prefix_matches(edges, order, ks=KS):
    """Incremental augmenting paths: exact maximum cardinality at every prefix.

    Unlike counting pre-labelled matching sites, resolves capacity conflicts anew
    for each selected prefix. Canonical sorted truth neighbours make it deterministic.
    """
    right = {}

    def augment(site, seen):
        for target in sorted(edges[site]):
            if target in seen:
                continue
            seen.add(target)
            if target not in right or augment(right[target], seen):
                right[target] = site
                return True
        return False

    results, total, checkpoints = [], 0, {min(k, len(order)) for k in ks}
    values = {0: 0}
    for position, site in enumerate(order[: max(checkpoints)], 1):
        total += bool(augment(int(site), set()))
        if position in checkpoints:
            values[position] = total
    for k in ks:
        results.append(values[min(k, len(order))])
    return np.asarray(results, dtype=int)


def conditional_order(order, strata, rng):
    """Shuffle site identities within strata, preserving each rank-slot stratum."""
    result = np.array(order, copy=True)
    labels = np.asarray(strata)[order]
    for label in sorted(set(labels)):
        positions = np.flatnonzero(labels == label)
        result[positions] = rng.permutation(result[positions])
    return result


def stream_seed(*parts):
    return np.random.default_rng(
        int.from_bytes(
            hashlib.sha256((str(SEED) + "|" + "|".join(map(str, parts))).encode()).digest()[:8],
            "little",
        )
    )


def summary(values, observed):
    mean = float(np.mean(values))
    return dict(
        random_mean=mean,
        random_median=float(np.median(values)),
        random_sd=float(np.std(values, ddof=1)),
        p05=float(np.quantile(values, 0.05)),
        p95=float(np.quantile(values, 0.95)),
        p025=float(np.quantile(values, 0.025)),
        p975=float(np.quantile(values, 0.975)),
        random_min=int(np.min(values)),
        random_max=int(np.max(values)),
        observed=int(observed),
        lift=observed / mean if mean else None,
        excess_matches=observed - mean,
        p_one_sided=(1 + int((values >= observed).sum())) / (len(values) + 1),
    )


def coverage_record(method, points, truth, km, tolerance=20, **context):
    edges = adjacency(points, truth, tolerance)
    covered = set(j for links in edges for j in links)
    distances = (
        cKDTree(xy(points)).query(xy(truth))[0] if len(points) else np.full(len(truth), np.inf)
    )
    n = cardinality(edges, range(len(points)), len(truth))
    finite = distances[np.isfinite(distances)]
    return dict(
        method=method,
        tolerance_m=tolerance,
        candidates=len(points),
        road_km=km,
        density_per_km=len(points) / km,
        inventory=len(truth),
        covered_any=len(covered),
        coverage_pct=100 * len(covered) / len(truth) if len(truth) else None,
        matches_one_to_one=n,
        matches_per_100=100 * n / len(points) if len(points) else None,
        matches_per_km=n / km if km else None,
        truth_nearest_median_m=float(np.median(finite)) if len(finite) else None,
        truth_nearest_p90_m=float(np.quantile(finite, 0.9)) if len(finite) else None,
        **context,
    )


def input_paths():
    paths = {PRIOR / "recovered_road_segments.parquet", PRIOR / "inventory_split_manifest.csv"}
    paths.update(HYBRID.glob("*.parquet"))
    paths.update(HYBRID.glob("*.json"))
    paths.update(PREVIOUS.glob("*RF_fold*.joblib"))
    paths.update(PREVIOUS.glob("scores_RF_fold*.parquet"))
    paths.update(PREVIOUS.glob("feature_contract_RF_fold*.json"))
    paths.update(PRIOR.glob("training_manifest_RF_fold*.csv"))
    for folder in ("staging", "web", "api", "src/culvert_ai"):
        paths.update(
            p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)
        )
    return paths


def provenance(out, roads, eligible):
    lineage = []
    model_hashes = json.loads((HYBRID / "reused_model_hashes.json").read_text())
    for fold in range(3):
        path = PREVIOUS / f"hydrology-sites-v2_RF_fold{fold}.joblib"
        assert sha256_file(path) == model_hashes[str(path.relative_to(ROOT))]
        train = pd.read_csv(PRIOR / f"training_manifest_RF_fold{fold}.csv")
        held = set(roads.loc[roads.fold.eq(fold), "corridor"])
        assert not set(train.corridor) & held
        assert train.distance_to_heldout_road_m.min() >= 100 - 1e-7
        line = dict(
            fold=fold,
            training_rows=len(train),
            heldout_inventory=int(eligible.fold.eq(fold).sum()),
            heldout_routes=sorted(held),
            overlapping_training_routes=0,
            minimum_training_distance_m=float(train.distance_to_heldout_road_m.min()),
            saved_model_sha256=sha256_file(path),
        )
        lineage.append(line)
    save_json(
        out / "prediction_lineage.json",
        dict(
            folds=lineage,
            prediction_source="saved v2 held-out score and site artifacts; no predict calls",
            original_frozen_v1_used=False,
            feature_parity_failed_fields=35,
            independent=False,
            development_inventory_exposure=True,
            feature_development_used_exposed_roads=True,
            evaluation_influenced_previous_research=True,
            candidate_geometry_not_generated_from_inventory=True,
            road_network="121 recovered whole segments,22 corridor groups; frozen previous selection",
            assignment="inventory-to-road matching assigns evaluation strata only, never proposals",
            caution="Model-copy route purges do not undo development exposure or establish independence.",
        ),
    )


def density_experiments(out, roads, eligible, all_truth):
    km = float(roads.length.sum() / 1000)
    uniform_rows, per_route, cohorts, sites = [], [], {}, {}
    for spacing in INTERVALS:
        points = uniform(roads, spacing=spacing)
        points.to_parquet(out / f"uniform_{spacing}m.parquet", index=False)
        for scope, truth in (("eligible705", eligible), ("all714", all_truth)):
            for tol in TOLERANCES:
                uniform_rows.append(
                    coverage_record(
                        "uniform", points, truth, km, tol, interval_m=spacing, scope=scope
                    )
                )
        for route in sorted(roads.corridor.unique()):
            p = points.loc[points.route_id.eq(route)]
            t = eligible.loc[eligible.corridor.eq(route)]
            length = float(roads.loc[roads.corridor.eq(route)].length.sum() / 1000)
            for tol in TOLERANCES:
                per_route.append(
                    coverage_record("uniform", p, t, length, tol, route=route, interval_m=spacing)
                )
        if spacing == 10:
            cohorts["uniform10"] = points
        print(f"Uniform {spacing}m: {len(points)} candidates", flush=True)
    pd.DataFrame(uniform_rows).to_csv(out / "uniform_sampling.csv", index=False)
    pd.DataFrame(per_route).to_csv(out / "uniform_by_route.csv", index=False)
    for method, file in (
        ("historical", "historical_proposals"),
        ("hybrid", "hybrid_proposals"),
        ("crossing_only", "crossing_proposals"),
    ):
        cohorts[method] = gpd.read_parquet(HYBRID / f"{file}.parquet")
    cross = cohorts["crossing_only"]
    cohorts["dem_crossings"] = cross.loc[cross.proposal_kind.eq("dem_crossing")].reset_index(
        drop=True
    )
    cohorts["mapped_crossings"] = cross.loc[cross.proposal_kind.eq("mapped_crossing")].reset_index(
        drop=True
    )
    density, route_metrics, curves = [], [], []
    for method, candidates in cohorts.items():
        # Same prior score-independent site definition, not retuned using labels.
        physical = cluster_sites(candidates)
        sites[method] = physical
        physical.to_parquet(out / f"scorefree_sites_{method}.parquet", index=False)
        for stage, frame in (("candidate", candidates), ("physical_site", physical)):
            density.append(coverage_record(method, frame, eligible, km, stage=stage))
            for route in sorted(roads.corridor.unique()):
                length = float(roads.loc[roads.corridor.eq(route)].length.sum() / 1000)
                route_metrics.append(
                    coverage_record(
                        method,
                        frame.loc[frame.route_id.eq(route)],
                        eligible.loc[eligible.corridor.eq(route)],
                        length,
                        stage=stage,
                        route=route,
                    )
                )
            edges = adjacency(frame, eligible)
            budgets = sorted(set([b for b in BUDGETS if b <= len(frame)] + [len(frame)]))
            draws = {b: [] for b in budgets}
            for rep in range(N_THIN):
                order = stream_seed("thin", method, stage, rep).permutation(len(frame))
                vals = prefix_matches(edges, order, budgets)
                for b, value in zip(budgets, vals):
                    draws[b].append(int(value))
            for b in budgets:
                arr = np.asarray(draws[b])
                curves.append(
                    dict(
                        method=method,
                        stage=stage,
                        budget=b,
                        density_per_km=b / km,
                        trials=N_THIN,
                        mean_matches=float(arr.mean()),
                        p025=float(np.quantile(arr, 0.025)),
                        p975=float(np.quantile(arr, 0.975)),
                        allocation="seeded uniform subset, no labels",
                    )
                )
        print(
            f"Density control {method}: {len(candidates)} candidates, {len(physical)} sites",
            flush=True,
        )
    # Exact-budget uniform spacing is a second, non-clumpy comparator, not a subset.
    matched_budgets = sorted(set(row["budget"] for row in curves))
    exact = []
    for budget in matched_budgets:
        points = uniform(roads, budget=budget)
        row = coverage_record("uniform_exact_budget", points, eligible, km, budget=budget)
        exact.append(row)
    pd.DataFrame(density).to_csv(out / "density_full_population.csv", index=False)
    pd.DataFrame(route_metrics).to_csv(out / "density_by_route.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "matched_budget_curves.csv", index=False)
    pd.DataFrame(exact).to_csv(out / "uniform_exact_budgets.csv", index=False)
    comparisons = pd.DataFrame(curves).merge(
        pd.DataFrame(exact)[["budget", "matches_one_to_one"]], on="budget"
    )
    comparisons["excess_over_exact_uniform"] = (
        comparisons.mean_matches - comparisons.matches_one_to_one
    )
    comparisons["lift_over_exact_uniform"] = (
        comparisons.mean_matches / comparisons.matches_one_to_one.replace(0, np.nan)
    )
    comparisons.to_csv(out / "density_controlled_comparison.csv", index=False)
    return cohorts, sites


def ranking_experiments(out, roads, eligible):
    records, route_records, concentrations, strata_rows, geometry_checks = [], [], [], [], []
    replicate_rows = []
    for method in ("historical", "hybrid", "crossing_only"):
        for fold in range(3):
            real = (
                eligible.loc[eligible.fold.eq(fold)]
                .sort_values("record_key")
                .reset_index(drop=True)
            )
            raw = gpd.read_parquet(HYBRID / f"sites_{method}_raw_rf_fold{fold}.parquet")
            rz = gpd.read_parquet(HYBRID / f"sites_{method}_dynamic_route_z_fold{fold}.parquet")
            # Raw and RZ have identical representatives: explicitly verify, don't assume.
            sites = raw.sort_values("physical_site_id").reset_index(drop=True)
            other = rz.set_index("physical_site_id").loc[sites.physical_site_id]
            assert len(raw) == len(rz) and raw.physical_site_id.is_unique
            assert set(raw.physical_site_id) == set(rz.physical_site_id)
            assert np.max(np.linalg.norm(xy(sites) - xy(other), axis=1)) < 1e-9
            np.testing.assert_array_equal(sites.route_id.to_numpy(), other.route_id.to_numpy())
            np.testing.assert_allclose(sites.model_score, other.model_score, rtol=0, atol=1e-12)
            # Verify preserved candidate scores against the previous v2 prediction cache.
            saved = gpd.read_parquet(PREVIOUS / f"scores_RF_fold{fold}.parquet").set_index(
                "candidate_id"
            )
            common = sites.loc[sites.candidate_id.isin(saved.index)]
            np.testing.assert_allclose(
                common.model_score,
                saved.loc[common.candidate_id, "model_score"],
                rtol=0,
                atol=1e-12,
            )
            geometry_checks.append(
                dict(
                    method=method,
                    fold=fold,
                    sites=len(sites),
                    identical_raw_rz_site_ids=True,
                    geometry_max_difference_m=0,
                    previous_score_cache_verified=len(common),
                    additional_previously_saved_scores=len(sites) - len(common),
                )
            )
            edges = adjacency(sites, real)
            index = pd.Series(np.arange(len(sites)), index=sites.physical_site_id)
            routes = sites.route_id.astype(str).to_numpy()
            blocks = (
                sites.route_id.astype(str)
                + "|"
                + sites.road_id.astype(str)
                + "|"
                + sites.road_part_index.astype(str)
                + "|"
                + (sites.road_chainage_m // 2000).astype(int).astype(str)
            ).to_numpy()
            for control, labels in (("route", routes), ("block2km", blocks)):
                for label in sorted(set(labels)):
                    n = int((labels == label).sum())
                    strata_rows.append(
                        dict(
                            method=method,
                            fold=fold,
                            control=control,
                            stratum=label,
                            n=n,
                            tiny_below10=n < 10,
                        )
                    )
            for policy, ordered in (("raw_rf", raw), ("dynamic_route_z", rz)):
                order = index.loc[ordered.sort_values("global_rank").physical_site_id].to_numpy()
                observed = prefix_matches(edges, order)
                route_obs = {}
                route_edges = {}
                for route in sorted(set(routes)):
                    mask = real.corridor.astype(str).eq(route).to_numpy()
                    # Per-route truth capacity is separate from global matching near junctions.
                    allowed = set(np.flatnonzero(mask))
                    route_edges[route] = [[j for j in links if j in allowed] for links in edges]
                    route_obs[route] = prefix_matches(route_edges[route], order)
                for control in ("county", "route", "block2km"):
                    values = np.zeros((N_RANDOM, len(KS)), dtype=int)
                    by_route = (
                        {r: np.zeros_like(values) for r in route_obs} if control == "route" else {}
                    )
                    for rep in range(N_RANDOM):
                        rng = stream_seed("ranking", method, fold, policy, control, rep)
                        perm = (
                            rng.permutation(len(sites))
                            if control == "county"
                            else conditional_order(
                                order, routes if control == "route" else blocks, rng
                            )
                        )
                        values[rep] = prefix_matches(edges, perm)
                        if control == "route":
                            # Only count selections on their own route, never foreign-route sites.
                            for route in by_route:
                                filtered = [int(s) for s in perm[: max(KS)] if routes[s] == route]
                                slots = [
                                    int((routes[perm[: min(k, len(perm))]] == route).sum())
                                    for k in KS
                                ]
                                by_route[route][rep] = (
                                    prefix_matches(route_edges[route], filtered, slots)
                                    if filtered
                                    else 0
                                )
                        replicate_rows.extend(
                            dict(
                                method=method,
                                fold=fold,
                                policy=policy,
                                control=control,
                                replicate=rep,
                                k=k,
                                matched=int(v),
                            )
                            for k, v in zip(KS, values[rep])
                        )
                    for col, k in enumerate(KS):
                        records.append(
                            dict(
                                method=method,
                                fold=fold,
                                policy=policy,
                                control=control,
                                k=k,
                                actual_slots=min(k, len(sites)),
                                population_sites=len(sites),
                                **summary(values[:, col], observed[col]),
                            )
                        )
                        top = sites.iloc[order[:k]]
                        shares = top.route_id.value_counts()
                        concentrations.append(
                            dict(
                                method=method,
                                fold=fold,
                                policy=policy,
                                k=k,
                                largest_route=str(shares.index[0]),
                                largest_share=float(shares.iloc[0] / len(top)),
                                roads_without_slots=int(len(set(routes)) - len(shares)),
                            )
                        )
                    for route, arr in by_route.items():
                        for col, k in enumerate(KS):
                            slots = int((routes[order[:k]] == route).sum())
                            own = [int(s) for s in order[:k] if routes[s] == route]
                            obs = cardinality(route_edges[route], own, len(real))
                            route_records.append(
                                dict(
                                    method=method,
                                    fold=fold,
                                    policy=policy,
                                    route=route,
                                    k=k,
                                    slots=slots,
                                    inventory=int(real.corridor.astype(str).eq(route).sum()),
                                    population_sites=int((routes == route).sum()),
                                    **summary(arr[:, col], obs),
                                )
                            )
                print(
                    f"Randomization {method} fold{fold} {policy}: {observed.tolist()}", flush=True
                )
    results = pd.DataFrame(records)
    replicates = pd.DataFrame(replicate_rows)
    # Save individual draws, not only summary statistics, allowing exact re-analysis.
    replicates.to_parquet(out / "random_replicates.parquet", index=False)
    results.to_csv(out / "ranking_random_by_fold.csv", index=False)
    pd.DataFrame(route_records).to_csv(out / "ranking_route_controlled_by_route.csv", index=False)
    pd.DataFrame(concentrations).drop_duplicates().to_csv(
        out / "route_concentration.csv", index=False
    )
    pd.DataFrame(strata_rows).to_csv(out / "randomization_strata.csv", index=False)
    save_json(out / "identical_site_geometry_checks.json", geometry_checks)
    combined, summed_draws = [], []
    for (method, policy, control, k), group in replicates.groupby(
        ["method", "policy", "control", "k"]
    ):
        sums = group.groupby("replicate").matched.sum().sort_index()
        measured = results.loc[
            (results.method == method)
            & (results.policy == policy)
            & (results.control == control)
            & (results.k == k)
        ]
        assert len(measured) == 3
        combined.append(
            dict(
                method=method,
                policy=policy,
                control=control,
                k=k,
                actual_slots=int(measured.actual_slots.sum()),
                **summary(sums.to_numpy(), measured.observed.sum()),
            )
        )
        summed_draws.extend(
            dict(
                method=method, policy=policy, control=control, k=k, replicate=int(r), matched=int(v)
            )
            for r, v in sums.items()
        )
    pd.DataFrame(combined).to_csv(out / "ranking_random_combined.csv", index=False)
    pd.DataFrame(summed_draws).to_parquet(out / "random_combined_replicates.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    original = protected()
    inputs = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(input_paths())}
    save_json(out / "input_hashes_before.json", inputs)
    save_json(
        out / "fixed_method.json",
        dict(
            seed=SEED,
            ranking_permutations=N_RANDOM,
            density_subsets=N_THIN,
            intervals_m=INTERVALS,
            tolerances_m=TOLERANCES,
            ranking_k=KS,
            budgets=BUDGETS,
            matching="maximum cardinality one-to-one20m, inclusive",
            representative_rule="unchanged staging score-free20m topology-aware anchor consolidation",
            strata="route; route/road-ID/part/floor(chainage/2000m), no labels",
            maps="first three route identifiers in lexical order;first stable road segment per route;2km centered at segment midpoint",
            null_interval="2.5th–97.5th percentiles of random outcomes, not confidence interval for deployment accuracy",
            p_value="(1+draws>=observed)/(1000+1), descriptive multiple exposed comparisons, no model selection",
            scope="same121 roads,705 eligible plus714 denominator audit;three separate budgets not pooled model scores",
            fits=0,
            prediction_calls=0,
            independent=False,
            warning="Recommendations require field verification.",
        ),
    )
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    roads["id"] = roads.segment_id.astype(str)
    truth = inventory()
    eligible = truth.loc[truth.fold.ge(0)].copy().reset_index(drop=True)
    assert len(roads) == 121 and len(eligible) == 705 and len(truth) == 714
    provenance(out, roads, eligible)
    density_experiments(out, roads, eligible, truth)
    ranking_experiments(out, roads, eligible)
    assert original == protected()
    after = {p: sha256_file(ROOT / p) for p in inputs}
    assert after == inputs
    save_json(
        out / "integrity.json",
        dict(
            all_inputs_unchanged=True,
            input_files=len(inputs),
            protected_unchanged=len(original),
            models_trained=0,
            prediction_calls=0,
            production_changed=False,
            staging_maps_unchanged=True,
        ),
    )
    save_json(out / "COMPUTATIONS_COMPLETE.json", dict(seed=SEED, permutations=N_RANDOM, fits=0))
    print("Density and ranking computations complete; all inputs unchanged", flush=True)


if __name__ == "__main__":
    main()
