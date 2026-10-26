"""Assemble density-control tables, provenance, map data, and an evidence-led report.

Does not fit or call estimators. Plot data are read by plot_density_controls.py
using the bundled document/image Python runtime (Pillow, no new dependencies).
"""

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from scripts.validate_density_controls import (
    OUT,
    HYBRID,
    PRIOR,
    KS,
    inventory,
    xy,
)
from scripts.validate_inventory_heldout_roads import match_one_to_one
from scripts.evaluate_hydrology_sites import save_json, protected


def table(frame):
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)

    return (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join("---" for _ in frame.columns)
        + " |\n"
        + "\n".join(
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
    )


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    running = 0
    for rank, index in enumerate(order):
        running = max(running, min(1, (len(order) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def maps(out, roads, truth):
    uniform = gpd.read_parquet(out / "uniform_10m.parquet")
    crossing = gpd.read_parquet(HYBRID / "crossing_proposals.parquet")
    ranked = gpd.GeoDataFrame(
        pd.concat(
            [
                gpd.read_parquet(HYBRID / f"sites_hybrid_raw_rf_fold{f}.parquet").head(250)
                for f in range(3)
            ],
            ignore_index=True,
        ),
        crs=32618,
    )
    payload = []
    for route in sorted(roads.corridor.unique())[:3]:
        road = roads.loc[roads.corridor.eq(route)].sort_values("segment_id").iloc[0]
        geom = road.geometry
        part = max(getattr(geom, "geoms", [geom]), key=lambda p: p.length)
        # Exactly preregistered rule; no inventory or matches choose the window.
        center = part.interpolate(0.5 * part.length)
        bounds = [center.x - 1000, center.y - 1000, center.x + 1000, center.y + 1000]
        window = box(*bounds)
        lines = []
        for g in roads.loc[roads.intersects(window)].geometry.intersection(window):
            for line in getattr(g, "geoms", [g]):
                if line.geom_type == "LineString":
                    lines.append(list(line.coords))
        layers = {}
        for name, frame in (
            ("uniform10", uniform),
            ("inventory", truth),
            ("crossing", crossing),
            ("raw_top250", ranked),
        ):
            layers[name] = xy(frame.loc[frame.intersects(window)]).tolist()
        payload.append(
            dict(
                route=str(route),
                segment_id=str(road.segment_id),
                bounds=bounds,
                roads=lines,
                layers=layers,
                selection="lexical route,first stable segment,segment midpoint,2km square",
            )
        )
    save_json(out / "map_examples.json", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT)
    out = parser.parse_args().input
    assert (out / "COMPUTATIONS_COMPLETE.json").exists()
    protected()
    inputs = json.loads((out / "input_hashes_before.json").read_text())
    assert all(sha256_file(ROOT / p) == digest for p, digest in inputs.items())
    uniform = pd.read_csv(out / "uniform_sampling.csv")
    eligible = uniform.loc[uniform.scope.eq("eligible705")]
    t1 = eligible.pivot(
        index=["interval_m", "candidates", "density_per_km"],
        columns="tolerance_m",
        values="covered_any",
    ).reset_index()
    t1.columns = [
        "interval_m",
        "candidate_count",
        "candidates_per_km",
        "coverage_5m",
        "coverage_10m",
        "coverage_20m",
        "coverage_25m",
    ]
    t1 = t1.sort_values("interval_m")
    t1.to_csv(out / "table1_uniform.csv", index=False)
    full = pd.read_csv(out / "density_full_population.csv")
    a = full.loc[full.stage.eq("candidate")].set_index("method")
    b = full.loc[full.stage.eq("physical_site")].set_index("method")
    t2 = pd.DataFrame(
        dict(
            candidates=a.candidates,
            candidates_per_km=a.density_per_km,
            physical_sites=b.candidates,
            sites_per_km=b.density_per_km,
            candidate_coverage20m=a.covered_any,
            site_matches20m=b.matches_one_to_one,
            matches_per100_sites=b.matches_per_100,
        )
    ).reset_index()
    t2.to_csv(out / "table2_density.csv", index=False)
    summary = pd.read_csv(out / "ranking_random_combined.csv")
    summary["p_holm_108"] = holm(summary.p_one_sided.to_numpy())
    summary.to_csv(out / "ranking_random_adjusted.csv", index=False)
    t3 = summary.loc[
        summary.method.eq("hybrid"),
        [
            "policy",
            "control",
            "k",
            "actual_slots",
            "random_mean",
            "p025",
            "p975",
            "observed",
            "lift",
            "excess_matches",
            "p_one_sided",
            "p_holm_108",
        ],
    ]
    t3.to_csv(out / "table3_ranking.csv", index=False)
    route = pd.read_csv(out / "ranking_route_controlled_by_route.csv", dtype={"route": str})
    route = route.loc[route.method.eq("hybrid") & route.k.eq(250)].copy()
    ur = pd.read_csv(out / "uniform_by_route.csv", dtype={"route": str})
    ur = ur.loc[
        ur.interval_m.eq(10) & ur.tolerance_m.eq(20), ["route", "matches_one_to_one"]
    ].rename(columns={"matches_one_to_one": "uniform10_full_matches"})
    counts = pd.read_csv(out / "density_by_route.csv", dtype={"route": str})
    counts = counts.loc[
        counts.method.eq("hybrid") & counts.stage.eq("candidate"), ["route", "candidates"]
    ]
    t4 = route.merge(ur, on="route").merge(counts, on="route")[
        [
            "route",
            "policy",
            "inventory",
            "candidates",
            "population_sites",
            "uniform10_full_matches",
            "slots",
            "random_mean",
            "observed",
            "lift",
            "p_one_sided",
        ]
    ]
    t4.to_csv(out / "table4_routes.csv", index=False)
    # Verify every observed TopK directly using the independent sparse implementation.
    verify = []
    truth = inventory()
    real = truth.loc[truth.fold.ge(0)].copy()
    for method in ("historical", "hybrid", "crossing_only"):
        for policy in ("raw_rf", "dynamic_route_z"):
            for fold in range(3):
                frame = gpd.read_parquet(
                    HYBRID / f"sites_{method}_{policy}_fold{fold}.parquet"
                ).sort_values("global_rank")
                t = real.loc[real.fold.eq(fold)]
                for k in KS:
                    n = len(match_one_to_one(frame.head(k), xy(t)))
                    saved = pd.read_csv(out / "ranking_random_by_fold.csv")
                    v = saved.loc[
                        saved.method.eq(method)
                        & saved.policy.eq(policy)
                        & saved.fold.eq(fold)
                        & saved.k.eq(k),
                        "observed",
                    ]
                    assert (v == n).all()
                    verify.append(
                        dict(
                            method=method,
                            policy=policy,
                            fold=fold,
                            k=k,
                            matched=n,
                            scipy_verified=True,
                        )
                    )
    pd.DataFrame(verify).to_csv(out / "observed_matching_crosscheck.csv", index=False)
    stages = []
    for method in (
        "uniform10",
        "historical",
        "hybrid",
        "crossing_only",
        "dem_crossings",
        "mapped_crossings",
    ):
        stages.append(
            dict(
                stage="candidate generation",
                method=method,
                value=int(a.loc[method, "matches_one_to_one"]),
                unit="full candidate one-to-one matches/705",
                interpretation="coverage,not detection accuracy",
            )
        )
        stages.append(
            dict(
                stage="score-free consolidation",
                method=method,
                value=int(b.loc[method, "matches_one_to_one"]),
                unit="full proposed-site one-to-one matches/705",
                interpretation="geometry/representative effect",
            )
        )
    scored = pd.read_csv(HYBRID / "score_ordered_coverage.csv")
    for method in ("historical", "hybrid", "crossing_only"):
        value = int(
            scored.loc[
                scored.method.eq(method) & scored.phase.eq("raw_rf"),
                "consolidated_matches_one_to_one",
            ].sum()
        )
        stages.append(
            dict(
                stage="score-ordered consolidation",
                method=method,
                value=value,
                unit="full proposed-site matches summed over3folds/705",
                interpretation="representative selection,not ranking",
            )
        )
    for row in t3.loc[t3.k.eq(250)].itertuples():
        stages.append(
            dict(
                stage="ranking over fixed geometry",
                method=f"hybrid {row.policy} vs {row.control}",
                value=float(row.excess_matches),
                unit="excess matches over null at3xTop250",
                interpretation=f"observed{row.observed},null mean{row.random_mean:.3f};p={row.p_one_sided:.4f}",
            )
        )
    t5 = pd.DataFrame(stages)
    t5.to_csv(out / "table5_contributions.csv", index=False)
    # Matched source-control comparisons include full candidate counts, not only convenient budgets.
    compare = pd.read_csv(out / "density_controlled_comparison.csv")
    source = compare.loc[
        compare.method.isin(["crossing_only", "dem_crossings", "mapped_crossings"])
    ]
    source = source.sort_values("budget").groupby(["method", "stage"]).tail(1)
    source.to_csv(out / "physical_signal_density_control.csv", index=False)
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    maps(out, roads, real)
    # All-budget candidate/site localisation distributions, not only successful matches.
    loc = []
    for method, file in [
        ("uniform10", out / "uniform_10m.parquet"),
        ("historical", HYBRID / "historical_proposals.parquet"),
        ("hybrid", HYBRID / "hybrid_proposals.parquet"),
        ("crossing_only", HYBRID / "crossing_proposals.parquet"),
    ]:
        for stage, path in (
            ("candidate", file),
            ("physical_site", out / f"scorefree_sites_{method}.parquet"),
        ):
            frame = gpd.read_parquet(path)
            pairs = match_one_to_one(frame, xy(real))
            distances = pairs.distance_m.to_numpy(float)
            loc.append(
                dict(
                    method=method,
                    stage=stage,
                    matches=len(pairs),
                    matched_distance_mean_m=float(distances.mean()),
                    matched_distance_median_m=float(np.median(distances)),
                    matched_distance_p90_m=float(np.quantile(distances, 0.9)),
                )
            )
    pd.DataFrame(loc).to_csv(out / "matched_localization_distances.csv", index=False)
    u = t1.iloc[0]
    main = t3.loc[t3.k.eq(250)]
    raw = main.loc[(main.policy == "raw_rf") & (main.control == "county")].iloc[0]
    rz = main.loc[(main.policy == "dynamic_route_z") & (main.control == "county")].iloc[0]
    rr = main.loc[(main.policy == "raw_rf") & (main.control == "route")].iloc[0]
    zr = main.loc[(main.policy == "dynamic_route_z") & (main.control == "route")].iloc[0]
    conditional = main.loc[main.control.isin(["route", "block2km"])]
    consistent = bool(((conditional.p_one_sided < 0.05) & (conditional.lift > 1)).all())
    decision = (
        "A. MODEL SHOWS MEANINGFUL RANKING SIGNAL"
        if consistent
        else (
            "B. COVERAGE IS MOSTLY EXPLAINED BY DENSE ROAD SAMPLING"
            if u.coverage_20m >= 690
            else "D. EVIDENCE IS INCONCLUSIVE"
        )
    )
    strata = pd.read_csv(out / "randomization_strata.csv")
    tiny = int(strata.loc[strata.control.eq("block2km"), "tiny_below10"].sum())
    rank250 = summary.loc[
        summary.k.eq(250),
        [
            "method",
            "policy",
            "control",
            "observed",
            "random_mean",
            "p025",
            "p975",
            "lift",
            "p_one_sided",
            "p_holm_108",
        ],
    ]
    draws = pd.read_parquet(out / "random_replicates.parquet")
    plot_draws = pd.read_parquet(out / "random_combined_replicates.parquet")
    plot_draws.loc[
        plot_draws.method.eq("hybrid") & plot_draws.k.eq(250) & plot_draws.control.eq("county")
    ].to_csv(out / "figure_random_draws.csv", index=False)
    seeds = draws[["method", "fold", "policy", "control", "replicate"]].drop_duplicates()
    seeds["seed_uint64"] = [
        str(
            int.from_bytes(
                hashlib.sha256(("20261015|ranking|" + "|".join(map(str, row))).encode()).digest()[
                    :8
                ],
                "little",
            )
        )
        for row in seeds.itertuples(index=False, name=None)
    ]
    seeds.to_csv(out / "random_seeds.csv", index=False)
    report = f"""# Density-controlled detection validation

2026-10-15. **{decision}**

Recommendations require field verification. Zero model fits, zero estimator prediction calls, no production/UI/staging-map changes, no deployment or promotion. This is retrospective development evaluation, not an independent field test.

## Main measured findings

Uniform10m sampling alone covers **{int(u.coverage_20m)}/705** eligible known culverts at20m. It uses{int(u.candidate_count):,} samples on the identical121 supplied whole road segments,22corridors. No DEM, stream, ML, inventory coordinate, label or inspection record generates these baseline locations. Thus high candidate coverage is achievable without learned detection. The archived51,417 scored pool is a different selected population; it is not substituted into these controlled comparisons.

On identical hybrid score-ordered site geometry, rawRF obtains **{int(raw.observed)}** matches across three separateTop250 queues (750slots), versus county-random mean **{raw.random_mean:.3f}** and route-controlled mean **{rr.random_mean:.3f}**. ROUTE-Z obtains **{int(rz.observed)}**, versus county-random **{rz.random_mean:.3f}** and route-controlled **{zr.random_mean:.3f}**. Neither random permutations nor nominal p-values create independent validation. Inventory-unmatched sites are not verified false positives; these are inventory matches, not accuracy or calibrated probabilities.

The dense system's failure is **not evidence that every ranking has no information**. Crossing-only RF raw58 andROUTE-Z59 exceed route-null44.058 and45.089 (about1.31×;both nominalp=.001), and block-null50.302 and50.783 (about1.15–1.16×;p=.002/.001). That is a narrow retrospective enrichment on a low-recall proposal population, not countywide detection accuracy. Hybrid ROUTE-Z's route-level excess disappears under2km block control:31observed versus32.481expected,p=.671. Hybridraw17 versus local19.587,p=.829. The dense high-recall queue does not show consistent within-local-road ranking value.

## Fixed methods and provenance

`fixed_method.json` was written before computations. Seed20261015, SHA256-derived separate deterministic streams;1,000 permutations for every method/fold/policy/null and100 label-free subsets for every density curve. Intervals10,20,40,75,150,300m; tolerances5,10,20,25m. No density, threshold, road subset or map example was chosen to improve observed labels.

Road lengths: segment-sum **{roads.length.sum() / 1000:.6f}km**; dissolved geometry **{roads.geometry.union_all().length / 1000:.6f}km**. All densities use segment-sum, including concurrent identities; same denominator throughout. Sampling starts half an interval from each stable road-part origin; short parts use midpoint; no endpoint additions. Exact-budget controls concatenate stable road parts and apply a half-offset grid, preserving exact budget without labels. Curves also show random thinning of each full candidate or score-free site population; thinning and fresh uniform placement are distinct controls. At a site's matched budget the exact uniform comparator is that many proposed road points (not an extra20m clustering); the same-budget uniform10-site subsets provide the consolidated-site control.

NativeEPSG32618 metre distances throughout; official26918 coordinates transformed. Eligible705 means previous geometric road assignment; nine of714 original inventory records lie outside the evaluated footprint. `uniform_sampling.csv` retains all714 denominator rows. Candidate coverage counts each known record with any nearby point; one-to-one maximum bipartite matching limits both candidate/site and inventory to one match. Tolerances are inclusive. Randomization uses exact incremental augmenting paths, independently crosschecked against SciPy maximum matching for every observed TopK. No precomputed binary match labels shortcut site-capacity conflicts.

Coverage tables match all eligible705 jointly. Ranking tests match within each fixed fold, then sum counts; they do not pool scores from separate models. At road junctions these matching populations can produce different full-list capacities. For example75m uniform has338 joint matches here versus334 within-fold one-to-one in the prior audit; neither is silently substituted for the other.

Current v2 saved evaluation-copy RF predictions only: fixed3 road folds,800trees/48feature schema, original100m road purge. All model hashes checked against previous run; every shared representative score checked against previous prediction cache. Added mapped-point scores are read from the preceding saved site evaluation only. `prediction_lineage.json` and `identical_site_geometry_checks.json` record this. Original frozenv1RF is **not** run against v2 features that failed35/48parity checks. No feature extraction, model loading, fitting or predict calls occur in this audit.

All714 inventory records have development exposure; previous research outcomes influenced development and feature choices. Purged folds avoid direct training-road overlap for those copies but do not erase this exposure. Inventory-to-road and spatial matching are evaluation operations only; proposals are neither generated nor repositioned using inventory. Historical33/250vs5/250 and latest34vs31over750 are different pipelines/populations and are not pooled. The present historical same-network saved v2 comparator reproduces18raw and34ROUTE-Z over750slots; hybrid17and31. These are not the original frozen model's33and5 benchmark.

## Table1 — uniform sampling

Coverage columns count nearby inventory records/705; candidate budget is explicit for every observed level. Full percentages and one-to-one results at all four tolerances and both denominators are in `uniform_sampling.csv`; all22route breakdowns in `uniform_by_route.csv`.

{table(t1)}

## Table2 — full-population coverage (descriptive, NOT density matched)

{table(t2)}

### Matched-budget physical signal

The following rows use each physical source's full candidate OR consolidated-site budget; the uniform comparator has exactly the same number of label-free samples on the same network.100 subset repetitions cannot change the full-population row. Do not interpret superior coverage at a larger budget as intelligent detection. Full curves for candidate and consolidated-site budgets are saved, including uncertainty over subsets, in `matched_budget_curves.csv`, `uniform_exact_budgets.csv`, and `density_controlled_comparison.csv`.

{table(source[["method", "stage", "budget", "mean_matches", "matches_one_to_one", "excess_over_exact_uniform", "lift_over_exact_uniform"]])}

At2,034raw crossings,63matches versus68exact-uniform matches shows no aggregate candidate localization advantage. DEM-only1,838points yield62versus56 (six extra); mapped196yield2versus9. After crossing consolidation,1,054proposals retain62matches versus43uniform, while978DEMsites retain62versus26uniform: useful low-budget physical localization, but only62/705inventory recall. At26,370hybridscore-free sites,653matches versus687exact-uniform matches; this dense system does not outperform fresh regular spacing at equal budget. These one-phase comparisons are descriptive; the separately seeded uniform-site subset curves provide a second density control, not independent field evidence.

## Table3 — ranking against random on identical hybrid sites

Each K is per held-out group, not a pooled TopK across different model scores. Actual slots handles crossing queues shorter than500. All three candidate methods have their own saved full tables; rawRF and ROUTE-Z are checked to have identical site IDs,coordinates,routes,scores,population. Random ranking only permutes these identities. County null is countywide within each held-out group. Route null independently shuffles all identities within a corridor while preserving the model's exact corridor at **every rank slot**, hence all TopK allocations simultaneously. Block null additionally preserves roadID/part/floor(chainage/2000m), chosen without labels.

`p025–p975` is the central95% null-outcome interval, not a confidence interval for model generalization. Mean,median,sampleSD,5th/95th,2.5th/97.5th,best/worst and plus-one empirical one-sidedp are in fold/combined CSVs. Holm-adjusted p covers all108 combined comparisons (3methods×2policies×3nulls×6K); effect sizes remain descriptive on exposed data. Zero random mean gives undefined lift, not infinity.

{table(t3)}

### All three methods at3×Top250

{table(rank250)}

The1,000-draw Monte Carlo p-value resolution is1/1001. Across108comparisons, even that minimum cannot pass the most conservative Holm first step(.05/108); adjusted non-significance must not be misrepresented as proof of no signal. Report observed lift,null ranges,nominalp,and local controls together. The broad density-driven decision rests on reproducible uniform coverage and absent consistent dense-queue conditional enrichment, not on this p-resolution limitation.

## Table4 — route-by-route, hybrid Top250 per group

`uniform10_full_matches` uses its FULL dense candidate budget, not the model's route slots; do not compare it as a ranking baseline. Route random expectation uses exactly `slots` and the SAME score-ordered site population. Zero-slot roads remain visible. Route-specific matching uses the assigned route's inventory and its own selected sites; junction ambiguity means route totals need not exactly equal unrestricted global matching. Per-route exploratory p-values are unadjusted and not used to choose routes or models.

{table(t4)}

Fixed2km block controls are sensitivity checks, not independent replicates: {tiny} method/fold strata have fewer than10sites. `randomization_strata.csv` exposes every population size, including tiny/immobile strata. No tiny-stratum significance claims. Road concentration for every method/policy/fold/K, including zero-slot counts, is in `route_concentration.csv`.

## Table5 — source of apparent performance

These are distinct stages, not additive causal percentages. The model does not deserve credit for a raw road point being near inventory. Representative selection can gain/lose coverage before ranking; randomization freezes that geometry. Comparing score-free and score-ordered full coverage measures selection effects, not pure ranking. Model ordering on score-free sites is not reconstructed because not all added mapped representatives have saved compatible scores; no inference is run to fill cells. Pure raw-vsROUTE-Z ranking is feasible and measured on their identical representatives.

{table(t5)}

## Answers to the eight research questions

1. **Uniform sampling alone?** Yes: measured{int(u.coverage_20m)}/705 at10m,20m tolerance; no intelligent generator required. At narrower tolerances use Table1 rather than claiming precise structure localization.
2. **Does density explain653–657?** Dense uniform score-free sites achieve{int(b.loc["uniform10", "matches_one_to_one"])}/705; historical and hybrid score-free sites each653. Hybrid score-ordered selection657 versus653 changes only four matches net. Density provides the overwhelming full-list coverage; representative geometry explains additional losses/gains. Different anchor orders and topology can yield different site counts even at the same20m radius; matched-budget controls expose this rather than assuming equal costs.
3. **RF beats random on identical geometry?** RawRF hybrid lift={raw.lift:.3f} versus county null, excess={raw.excess_matches:.3f} matches/750slots,p={raw.p_one_sided:.4f}. ROUTE-Z lift={rz.lift:.3f},excess={rz.excess_matches:.3f},p={rz.p_one_sided:.4f}. See all K and method results before generalizing.
4. **After controlling route concentration?** RawRF hybrid lift={rr.lift:.3f},excess={rr.excess_matches:.3f},p={rr.p_one_sided:.4f}; ROUTE-Z lift={zr.lift:.3f},excess={zr.excess_matches:.3f},p={zr.p_one_sided:.4f}. This measures within-road prioritization relative to the fixed allocation; it is not a new road test.
5. **Hydrology beyond sampling?** Some low-budget physical-site enrichment exists:62matches/1,054crossingsites versus43exact-uniform matches at the same budget;DEM62/978versus26. But the raw2,034crossing proposals63versus68uniform show no aggregate candidate-budget advantage, and mapped-stream-only localization is poor. Crossings do not add coverage beyond dense uniform700; recall remains62/705after consolidation. The narrow crossing-only RF ranking adds about14matches versus route-random over750slots, not a general high-recall solution. Matches per100,perkm,localization and route results are saved. Missing DEM crossings never remove known inventory from denominators.
6. **Top sites better than arbitrary road locations?** Only the measured lift over geometry-fixed and route/block-fixed nulls can support that statement. Small conditional excesses or an unadjusted isolatedp do not establish consistently useful culvert discrimination; no70%accuracy claim is supported.
7. **Geographic coverage versusML?** Full candidate700 is attained withoutML; the fixed-budget ML contribution is the excess-match column, which can be negative. It is invalid to attribute700covered or653–657representative matches to a successful classifier. These data cannot assign causal percentages or verified precision because inventory is incomplete and roads are development-exposed.
8. **Next phase?** Prioritize independent field evidence with preregistered road/site sampling and verified matched AND unmatched locations, plus physical localization/feature-contract repair. Do not launch another RF/boosting search or optimize exposed thresholds. Better learning can be assessed only after trustworthy targets/features and independent evaluation exist. No field deployment recommendation follows from current coverage.

## Figures and map examples

`figures/01_candidate_density.png`, `01b_site_density.png`: coverage-versus-budget controls (same road-km denominator). `02_tolerance.png`: uniform tolerance sensitivity. `03_random_distribution.png`: both hybrid policy null distributions with observed3×Top250. `04_conditional_random.png`: county/route/block nulls and observed counts. `05_route_lift.png`: all-route conditional excess (undefined lift at zero expectation is not filled). `06_map_examples.png`: first three lexical route IDs,first stable segment,midpoint window; no selection by performance. Empty map layers are retained honestly. Map proximity is not proof of a structure; red sites are model-ranked proposals,blue symbols are drainage evidence,black inventory crosses are reference records.

## Limitations and recommendation

Retrospective single-region development exposure; incomplete inventory means no trueFP/precision;20m tolerance and centerline errors; concurrent/fragmented road duplication and conservative topology; short strata; source-network resolution and mapped-water incompleteness; normalization batch effects;35feature mismatches for frozenv1; no independent untouched roads,crew-time study or calibrated probabilities. Stated p-values test only rankings against specified nulls, not deployment generalization. No new model training or promotion is justified by this audit alone.

Reproduce: `scripts/python.sh scripts/validate_density_controls.py --output <new-directory>` then `scripts/python.sh scripts/report_density_controls.py --input <new-directory>`; render figures with bundled Python and `scripts/plot_density_controls.py --input <new-directory>`. Existing output directory is never overwritten by the computation script. Seeds and all individual permutations' match counts are saved in Parquet. All sources/models/protected files and existing staging maps are hash-guarded. See `integrity.json`, input hashes and source snapshots. Recommendations require field verification.
"""
    (out / "REPORT.md").write_text(report)
    save_json(
        out / "decision.json",
        dict(
            decision=decision,
            uniform10_coverage20=int(u.coverage_20m),
            raw_top250=int(raw.observed),
            route_z_top250=int(rz.observed),
            budget=750,
            raw_county_mean=float(raw.random_mean),
            raw_route_mean=float(rr.random_mean),
            route_z_county_mean=float(rz.random_mean),
            route_z_route_mean=float(zr.random_mean),
            independent=False,
        ),
    )
    snapshot = out / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    files = [
        ROOT / "scripts" / f
        for f in (
            "validate_density_controls.py",
            "report_density_controls.py",
            "plot_density_controls.py",
        )
    ]
    files.append(ROOT / "tests/test_density_controls.py")
    for p in files:
        (snapshot / (p.name + ".txt")).write_bytes(p.read_bytes())
    save_json(out / "code_manifest.json", {str(p.relative_to(ROOT)): sha256_file(p) for p in files})
    print(json.dumps(json.loads((out / "decision.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
