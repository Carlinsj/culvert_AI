"""Audit archival replay and summarize fixed hybrid results; never fit a model."""

# ruff: noqa: E402
import argparse
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.candidates import merge_candidate_layers
from culvert_ai.features import add_raster_samples, add_dem_hydrology_proxies
from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hydrology_sites import protected, save_json
from scripts.evaluate_hybrid_sites import DEFAULT_OUT, PREVIOUS, PRIOR, inventory, coverage_row
from scripts.validate_inventory_heldout_roads import coverage, match_one_to_one, fold_for
from staging.hybrid_service import analyze
from staging.hybrid_sites import cluster_sites


def table(frame):
    rows = [list(map(str, frame.columns)), *frame.astype(str).values.tolist()]
    return "\n".join(
        "| " + " | ".join(r) + " |" for r in [rows[0], ["---"] * len(rows[0]), *rows[1:]]
    )


def archival_replay(out):
    path = out / "archival_replay.json"
    if path.exists():
        return json.loads(path.read_text())
    inputs = [
        "actual_ulster_candidates.gpkg",
        "actual_ulster_route_candidates.gpkg",
        "actual_ulster_observed_route_candidates.gpkg",
    ]
    layers = [gpd.read_file(ROOT / "data/interim" / name) for name in inputs]
    print(
        "Replaying exact legacy merge of1341 mapped+76098 numbered+51336 observed-route proposals",
        flush=True,
    )
    replay = merge_candidate_layers(layers, min_spacing_m=5.0)
    saved = gpd.read_file(ROOT / "data/interim/actual_ulster_candidates_with_route_samples.gpkg")
    exact = (
        len(replay) == len(saved)
        and replay.candidate_id.equals(saved.candidate_id)
        and np.max(replay.geometry.distance(saved.geometry, align=False)) < 1e-8
    )
    saved_tree = cKDTree(np.c_[saved.geometry.x, saved.geometry.y])
    nearest = saved_tree.query(np.c_[replay.geometry.x, replay.geometry.y])[0]
    data = dict(
        inputs=dict(zip(inputs, [len(x) for x in layers])),
        replayed=len(replay),
        saved=len(saved),
        exact_id_and_geometry_replay=bool(exact),
        max_nearest_geometry_distance_m=float(nearest.max()),
        official_points_read_during_generation=False,
        limitation="Archival route subsets reflect old numbered/field-report route selection, not a universally51417-candidate recipe. Generic generator applies the same geometry functions to all explicitly supplied roads.",
    )
    save_json(path, data)
    return data


def raster_audit(out):
    rows = []
    for p in [
        ROOT / "data/raw/dem.tif",
        ROOT / "data/processed/hydrology/usgs_3dep_10m/dem_projected.tif",
        ROOT / "data/processed/hydrology/usgs_3dep_10m/edge_originating_flowpaths.tif",
    ]:
        with rasterio.open(p) as raster:
            rows.append(
                dict(
                    path=str(p.relative_to(ROOT)),
                    crs=raster.crs.to_string(),
                    resolution=list(raster.res),
                    nodata=raster.nodata,
                    shape=[raster.height, raster.width],
                )
            )
    parity = pd.read_csv(out / "feature_parity_48.csv")
    replay_path = out / "original_raster_feature_replay.csv"
    if not replay_path.exists():
        original = gpd.read_parquet(PRIOR / "historical10_features.parquet").set_index(
            "candidate_id"
        )
        cached = gpd.read_parquet(PREVIOUS / "physical_measurements_v2.parquet").set_index(
            "candidate_id"
        )
        ids = sorted(set(original.index) & set(cached.index))
        sampled = add_dem_hydrology_proxies(
            add_raster_samples(original.loc[ids, ["geometry"]], ROOT / "data/raw/dem.tif", "dem")
        )
        fields = [
            "slope_degrees",
            *[
                f"{prefix}_{w}x{w}{suffix}"
                for w in (9, 31)
                for prefix, suffix in [
                    ("elevation_relief", "_m"),
                    ("terrain_roughness", "_m"),
                    ("topographic_position", "_m"),
                    ("valley_depth", "_m"),
                    ("negative_tpi", "_m"),
                    ("topographic_wetness_proxy", ""),
                    ("terrain_break_score_proxy", ""),
                ]
            ],
        ]
        audit = []
        for column in fields:
            a, b = original.loc[ids, column].to_numpy(float), sampled[column].to_numpy(float)
            audit.append(
                dict(
                    feature=column,
                    rows=len(ids),
                    equal_values=int(np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True).sum()),
                )
            )
        pd.DataFrame(audit).to_csv(replay_path, index=False)
    save_json(
        out / "feature_compatibility_supplement.json",
        dict(
            different_fields=int(parity.differing_values.gt(0).sum()),
            compared_fields=48,
            matched_coordinate_rows=int(parity.paired_rows.iloc[0]),
            raster_inputs=rows,
            historical_terrain="Preserved base terrain measurements originate in data/raw/dem.tif (EPSG4269,1arcsecond grid), whereas v2 resamples the10m projected grid. Pixel-window footprints and availability therefore differ even at identical candidate coordinates.",
            edge_feature="Whitebox TraceDownslopeFlowpaths values can exceed1. v2 inherited raw path values/maxima, not a binary quality flag; current models were fitted to those v2 values. Do not silently binarize for them or call these values probabilities. Original archived feature is binary.",
            crossing_context="v1 restored archived crossing associations; v2 computes current context against the available road network. Nearest crossing angle/separation can differ without coordinate changes.",
            action="Frozen RF is incompatible. Reuse existing leakage-safe v2 evaluation copies unchanged; no fits, new features, calibration or parameter tuning.",
        ),
    )


def per_record(out):
    truth = inventory()
    records = truth[["record_key", "source_excel_row", "declared_route", "corridor", "fold"]].copy()
    for method in ("historical", "crossing_only", "hybrid"):
        source = "crossing" if method == "crossing_only" else method
        frame = gpd.read_parquet(out / f"{source}_proposals.parquet")
        sites = gpd.read_parquet(out / f"geometry_sites_{method}.parquet")
        records[f"{method}_candidate_distance_m"] = coverage(frame, truth)
        records[f"{method}_site_distance_m"] = coverage(sites, truth)
        matched = match_one_to_one(sites, np.c_[truth.geometry.x, truth.geometry.y])
        records[f"{method}_one_to_one_site_matched"] = records.index.isin(matched.real_row)
    records["hybrid_recovers_crossing_miss"] = records.hybrid_candidate_distance_m.le(
        20
    ) & records.crossing_only_candidate_distance_m.gt(20)
    records["hybrid_candidate_lost_at_site"] = (
        records.hybrid_candidate_distance_m.le(20) & ~records.hybrid_one_to_one_site_matched
    )
    records["hybrid_geometry_outcome"] = np.select(
        [
            records.hybrid_candidate_distance_m.gt(20),
            records.hybrid_site_distance_m.gt(20),
            ~records.hybrid_one_to_one_site_matched,
        ],
        ["no_candidate20m", "representative_shift_beyond20m", "one_to_one_site_capacity_loss"],
        default="matched",
    )
    records.to_csv(out / "inventory_coverage_diagnostics.csv", index=False)
    fold_rows = []
    for method in ("historical", "crossing_only", "hybrid"):
        sites = gpd.read_parquet(out / f"geometry_sites_{method}.parquet")
        for fold in range(3):
            selected = sites.loc[sites.route_id.map(fold_for).eq(fold)]
            real = truth.loc[truth.fold.eq(fold)]
            fold_rows.append(
                dict(
                    method=method,
                    fold=fold,
                    matched=len(
                        match_one_to_one(selected, np.c_[real.geometry.x, real.geometry.y])
                    ),
                    official_culverts=len(real),
                )
            )
    pd.DataFrame(fold_rows).to_csv(out / "score_free_fold_coverage.csv", index=False)
    return records


def report(out, replay, records):
    metrics = pd.read_csv(out / "ranking_metrics.csv")
    tops = (
        metrics.loc[metrics.k.isin([25, 50, 100, 250])]
        .groupby(["cohort", "policy", "k"])[["matched", "actual_denominator", "official_culverts"]]
        .sum()
        .reset_index()
    )
    tops["site_recall"] = tops.matched / tops.official_culverts
    tops["inventory_match_fraction"] = tops.matched / tops.actual_denominator
    tops.to_csv(out / "topk_summary.csv", index=False)
    coverage_table = pd.read_csv(out / "candidate_coverage.csv")
    geometry = coverage_table.query("scope=='eligible705'")[
        [
            "method",
            "candidates",
            "physical_sites",
            "candidate_matches_one_to_one",
            "consolidated_matches_one_to_one",
            "matches_lost_consolidation",
            "sites_per_road_km",
        ]
    ]
    score_diagnostics = pd.read_csv(out / "score_ordered_coverage.csv")
    road_lengths = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    fold_lengths = road_lengths.assign(km=road_lengths.length / 1000).groupby("fold").km.sum()
    correct_km = score_diagnostics.fold.map(fold_lengths)
    if not np.allclose(score_diagnostics.road_km, correct_km):
        save_json(
            out / "diagnostic_corrections.json",
            dict(
                fields=["road_km", "sites_per_road_km"],
                reason="Per-fold site-rate diagnostics must divide by that fold's road length, not whole-network length. No candidate, representative, score or match count changed.",
                old_road_km=sorted(score_diagnostics.road_km.unique().tolist()),
                correct_fold_km=fold_lengths.to_dict(),
            ),
        )
        score_diagnostics["road_km"] = correct_km
        score_diagnostics["sites_per_road_km"] = score_diagnostics.physical_sites / correct_km
        score_diagnostics.to_csv(out / "score_ordered_coverage.csv", index=False)
    scored = (
        score_diagnostics.groupby(["method", "phase"])[
            [
                "candidate_matches_one_to_one",
                "consolidated_matches_one_to_one",
                "physical_sites",
                "matches_lost_consolidation",
            ]
        ]
        .sum()
        .reset_index()
    )
    scored.to_csv(out / "score_ordered_coverage_summary.csv", index=False)
    at250 = metrics.loc[
        metrics.k.eq(250),
        [
            "cohort",
            "fold",
            "policy",
            "matched",
            "official_culverts",
            "largest_route",
            "largest_route_share",
        ],
    ]
    at250.to_csv(out / "route_concentration.csv", index=False)
    # Raw before1m overlap is a separate stage, not silently substituted for the
    # merged candidate count reported in the primary comparison.
    raw = gpd.read_parquet(out / "all_raw_proposals.parquet")
    truth = inventory()
    eligible = truth.loc[truth.fold.ge(0)]
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    raw_row = coverage_row(
        "hybrid",
        "raw_before1m",
        raw,
        cluster_sites(raw),
        eligible,
        float(roads.length.sum() / 1000),
        "eligible705",
    )
    save_json(out / "raw_overlap_stage.json", raw_row)
    regenerated = gpd.read_parquet(out / "historical_proposals.parquet")
    saved = gpd.read_file(ROOT / "data/interim/actual_ulster_candidates_with_route_samples.gpkg")
    d = cKDTree(np.c_[saved.geometry.x, saved.geometry.y]).query(
        np.c_[regenerated.geometry.x, regenerated.geometry.y]
    )[0]
    regenerated.loc[d > 0.001].groupby("route_id").size().rename(
        "fresh_positions_not_in_archival_output"
    ).to_csv(out / "coordinate_difference_by_route.csv")
    profiles = pd.read_csv(out / "road_profile_parity.csv")
    parity = pd.read_csv(out / "feature_parity_48.csv")
    path = out / "beaver_hybrid.json"
    if not path.exists():
        save_json(path, analyze(42.07647, -74.207172, 1500, "dynamic_route_z"))
    demo = json.loads(path.read_text())
    source = pd.read_csv(out / "candidate_source_contributions.csv")
    recovery = int(records.hybrid_recovers_crossing_miss.sum())
    lost = int(records.hybrid_candidate_lost_at_site.sum())
    content = f"""# High-recall hybrid site discovery — fixed staging evaluation

2026-10-15. Recommendations require field verification. **No production changes, deployment, model search, new model fits or promotion.**

## Decision in brief

Hybrid generation recovers candidate recall but **does not improve fixed-budget ranking**. It covers700/705 eligible inventory records versus63/705 for crossing-only proposals, restoring{recovery} previously uncovered records. Score-free20m physical consolidation retains653 one-to-one matches, losing47 candidate matches; score-ordered hybrid consolidation retains657, losing43. More candidate coverage is not automatically a more useful inspection queue.

AtTop100 per held-out group, hybrid RF ROUTE-Z matches13 versus13 for the historical same-road rerun and29 for crossing-only. AtTop250 per group it matches31 versus34 historical and59 crossing-only. Raw RF hybrid matches6/300 and17/750 respectively. Do not select a post-hoc threshold, evidence boost, road subset, model or ranking parameter to improve these exposed outcomes. Keep hybrid as a **research diagnostic**, not a field deployment.

## Phase1 — reconcile exact archival generation and portable geometry

Recovered code: `generate_candidates`, `generate_road_route_candidates`, `merge_candidate_layers` in `src/culvert_ai/candidates.py`, and the explicit commands in `scripts/run_actual_ulster_census_pipeline.sh`. Original geometry recipe:10m half-offset road sampling with zero lateral offsets, mapped exact crossings and<=20m nearest approaches, mapped20m deduplication then5m source-aware merge. Original road selection included numbered roads and additional named routes selected from field-report metadata. No official culvert coordinates generate, move or select new proposals here.

Exact replay of the preserved upstream proposal layers: {json.dumps(replay["inputs"])}. Output{replay["replayed"]} versus archived{replay["saved"]}; exact candidate-ID/geometry replay={replay["exact_id_and_geometry_replay"]}. See `archival_replay.json`. Cached proposals are label-free geometry, not official culvert points. The source files and pure generator code are preserved. Portable generation accepts every supplied road, including opaque/unseen nonnumbered IDs; regression tests verify this arithmetic and deterministic behavior. That is software portability, **not validated accuracy on a new region**.

The earlier51,417 score pool is a selection from the archived80,216 generated candidates, not the definition of a generic candidate generator. All51,417 native geometries were recovered and their700/714 raw coverage reproduced (`generator_reconciliation.json`), without reinterpreting coordinates or transferring consolidated653 to a different pipeline. The archived653 benchmark used different scores/representatives; the same numeric653 in today's score-free clustering is a newly measured result, not an assumed invariant.

Primary comparisons use exactly121 complete TIGER road segments,22 geometric corridor groups and the same inventory assignment from the prior audit. These roads were recovered from interval metadata, not selected by inventory distance. Sum of supplied segment lengths={float(roads.length.sum() / 1000):.6f}km; dissolved-line length={float(roads.geometry.union_all().length / 1000):.6f}km. Counts/km use the stated segment sum, which can count concurrent road identities separately.

Fresh historical recipe on that network produces79,063 proposals versus79,045 archived proposals attributed to those road IDs. Only74,592 fresh positions agree within1mm with that selected archive; most differences are on the concurrent44/55 corridor (4,412 on44). The archived merge included global county mapped contacts, two overlapping road-sampling layers and their original ordering. Sorting the restricted supplied-road population and retaining complete concurrent identities changes which nearby coordinates survive. We do **not** claim the portable supplied-road function recreates the exact historical51,417-ID selection on arbitrary roads. Exact archive replay and controlled same-network regeneration are distinct, both measured. See `coordinate_difference_by_route.csv`.

Current main inference uses75m samples with37.5m first offset and20m score-ordered suppression; these are not the historical10m/5m recipe. Prior strict staging uses actual mapped and DEM-network intersections with>=30° local transverse angle and two-sided drainage within60m. Its dense probes were internal only. Today's primary crossing-only control regenerates exactly the same2,034 crossing IDs as prior staging; current dense10m probes exactly match all83,478 prior probe IDs/geometries within1micrometre.

## Phase2 — high-recall hybrid and provenance

Three inputs:83,478 corridor probes;202 historical mapped crossing/approach proposals;2,034 strict mapped/DEM intersections. No DEM intersection is required for eligibility. Raw union has85,714 proposals;1m bounded same-road/part overlap grouping retains84,433 candidates. `raw_overlap_stage.json` records raw versus merged coverage; overlap keeps700 one-to-one candidate matches. No official coordinate is injected and no selected point is moved.

Evidence types are descriptive, not a scoring feature or a boost:

- **Crossing-supported:** a non-collinear mapped exact intersection or strict mapped/DEM crossing. Historical mapped exact intersections are distinguishable in provenance from strict bilateral/angle-checked events. Legacy near-zero-angle/collinear overlap centroids remain eligible as stream support instead, never confirmed crossings; this descriptor safeguard changes none of the85,714 evaluated evidence labels and never changes scores or clustering.
- **Stream-supported:** an existing mapped nearest-approach candidate or mapped-stream distance<=20m. This can be a parallel stream; it is not claimed as crossing evidence.
- **Terrain-supported:** an available v2 road depression>=0.15m and positive pre-existing low-point score, without stronger crossing/stream evidence. This is a measured terrain cue, not proof of a pipe or connectivity.
- **Road-corridor fallback:** no stronger available evidence; still eligible. It is an unverified corridor search proposal, not a detected drainage structure.

Fallback coverage is not discarded merely to make the map look more physical. `member_candidate_ids`, `proposal_sources`, member counts, representative evidence and aggregated site evidence preserve provenance. A representative can be a corridor point whose20m members include crossing evidence; the UI distinguishes these facts. Source indicators are **not** added to the48 model inputs.

Spatial clustering is bounded around an actual proposal, not transitive single-link chaining: same road ID/part, Euclidean distance<=20m **and** chainage difference<=20m. Distinct nonempty channel identities remain separate. Parallel roads and hairpin recrossings are protected by topology. A site is a proposed20m inspection footprint, not a confirmed physical structure. Conservative topology can leave duplicate nearby real-world sites across fragmented/concurrent road IDs; counts and one-to-one matching reveal rather than hide that cost.

Overlap representatives use deterministic candidate-ID order, independent of labels and scores. Final ranked representatives use only fixed rawRF or ROUTE-Z priority, then raw score and ID, without preferring all crossings. Original representative geometry and score are preserved. Raw/RZ start from identical merged candidate IDs, features and scores.

## Phase3 —48-feature compatibility fails for the frozen model

The ordered48 names are retained, but **name equality does not establish feature parity**. At3,626 identical-coordinate shared candidate IDs, {int(parity.differing_values.gt(0).sum())}/48 fields differ numerically or in missingness. `feature_parity_48.csv` contains every field, paired count, equality count, missingness and numeric deltas. All paired geometry distances are0; this is not a candidate-ID/coordinate join error.

Important finding beyond the profile-spacing bug: archived base terrain was sampled from `data/raw/dem.tif`, EPSG:4269,1arcsecond pixels, while v2 uses `dem_projected.tif`, EPSG:32618,10m pixels. Fixed pixel windows therefore cover different physical areas and have different nodata availability. This affects slope, relief, roughness, TPI and derived terrain composites at the same coordinate. `original_raster_feature_replay.csv` independently resamples the original raster and records equality for15 non-batch-normalized terrain fields on the shared coordinates. Repaired profiles, training-only composite references, density context and current crossing associations also differ. `feature_compatibility_supplement.json` records the exact raster metadata and causes.

On83,478 dense probes, the existing original profile formula and v2 independent10m measurement yield identical raw dip values on all82,998 finite pairs. However,480 v2 profiles correctly remain unavailable at missing/end shoulders rather than becoming0. The positive-dip percentile definition differs on897 finite rows (v2 upper empiricalCDF versus original average tied ranks); low-point score differs on489. Thus **raw finite depression-depth parity is valid, but complete three-feature semantic parity is not**.

{table(profiles)}

The inherited v2 `hydrology_edge_risk` also uses raw Whitebox edge-origin path values/maxima, which can exceed1, unlike the archived binary quality flag. It must not be described as a calibrated risk/probability or silently binarized for an estimator trained on v2 values. A future correction would need its own explicitly compatible contract/evaluation. No such change or extra model fit is smuggled into this comparison.

The original frozen RF is therefore **not scored with these v2 inputs**. We reuse only the three previously fitted fixed800-tree **v2 experimental RF evaluation copies**, with their training medians/contracts, held-out road folds and100m road purges. No additional training, model search, source labels, training rows or weights changed. Every shared probe/event score reproduces the preceding v2 run within1e-12. `reused_model_hashes.json` records the artifacts. Live staging explicitly loads the prior final **experimental v2 RF**, never substitutes frozen v1, and does not read inventory. Live local normalization depends on analyzed road extent; no independent deployment validation is claimed.

## Phase4 — same-network candidate and physical-site coverage

All distances are native EPSG:32618 metres. Official EPSG:26918 coordinates are transformed, not relabeled. Matching is deterministic maximum-cardinality one-to-one within20m. Both candidate one-to-one capacity and any-nearby coverage are included; they are not interchangeable with representatives or accuracy.

{table(geometry.round(4))}

Table denominator:705 in-footprint official records. `candidate_coverage.csv` also reports full714 context; counts happen to be identical because nine outside-footprint records gain no candidates. No missing/outside records were silently declared covered. `coverage_by_route.csv` reports all22 corridors, including those with zero inventory; geometric assignment and declared official route remain separate.

Score-free hybrid consolidation loses{lost} one-to-one matches;657 survive under either score-ordered policy instead of653 under ID-order geometry anchors. The difference illustrates how representatives affect coverage. Score ordering does not restore all700 candidate matches. Conservative topology leaves about27,250 hybrid sites across the folds; a group can contain multiple nearby raw search proposals without guaranteeing there is a culvert there.

The653 score-free hybrid count is also independently reproduced by within-fold matching (202+160+291), rather than assuming pooled matching and held-out matching are interchangeable. `score_free_fold_coverage.csv` records each control. Of the47 score-free lost matches,36 move beyond20m of surviving representatives and11 are one-to-one site capacity losses;14 inventory records have no candidate at20m (nine outside-footprint and five eligible).

{table(scored)}

`inventory_coverage_diagnostics.csv` lists every official record with source row, group, candidate/site distances, one-to-one match status,637 recovered crossing misses and consolidation losses. This file is research-only, never served to the map or model. `candidate_source_contributions.csv` reports raw source/evidence coverage and coverage unique to each subgroup, not additive attribution. In particular corridor probes provide636 unique records not covered by all other source kinds together; physical crossings do not expand the already dense700-record union in this network.

{table(source)}

## Phase5 — fixed ranking on identical held-out populations

Prior3 fixed corridor folds and exact prior training manifests/purges are reused. All714 labels have previous development exposure; the geographic holds are **retrospective**, not a new independent test. Each row below sums three separate TopK queues, so K25 means75 total slots, K100 means300, K250 means750. Different fold-model scores are never globally pooled. Inventory-unmatched recommendations are not field-confirmed false positives. No parameter, candidate class boost or model was selected from these outcomes.

ROUTE-Z reuses the preceding v2 evaluation's default `RouteZConfig`: at least8 candidates for mean/population-standard-deviation normalization, at least4 for median/IQR fallback, otherwise neutral0 with raw-score/ID ties. Frozen v1 global OOF fallback statistics are not transferred to the different v2 score distribution. These fixed defaults, not a tuned alternative, are used for all three candidate cohorts and the new map.

{table(tops.round(6))}

Site Recall@250 is matched count/705 for these disjoint groups: raw hybrid17/705 and ROUTE-Z hybrid31/705. Exact per-fold fractions/denominators are in `ranking_metrics.csv`. Full-list counts, candidate counts, distinct proposed site counts and budget fractions are retained. `ranking_by_route.csv` contains per-road queues; `global_budget_route_allocation.csv` separately records which roads receive no slots in the actual global group budgets—per-road reranking must not be mistaken for global allocation.

{table(at250.round(4))}

Hybrid rawRF Top250 concentration reaches72.4% on route87 in fold2; ROUTE-Z reduces that to50.8% but does not eliminate concentration. RawRF allocates noTop250 slots to road52 despite149 eligible inventory records, or32A despite one. ROUTE-Z allocates noTop250 slots to375 despite20 eligible records. These are observed omissions, not a reason to impose quotas or tune on outer outcomes.

Crossing-only still wins small budgets but has a63-record ceiling. Hybrid improves broad candidate recall, not precision-like inventory-match fractions. ROUTE-Z versus raw is a fixed comparison, not evidence of a globally validated champion. A ranking policy trained/selected on inner folds could be studied later, but none was invented or tuned here.

## Phase6 — separate staging map/API

`scripts/serve_hybrid_staging.py`, `staging/hybrid_service.py`, `staging/hybrid_sites.py` and `staging/hybrid_web/` implement an isolated loopback service on8095. Main8093 and crossing-only8094 implementations remain unchanged. The new map defaults toTop25 proposed consolidated sites, with matching list/export limits. Optional diagnostics: raw candidates, mapped+derived networks, actual road/drainage intersections, all consolidated representatives. Large diagnostic point sets use one SVG path, not thousands of clickable intermediate markers.

Green/blue/ochre/grey distinguish crossing, stream, terrain and corridor fallback evidence. No unverified point is labeled a confirmed culvert. Details show representative versus aggregated evidence, RF score, policy priority, model version, source membership, measured terrain/catchment values and unavailable values explicitly. Supported-area recovery and manual coordinates remain available; GPS needs user permission and was not automatically granted.

Representative Beaver Kill analysis,42.07647,-74.207172,1500m, fixedROUTE-Z: {demo["diagnostics"]["raw_candidates"]} raw proposals, {demo["diagnostics"]["merged_candidates"]} merged candidates, {demo["site_count"]} proposed physical sites. Default25 shown, not all road probes. Local evidence counts: {json.dumps(demo["diagnostics"]["evidence_counts"])}. These are actual GIS/model outputs, not independently validated field findings. `beaver_hybrid.json` is a label-free traceable payload. Browser/API checks and screenshot are recorded separately in `VERIFICATION.md`.

## Phase7 — final answers and readiness

1. **Historical reproducibility:** exact archive merge replay={replay["exact_id_and_geometry_replay"]}; portable10m/crossing/approach geometry works on arbitrary supplied road IDs. Historical route selection/51417 score subset is not a universal generator. Restricted same-network and exact archival geometries are explicitly reconciled.
2. **Hybrid recovery:** yes,637 inventory locations missed by crossing-only receive a candidate. They are inventory matches, not new field confirmations.
3. **Consolidation retention:** most, not all:653 score-free or657 fixed-score-ordered one-to-one matches, versus700 raw candidate matches. Losses47 or43 remain and are individually audited.
4. **Ranking improvement:** no. Hybrid ROUTE-Z13 at100/group ties historical13 and loses31 versus34 at250/group; both lose to crossing-only29/59. Raw hybrid6/17 also does not improve historical7/18. No promotion.
5. **Profile parity:** raw finite dips match on dense10m reference, but missingness, tied ranks, score semantics and other raster/context features prevent frozenRF compatibility. Reused v2 copies are the controlled alternative.
6. **Field-review cost:** roughly31.6 proposed sites per supplied-road km score-free, and32.6 score-ordered, versus1.26 crossing-only. This is about25–26times as many full-list sites for only about7.8times the consolidated inventory matches. There is no measured crew-time acceptance threshold, so operational acceptability is unverified. Poor budget matches make the all-sites queue unsuitable for routine field dispatch;Top25 merely limits display, not establishes confidence.
7. **Next evidence:** ranking and source-quality bottlenecks remain. Independent verified positives/negatives on isolated roads, better registered small-drainage/ditch/structure evidence, and a single documented raster/profile contract are needed before field claims. No further broad model search, external data acquisition or deployment was performed.

Recommendations require field verification.

## Reproduce and integrity

Use the existing compatible local environment and preserved assets. All291 protected files are SHA256-checked unchanged before/after evaluation. No original model, inventory, training/ground truth, production code, older report or deployment was modified. V2 source/model reuse is explicit; no fits occurred in this task.

```sh
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/evaluate_hybrid_sites.py --output /private/tmp/culvert_hybrid_reproduction
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/report_hybrid_sites.py --input /private/tmp/culvert_hybrid_reproduction
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider
node --test scripts/test_hydrology_web.mjs scripts/test_hybrid_web.mjs
node scripts/test_product_ui.js
node scripts/verify_security.js
npm run build
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/serve_hybrid_staging.py --port 8095
```

Output directories must be new. Reporting adds deterministic summaries/replay diagnostics and a real local demo, but does not alter evaluation policy or refit any model. LivePOST `/api/analysis/nearby` accepts `{{"location":{{"latitude":42.07647,"longitude":-74.207172}},"radius_m":1500,"policy":"dynamic_route_z"}}`; choose `raw_rf` for the other fixed policy. Local artifacts are under `reports/experiments/high_recall_hybrid_20261015/`, ignored by Git. `code_manifest.json` and nonexecutable source snapshots record exact changes; `input_integrity.json` records protected inputs.
"""
    (out / "REPORT.md").write_text(content)
    paths = [
        "staging/hybrid_sites.py",
        "staging/hybrid_service.py",
        "staging/hybrid_web/index.html",
        "staging/hybrid_web/app.js",
        "scripts/evaluate_hybrid_sites.py",
        "scripts/report_hybrid_sites.py",
        "scripts/serve_hybrid_staging.py",
        "scripts/test_hybrid_web.mjs",
        "tests/test_hybrid_sites.py",
    ]
    manifest = {}
    for name in paths:
        manifest[name] = sha256_file(ROOT / name)
        target = out / "code_snapshot" / (name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save_json(out / "code_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    before = protected()
    replay = archival_replay(args.input)
    raster_audit(args.input)
    records = per_record(args.input)
    report(args.input, replay, records)
    assert before == protected()
    save_json(
        args.input / "input_integrity.json",
        dict(protected=291, unchanged=True, hashes=before, model_fits=0, production_changed=False),
    )
    print(f"Report saved: {args.input / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
