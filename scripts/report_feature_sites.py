"""Report fixed localization evaluation and source/feature provenance;no tuning."""

# ruff:noqa:E402
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_feature_sites import OUT
from scripts.report_density_controls import table
from scripts.evaluate_hydrology_sites import protected, save_json
from culvert_ai.flow_c_bundle import sha256_file

SOURCES = [
    "staging/feature_sites.py",
    "staging/feature_service.py",
    "staging/feature_web/index.html",
    "staging/feature_web/styles.css",
    "staging/feature_web/app.js",
    "staging/feature_web/core.js",
    "staging/feature_web/ruler.js",
    "staging/FEATURE_README.md",
    "scripts/serve_feature_staging.py",
    "scripts/evaluate_feature_sites.py",
    "scripts/report_feature_sites.py",
    "tests/test_feature_sites.py",
    "scripts/test_feature_web.mjs",
    "scripts/test_ruler_adapter.mjs",
    "scripts/validate_feature_spatial_null.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT)
    out = parser.parse_args().input
    assert (out / "COMPUTATIONS_COMPLETE.json").exists()
    protected()
    guards = json.loads((out / "input_hashes.json").read_text())
    assert all(sha256_file(ROOT / p) == digest for p, digest in guards.items())
    cover = pd.read_csv(out / "coverage.csv")
    cover = cover.loc[cover.scope.eq("eligible705")]
    density = pd.read_csv(out / "density_matched.csv")
    spatial = pd.read_csv(out / "spatial_null_summary.csv")
    ranks = pd.read_csv(out / "ranking_combined.csv")
    top = ranks.loc[ranks.k.eq(250)]
    all_methods = pd.read_csv(out / "ranking_comparison.csv")
    compare = all_methods.loc[
        all_methods.k.eq(250) & all_methods.control.eq("block2km"),
        ["method", "policy", "actual_slots", "observed", "random_mean", "lift", "p_one_sided"],
    ]
    parity = pd.read_csv(out / "feature_parity_48.csv")
    parity["frozen_v1_live_use"] = "disabled"
    parity["v1_vs_v2_status"] = parity.differing_values.map(
        lambda x: "Different semantics/values" if x else "Numerically reproduced"
    )
    parity.to_csv(out / "all48_status.csv", index=False)
    schema = json.loads((ROOT / "models/flow_c_frozen_v1.joblib.schema.json").read_text())
    contracts = []
    for ordinal, name in enumerate(schema["feature_columns"], 1):
        if "robust_route_normalized" in name:
            unit = "dimensionless robust route normalization of m/km² density"
        elif name.startswith("log1p_"):
            unit = "ln(1+numeric metres or m² in the named original convention)"
        elif "m_per_sqkm" in name:
            unit = "m/km²"
        elif name.endswith("degrees"):
            unit = "degrees"
        elif "count_" in name:
            unit = "count"
        elif name == "hydrology_edge_risk":
            unit = "v1 binary flag;v2 raw edge-path accumulation,not probability"
        elif name.endswith("_m") or name == "route_elevation_dip_60m":
            unit = "metres"
        elif "rank_pct" in name:
            unit = "dimensionless percentile fraction0–1"
        else:
            unit = "dimensionless proxy/composite,not a calibrated probability"
        contracts.append(
            dict(
                ordinal=ordinal,
                feature=name,
                dtype=schema["expected_dtypes"][name],
                units=unit,
                infinite_values="replace withNaN",
                imputation="saved training medians",
                scaling="none",
            )
        )
    pd.DataFrame(contracts).to_csv(out / "feature_contract_48.csv", index=False)
    routes = pd.read_csv(out / "coverage_by_route.csv", dtype={"route": str})
    no_sites = routes.loc[routes.method.eq("feature_scorefree_sites") & routes.candidates.eq(0)]
    scored = pd.read_csv(out / "scoreordered_coverage.csv")
    final = scored.groupby("policy")[["sites", "full_matches"]].sum().reset_index()
    sources = pd.read_csv(out / "source_diagnostics.csv")
    compatibility = json.loads((out / "compatibility.json").read_text())
    raw = top.loc[top.policy.eq("raw_rf") & top.control.eq("block2km")].iloc[0]
    rz = top.loc[top.policy.eq("dynamic_route_z") & top.control.eq("block2km")].iloc[0]
    decision = "B. PHYSICAL FEATURES HELP SOME LOCATIONS, BUT COVERAGE IS INSUFFICIENT."
    final_rows = "\n".join(
        f"| {r.policy} | {r.sites:,} | {r.full_matches} |" for r in final.itertuples()
    )
    report = f"""# Feature-driven localization and geographic ruler

Research run: 2026-10-15. Final verification: 2026-10-16.

**{decision}**

Recommendations require field verification. Staging only: **http://127.0.0.1:8096/**.
No production model/UI changes,model training,new model families,datasets or predictive feature columns. Existing8093/8094/8095maps and historical research artifacts remain unchanged.

## Outcome

This fixed policy produces **7,244 evidence-admitted locations** on the same 121 whole road segments and 22 corridors (834.809416 segment-km). It matches **302/705 one-to-one within 20 m**, versus 219 from exactly 7,244 uniform road points: **83 additional matches, 1.379×**. After score-free 20 m consolidation, coverage is **285/705 at 5,241 sites**; exactly 5,241 uniform points match 138. A 100-repeat uniform consolidated-site subset control averages 153.20 matches (central 95% subset range 131–174), still below 285. These are measured equal-budget localization gains, not credit for dense sampling or fewer visible markers.

However, nearest-proposal coverage falls from 700 to 310 known records, and one-to-one candidate capacity from 700 to 302. Score-free site recall is 285/705 = 40.43%. Only 43 raw RF and 52 ROUTE-Z matches occur across three separate Top-250 queues (750 slots); inventory-matched fractions are 5.73% and 6.93%, **not verified precision or accuracy**. Neither beats a 2 km local-block null on identical representative sites: raw 43 versus {raw.random_mean:.3f}, p={raw.p_one_sided:.4f}; ROUTE-Z 52 versus {rz.random_mean:.3f}, p={rz.p_one_sided:.4f}. Thus feature selection adds some geographic localization, while current ML does not demonstrate additional within-local-road ranking value. It is not a reliable countywide detector or field-ready inspection queue.

## 1. Marker-provenance audit

| Marker class | Coordinate origin | What ML does | New default visibility |
| --- | --- | --- | --- |
| Raw sampling anchor | Half-offset10m road/part chainage in `generate_road_route_candidates`,wrapped by `search_candidates`/`historical_candidates` | May score anchor;does not move it | Hidden,diagnostic only |
| Mapped stream crossing | True transverse,two-sided stream-road intersection in `detect_events` | Ranks location | Hidden source layer;eligible proposal |
| DEM crossing | Road/derived drainage line intersection,existing angle/bilateral checks | Ranks location | Hidden source layer;eligible proposal |
| Terrain proposal | Genuine two-sided local profile/flow extrema at an original measured anchor | Does not create or move physical extremum | Hidden proposal layer;eligible for final recommendations |
| Consolidated site | Original representative retained within20m same-road/part/chainage bounded grouping,separate known channels | RawRF/ROUTE-Z can select representative | Hidden all-site diagnostic |
| Ranked recommendation | Consolidated admitted location;no backfill | Orders queue with fixed uncalibrated policy | Only default visible point layer,up to25 |

Precise old appearance: `src/culvert_ai/candidates.py:generate_road_route_candidates` places fixed intervals; `staging/hybrid_service.py:analyze` concatenates probes/mapped/events,then `overlapping_proposals` and `rank_sites` admit **all** probes includingfallbacks; `staging/hybrid_web/app.js:render` draws those representatives. Production `web/app.js:renderMapData` loops every returned site,even when list filtering differs. These original files are read,not modified. The priorSVG staging colors classify descriptive evidence; production priority colors are redHigh(#cb513d),amberMedium(#d69435),greenLower(#578575). NewLeaflet ranking preserves those priority colors and rank cutoffs25/100,not calibrated confidence.

Site consolidation NEVER jitters,averages,repositions or replaces a representative score with a member maximum. `member_candidate_ids` and `member_localization_evidence` retain support while explanations identify the representative's actual measurements. A geometric proposal,crossing,local extremum or high rank is not confirmation of a structure.

## 2. Four explicit processing levels

1. Raw computational anchors are measurement locations,not detections.
2. Physical proposals pass the fixed evidence admission below;zero is legitimate.
3. Compatible experimental-v2RF scores proposed locations;frozenv1remains disabled.
4. Twenty-metre consolidated inspection recommendations use fixed rawRF/ROUTE-Z ordering. The display limit is a cap,not a target.No road quotas/no marker-count backfill.

The original high-recall comparison remains available as a separate mode and at8095. It intentionally retains weak road samples,clearly explained as limited physical support. The8096comparison uses the same old generator/scoring/20m policy with a60m measurement halo;it is not silently relabeled a new detector. Halo points support extrema only;none outside the selected radius become recommendations. Normalization context uses in-radius anchors and the unchangedv2training contract. Different live road extents can change normalization/ranking and are not equivalent to frozen whole-road held-out evaluation.

## 3. Fixed feature-driven admission policy

`staging/feature_sites.py:proposals` has no inventory,inspection-record,training-label or official-coordinate input. It admits:

- Existing strict mapped/DEM road-water crossings.
- A local road-depression maximum with existing dip>=0.15m,available road profile and existing low-point score>0.
- A local upstream contributing-area maximum>0 with existing negativeTPI9px>0(concavity).

Local extrema require complete finite measurements60m on BOTH sides along the SAME road ID/part,at least one strictly lower measurement on BOTH sides,and stable candidate-ID ties on plateaus. Constant fields,edge windows,missing data and isolated anchors do not create peaks. Original coordinates are preserved with `originated_at_sampling_anchor` explicit. No interpolation or re-spacing is introduced.

Sixty metres and0.15m are inherited from the existing20–60m road-profile definition/prior floor,not chosen from outer-fold inventory outcomes. Sign tests preserve physical zero/missing distinctions.No threshold sweep,training search,road quota or post-outcome policy change occurs. This is a fixed hypothesis evaluated retrospectively,not a learned optimal rule.

Compatible localRFscore peaks are measured as a diagnostic and included in provenance,**not sufficient alone for admission**. This separates physical localization from score ordering. Existing road/stream/flow/area/concavity/valley/profile measurements are used;no additional RF feature is added. Across-road elevation difference and a dedicated embankment-convergence measurement are unavailable in the existing compatible representation,so details explicitly say unavailable.No unsupported measurement is invented.

### Source diagnostics (overlapping sets,not additive)

{table(sources[["method", "candidates", "covered_any", "matches_one_to_one", "admitted_alone"]])}

## 4. Frozen-feature parity and preprocessing

The frozen800-tree48-field bundle is byte-identical and is never scored by this mode. Original schema/order and preprocessing are recovered from `.schema.json`/training implementation:float64 numeric coercion,nonfinite→NaN,saved full-training medians,no categorical transforms,no scaler. Original baseDEM isEPSG4269/1arcsecond;v2usesEPSG32618/10m. Original pixel windows therefore have different physical footprints.Metre road/stream distances and EPSG26918official coordinates are transformed toEPSG32618,not relabeled.

`legacy_terrain_at` is an isolated partial original extractor using the original raster and original terrain routines. Freshly replayed **15original fields at3,626identical coordinates**,all equal within1e-6includingmissingness. This repairs/reproduces that portion for research,**not full frozenRFcompatibility**. No partial legacy table is fed into either model. Original15fields are not substituted into v2inputs;doing that would break the v2training contract.

The complete same-coordinate audit still finds35/48v1-v2differences:profiles/shoulder missingness,positive-dip rank ties,training-versus-batch composite references,density context,crossing associations,DEMgrid footprints,and raw Whitebox edge-path values versus legacybinary quality flag. Full per-field equality counts andmissingness are in `all48_status.csv` and below. `legacy_terrain_recovery.csv` records fresh partial recovery. There is **no defensible exact48-field livev1extractor yet**,so it remains disabled.

For live research,`check_model` fails closed unless the saved artifact explicitly has the v2contract,experimental flag,exact ordered48names,800trees,and saved training medians.No frozen fallback.Live input extraction uses that unchangedv2contract.Held-out analysis instead uses only verified savedv2evaluation-copy predictions;zero new fit/predict calls in evaluation. Whole-road folds and100m purges match the prior manifests.

## 5. Same-road coverage and density controls

Matching:inclusive20m nativeEPSG32618,deterministic maximum-cardinality one-to-one between selected sites and official records.Nearest coverage is reported separately;one point cannot match multiple records. Inventory-unmatched sites are **unlabeled**,not verified false positives. Nine outside-footprint records remain in714denominator tables.

{table(cover[["method", "candidates", "density_per_km", "covered_any", "matches_one_to_one", "coverage_pct", "matches_per_100"]])}

| Consolidation variant | Sites | One-to-one full matches |
| --- | --- | --- |
| Score-free20m | 5,241 | 285 |
{final_rows}

Score-ordered matching is summed within the three held-out groups,whereas joint coverage matches all705together;junctions can produce minor capacity differences.Representative geometry and site counts can differ between score-free/score-ordered selection;do not mistake that for pure ranking performance.Candidate700→302loss398one-to-one;302→285score-free consolidation loss17.Nearby inventory700→310loss390 is a distinct metric.

### Density-matched localization curves

For each fixed candidate/site budget,100seeded uniform subsets of the feature population and corresponding uniform10m/consolidated-uniform population are compared.Fresh exact-budget uniform spacing is an additional non-clumpy control;no budget is chosen for favorable outcomes.Full source populations' measured budgets are included without using labels to select them.

{table(density)}

At5,000sites,feature subsets average272.61matches versus146.59uniform-site subsets and144fresh uniform points.This is physical-location enrichment,not an effect of simply proposing more points.The endpoint5,241site comparison285versus138fresh uniform and153.20subset mean gives the same qualitative result.Budget-specific phase sensitivity remains;these are single-region exposed data,not independent statistical proof of generalization.

### Stronger controls for proposal location, not just ranking

The following additional 1,000 random-phase road-placement trials per row preserve the exact budget globally, per actual road part, or per fixed 2 km road-part block. Road geometry defines all strata; no inventory point defines a boundary. Each stratum receives its original number of feature proposals/sites, with uniform spacing and an independently seeded phase. This controls route concentration and local proposal density before model ranking. Control points are proposed road locations, not confirmed structures, and are not re-clustered to change their exact budget. One-to-one matching still enforces capacity at close boundaries. These tests are separate from shuffling scores on fixed representative geometry.

{table(spatial[["stage", "control", "budget", "observed", "random_mean", "p025", "p975", "lift", "excess_matches", "p_one_sided"]])}

These localization gains must not be attributed to RF ranking. The rule uses physical measurements; pure ML peaks alone do not admit locations. All null outcomes/stratum counts are saved in `spatial_null_replicates.csv` and `spatial_null_strata.csv`. Small strata limit relocation freedom; inference remains retrospective and regional. The fixed policy was not changed after these results.

## 6. Held-out retrieval and randomization

Three separateTopKqueues are summed;Top250means750total slots,not a single countyTop250.Scores from different evaluation-copy models are never globally pooled.Candidates are selected without inventory first;then within each fixed fold the saved compatible scores apply rawRF/ROUTE-Z and existing20m consolidation.

Randomization freezes site IDs,coordinates,routes and counts:1,000seeded permutations per fold/policy/null,18,000total.County,exact route-slot and fixed roadID/part/2km-chainage block nulls are reported.Seed20261015with documented SHA256 streams in the evaluator.Each observedTopKis crosschecked using the existing SciPy maximum matcher;random prefixes use exact augmenting paths,not fixed binary prelabels.

### Every requestedTopK

{table(ranks.loc[ranks.control.eq("block2km"), ["policy", "k", "actual_slots", "observed", "inventory_match_fraction", "site_recall", "random_mean", "p025", "p975", "lift", "p_one_sided"]])}

### Top250 under all null controls

{table(top[["policy", "control", "observed", "random_mean", "p025", "p975", "lift", "excess_matches", "p_one_sided"]])}

### Existing same-road comparisons (local-block null)

{table(compare)}

Newfeature43/52versusoldhigh-recall17/31 is not by itself proof of improvedML;the proposal population changed.Crossing-only58/59still exceeds the new mixed-policy43/52at this budget,despite lower full-list recall.This is a recall/retrieval tradeoff,not permission to optimize policy on these outcomes.Neithernewrankingbeatslocalblocks;absolute gains largely arise from physical selection,not additionalMLdiscrimination.

Nominal p-values are exploratory on development-exposed roads and repeated budgets,not independent confirmatory significance.Central95%null ranges describe random outcomes,not deployment-accuracy confidence intervals. Tiny/immobile2km strata can constrain randomization;no stratum-specific significance is claimed.Source policy,window,floor andranking defaults remain fixed after seeing results.

`ranking_by_route.csv` lists every corridor for both policies/everyK,including zero-slot roads,known counts,full proposed-site population,slot allocation,matches and matched-distance means.`ranking_by_fold.csv` includes largest-road shares.Route-concentration and no-proposal-road diagnostics are not hidden.Zero-proposal corridors in this121-segment evaluation:{", ".join(no_sites.route.astype(str)) or "none"}.Live searches can still return zero (the currentBeaverKill window has11roads without a feature proposal).Nearest-inventory median/p90distances are in coverage tables;matched distances use deterministic maximum matching,not minimum-cost matching and not field-survey localization error.

## 7. Leaflet distance ruler and spacing

The toolbar's **Measure distance** activates firstclickA/secondclickB.Subsequent clicks replaceB.Two independently draggable,purple/blue lettered pins and a dashed line show continuously updated **Straight-line distance** using `map.distance` on geographiclat/lon ([Leaflet1.9.4reference](https://leafletjs.com/reference.html)).No pixel distances enter measurement.Earth-sphere/geodesic calculation follows Leaflet;displayed decimals do not imply survey-grade endpoint accuracy.

Meters/feet toggle uses exact0.3048m/ft.Reset starts a fresh active measurement;Clear removes endpoints;Exit/Escape stop measurement and disable pin dragging while preserving the last line until cleared.Snap is optional within14CSSscreenpixels of actually visible point layers;snapped endpoint outlines and source IDs make it explicit.Drag outside the radius detaches,without moving the underlying site.Marker clicks while measuring set endpoints instead of opening details.**Measure from this site** setsAand awaitsB.Arbitrary coordinate entry and1m/Shift10m arrow-key nudges support keyboard access.Measurement has no fetch/write/training path and does not mutate ranks,site coordinates,records or active analysis area.

Spacing is independently optional.Raw anchors and feature proposals have separate controls.Consecutive points are ordered by actualroadID/part and measured chainage,not geographic nearest neighbour.Segments follow the actual road substring;distances are native-metre chainage differences,not straight-line shortcuts or culvert-to-culvert travel distances.Gaps can reflect the visible search window,so displayed anchors are not a complete unbounded-road census.All line segments remain available;at close zoom persistent short labels appear only when the projected segment has enough room for that label,otherwise hover/tap still exposes its distance.No candidate or recommendation is removed for label decluttering.

The seven independent layers are rawanchors,mappedcrossings,DEMcrossings,featureproposals,consolidatedsites,rankedrecommendations,and drainage diagnostics.Only ranked recommendations are checked bydefault.Map/list/GeoJSON share the display cap;exports retain exactgeometry,IDs,ranks,scores,model/policy and localization provenance.No backfilling when few/zero sites pass.Local field notes are browser-session-only and exported separately;they do not write production inventory or training data.Saving is disabled during measurement.

## 8. Verification and screenshots

See `VERIFICATION.md` for executed browser boundaries,tests and remaining hardware limits.Screenshots are actual rendered app states,not mockups:

- `screenshots/main-map-desktop.jpg`:desktopfeaturemode/defaultpointlayers.
- `screenshots/distance-ruler-dragged-pins.jpg`:both geographicpins after independentmouse drags,feet conversion.
- `screenshots/raw-anchors-spacing.jpg`:explicitrawdiagnosticand roadspacing.
- `screenshots/feature-proposals.jpg`:admittedproposal layer.
- `screenshots/site-evidence.jpg`:actualsite measurements/provenance.
- `screenshots/mobile-ruler.jpg`:mobilebreakpoint after compact-control repair.

Full model/schema,selection/consolidation and ruler tests are reproducible from the commands below.Live compatible finalv2RFscores are research-only;the held-out table uses different savedfoldcopies and must not be treated as that live map's independent validation.Geolocation permission/real touch hardware are not silently claimed tested;permission failure and touch event handling are documented separately.

## 9. All48field parity status

These comparev1andv2atidenticalnativecoordinates.Allfieldsremain disabled forfrozenv1inthisnewmode;some numerical matches do not validate the complete representation.

{table(parity[["feature", "paired_rows", "differing_values", "missing_v1", "missing_v2", "v1_vs_v2_status"]])}

## 10. Reproduction,exact new files and rollback

```sh
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/serve_feature_staging.py
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/evaluate_feature_sites.py --output <new-directory>
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/validate_feature_spatial_null.py --input <new-directory>
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/report_feature_sites.py --input <new-directory>
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider
node --test scripts/test_hydrology_web.mjs scripts/test_hybrid_web.mjs scripts/test_feature_web.mjs scripts/test_ruler_adapter.mjs
node scripts/test_product_ui.js
node scripts/verify_security.js
```

All task source files are new;no existing production or staging file needs to be changed to enable this mode.Stop the8096process and continue using8093/8094/8095to roll back.No production configuration switch,commit,push or automatic promotion occurred.

{table(pd.DataFrame({"New file": SOURCES, "Purpose": ["Fixed label-free proposal policy/partial legacy extractor/spacing", "Read-only live compatible-v2service", "Leaflet map structure", "Responsive map/ruler styles", "Layer/detail/field-note integration", "Pure geographic/state/export helpers", "Independent ruler event adapter", "Staging setup and rollback", "Loopback8096static/API server", "Executed fixed held-out/density evaluation", "Reproducible report and provenance", "Scientific/backend regression tests", "Geographic/export/semantic tests", "Ruler adapter click/drag/touch/key tests", "Exact-budget road-part/block spatial controls"]}))}

Source snapshots and SHA256manifest record exact code;`proposal_code_as_run.txt` preserves the rule implementation used before metrics were observed.`input_hashes.json` and291-file protected guard verify original files/models unchanged.Individual random outcomes,all source/evidence selections,all48parity values and per-route data are saved,not only headline summaries.

## Deployment recommendation

**Research diagnostic only.Do not deploy or call this a validated high-precision detector.** Physicalfeatures provide measured equal-density enrichment on these exposedroads,but miss most inventory after filtering and do not establish reliable structure identification.TheRFadds no measurable local-block ranking benefit.Need independent field verification of both proposed and unproposed sites,stable feature contracts/measurement availability,and preregistered new roads before deployment.No70%accuracy,calibratedprobability,trueFP or new independent test is supported by this work.

Recommendations require field verification.
"""
    (out / "REPORT.md").write_text(report)
    save_json(
        out / "decision.json",
        dict(
            decision=decision,
            candidate_matches=302,
            eligible=705,
            equal_budget_uniform_candidate_matches=219,
            scorefree_site_matches=285,
            equal_budget_uniform_site_matches=138,
            frozen_compatible=False,
            independent=False,
            deployment_ready=False,
            compatible_original_terrain_fields=compatibility["original_terrain_fields_recovered"],
        ),
    )
    snapshot = out / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in SOURCES:
        (snapshot / (path.replace("/", "__") + ".txt")).write_bytes((ROOT / path).read_bytes())
    save_json(out / "code_manifest.json", {p: sha256_file(ROOT / p) for p in SOURCES})
    if (out / "VERIFICATION.md").exists():
        save_json(
            out / "COMPLETE.json",
            dict(
                completed="2026-10-16",
                decision=decision,
                deployment_ready=False,
                production_changed=False,
                model_fits=0,
                evaluation_prediction_calls=0,
                python_tests_passed=229,
                javascript_tests_passed=19,
                ranking_null_draws=18000,
                localization_null_draws=6000,
                protected_files_unchanged=291,
                additional_inputs_unchanged=len(guards),
                artifact_sha256={
                    str(p.relative_to(out)): sha256_file(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "COMPLETE.json"
                },
            ),
        )
    print(decision)


if __name__ == "__main__":
    main()
