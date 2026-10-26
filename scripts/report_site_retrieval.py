"""Publish the completed fixed retrieval audit; never select a threshold/model."""

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hydrology_sites import protected, save_json
from scripts.evaluate_site_retrieval import OUT, PREVIOUS_FEATURE
from scripts.validate_density_controls import HYBRID, PRIOR, xy
from scripts.report_density_controls import table

SOURCES = [
    "README.md",
    "staging/retrieval_sites.py",
    "staging/feature_service.py",
    "staging/feature_web/index.html",
    "staging/feature_web/app.js",
    "staging/feature_web/core.js",
    "staging/feature_web/ruler.js",
    "staging/feature_web/styles.css",
    "scripts/serve_feature_staging.py",
    "scripts/evaluate_site_retrieval.py",
    "scripts/report_site_retrieval.py",
    "scripts/plot_site_retrieval.py",
    "tests/test_retrieval_sites.py",
    "scripts/test_feature_web.mjs",
    "scripts/test_ruler_adapter.mjs",
    "staging/FEATURE_README.md",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT)
    out = parser.parse_args().input
    assert (out / "COMPUTATIONS_COMPLETE.json").exists()
    protected()
    guards = json.loads((out / "protected_inputs.json").read_text())
    assert all(sha256_file(ROOT / p) == h for p, h in guards.items())
    assert (out / "grouping_code_as_run.txt").read_bytes() == (
        ROOT / "staging/retrieval_sites.py"
    ).read_bytes()
    generation = pd.read_csv(out / "candidate_generation.csv")
    grouping = pd.read_csv(out / "grouping_sensitivity.csv")
    ranks = pd.read_csv(out / "ranking_combined.csv")
    waterfall = pd.read_csv(out / "recall_waterfall.csv")
    counts = pd.read_csv(out / "candidate_counts_per_inventory.csv")
    route = pd.read_csv(out / "ranking_by_route.csv")
    concentration = (
        route.loc[route.k.eq(250)]
        .sort_values("slot_share", ascending=False)
        .groupby(["method", "policy", "fold"])
        .head(1)
    )
    concentration.to_csv(out / "top250_largest_route_shares.csv", index=False)
    summaries = []
    for method, part in counts.loc[counts.fold.ge(0)].groupby("method"):
        values = part.candidates_within20
        summaries.append(
            dict(
                method=method,
                zero=int(values.eq(0).sum()),
                one=int(values.eq(1).sum()),
                two_or_more=int(values.ge(2).sum()),
                median=float(values.median()),
                p95=float(values.quantile(0.95)),
                maximum=int(values.max()),
            )
        )
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(out / "candidate_multiplicity_summary.csv", index=False)
    at20 = generation.loc[generation.tolerance_m.eq(20)]
    groups20 = grouping.loc[grouping.tolerance_m.eq(20)]
    groupcols = [
        "method",
        "grouping_distance_m",
        "candidates",
        "matches_one_to_one",
        "covered_any",
        "cross_road_groups",
        "groups_spanning_over_2radius",
    ]
    rankcols = [
        "method",
        "policy",
        "k",
        "actual_slots",
        "observed",
        "inventory_match_fraction",
        "site_recall",
        "random_mean",
        "p025",
        "p975",
        "p_one_sided",
    ]
    new = groups20.loc[
        groups20.method.eq("evidence_identity") & groups20.grouping_distance_m.eq(20)
    ].iloc[0]
    feature = at20.loc[at20.method.eq("feature_only") & at20.stage.eq("candidates")].iloc[0]
    baseline = at20.loc[at20.method.eq("feature_only") & at20.stage.eq("legacy20_sites")].iloc[0]
    report = f"""# Physical-site retrieval, grouping and visible A/B measurement

Completed 2026-10-16. **Staging only. No production changes, training or promotion.**

## Outcome and practical decision

The requested first-marker/second-marker tool is now prominently available on the map as
**↔ Measure A → B**, in addition to the header control. Activate it, click the first
point, then the second. Either pin can be dragged; explicit A/B buttons let you choose
which endpoint to replace. The line and metres/feet update immediately. This operates
independently of the candidates, API, ranking and field records.

Evidence-aware grouping reduces **{int(feature.candidates):,} admitted proposals to
{int(new.candidates):,} conservative groups** (13.5% fewer raw representations), retaining
**{int(new.matches_one_to_one)}/705** maximum one-to-one matches within 20 m. The old
20 m policy yields **{int(baseline.matches_one_to_one)}/705 at {int(baseline.candidates):,} groups**.
Thus the new option recovers 16 net matches but requires 1,023 more review locations.
Matches per 100 sites decline from {baseline.matches_per_100_sites:.2f} to
{new.matches_per_100_sites:.2f}. This is a coverage/cost tradeoff, **not a uniformly
better retrieval policy**. No radius was selected by evaluation labels. Existing 20 m
remains the reversible baseline; the more conservative option is explicitly experimental.

Do not merge harder to make the map look cleaner: 40 m legacy NMS lowers full matches
to 266. Global spatial grouping also merges separate roads. Conversely, retaining every
road sample attains high proximity coverage mostly by density, not structure detection.
No evaluated configuration is a validated field-ready detector or a 70% accuracy tier.

![Measured grouping/coverage tradeoff](grouping-tradeoff.png)

Recommendations require field verification.

## 1. What caused the screenshot and apparent clustering?

The supplied screenshot's nearby panel, mile-radius selector, Recenter/Export controls
and dense lower-priority circles match the **older production-style map**, not the
8096 localization lab. Exact screenshot coordinates/payload were not supplied, so it
cannot be uniquely replayed or assigned a fabricated candidate count. The current
browser was opened to 8096 and verified separately.

Source trace (Graphify navigation, then direct source confirmation):

- `src/culvert_ai/candidates.py:generate_road_route_candidates` defaults to 75 m in
  the production path. Historical/staging wrappers use 10 m half-offset samples.
  These coexist; it is incorrect to call every current map a 75 m generator.
- `web/app.js:renderMapData` renders every returned site, independently of the
  inspection-list cap. It has no A/B ruler. Production is intentionally unchanged.
- `staging/feature_service.py` has separate raw evidence, admitted proposals,
  score-free sites and ranked representatives. Default 8096 draws 25 queue groups,
  not every internal sample. Raw view is explicitly diagnostic and reversible.
- Multiple mapped/DEM resolutions can describe the same channel; exact coincident
  samples/contact sources create real representation duplication. However, distinct
  roads, road parts, channels and local extrema can be genuinely different sites.
- No multiple live models are ensembled here. Raw RF and ROUTE-Z are alternative
  policies using the same existing compatible RF predictions, not separate detectors.
- Diagnostic layers may intentionally draw multiple representations. They remain
  independently optional; no visual marker-cluster algorithm hides actual sites.

The 121 whole road segments/22 corridors total 834.809416 segment-km; overlapping road
records make the dissolved network shorter (763.784577 km). Density denominators here
consistently use segment-km. EPSG:26918 inventory coordinates are transformed to native
EPSG:32618, not relabeled. Unique-coordinate counts below are exact projected pairs,
not rounded lat/lon and not automatically independent physical culverts.

![Spatial distribution of all raw evidence, admitted proposals and grouped sites](spatial-distribution.png)

## 2. Why site recall was 40.4%

The disjoint waterfall uses the same 20 m tolerance. Nearest-candidate coverage and
one-to-one capacity are explicitly separate; individual loss records are saved.

{table(waterfall)}

Among the 705 eligible records: raw candidates cover 700, feature admission covers 310,
legacy representatives cover 294, and maximum one-to-one assignment retrieves 285.
The major loss is **390 records rejected before grouping**. Grouping changes cannot
recover locations absent from the admitted population. Nine eligible proximity matches
cannot be assigned simultaneously because a group can represent only one inventory
record. These are geometric capacity conflicts, not proof of duplicate inventory.

`coverage_failure_audit.csv` records all 714 original record keys, nearest distances,
eligibility, old/new one-to-one assignments and loss categories. No record was deleted,
relabeled or treated as a verified negative. All inventory has previous development
exposure; this is retrospective validation, not a new independent test.

## 3. Candidate generation held separate from ranking

Same roads, half-offset spacing and native matching. Sampling-only controls hold
geometry factors constant. Feature-plus-sampling is a union of unchanged proposals
and samples, not a tuned new feature gate. It deliberately includes redundant sources
so the effects of later grouping are measurable. 75 m here is a controlled same-road
geometry baseline, not a claim to replay every live preprocessing/model detail.

{table(at20[["method", "stage", "candidates", "unique_coordinates", "density_per_km", "covered_any", "matches_one_to_one", "missed"]])}

10 m and 20 m both reach 700 one-to-one matches before grouping, while 20 m uses half
as many samples. 40 m reaches 635, and 75 m reaches 338. Denser sampling beyond 20 m
adds mainly redundant locations at this tolerance. Do not generalize to smaller
tolerances: complete 5/10/15/20/25/50 m results and nearest median/p90 and matched
localization error are in `candidate_generation.csv`.

![Controlled sampling-density comparison](sampling-density.png)

Combining features with sparse samples improves pre-group coverage (75 m: 338→495;
40 m: 635→663), but existing 20 m grouping loses some of those gains. The union does
not establish more correct structures. A cleaner consolidated map can hide a real
coverage reduction, so no visual-only winner was selected or promoted.

### Candidate multiplicity near official records (eligible 705, within 20 m)

{table(summaries)}

`density_by_road.csv` lists each actual road ID, corridor, length, total/unique counts
and density, including roads with zero proposals. `candidate_counts_per_inventory.csv`
preserves all 714 record-level counts and nearest distances. Nearby candidates are
possible representations, not automatically duplicates or separate confirmed culverts.

## 4. Grouping comparison, fixed before observing outcomes

{table(groups20[groupcols])}

All 5/10/15/20/25/50 m matching tolerances are provided for every grouping radius in
`grouping_sensitivity.csv`. Radii 5/10/15/20/30/40 m are sensitivity analyses, not a
threshold search with a label-selected winner. The new live alternative retains the
inherited 20 m upper bound and remains an explicit option.

New `staging/retrieval_sites.py` requires the same road **and** part plus an explicit
shared channel identity or numerically identical coordinate (1e-6 m tolerance).
Conflicting nonempty channels stay separate even if coincident. Every pair in a group
must meet spatial and chainage bounds, preventing DBSCAN-style chaining and winding
road shortcuts. Missing identity, distinct extrema and nearby ordinary road samples
stay separate. Road intersections do not justify cross-road merging.

The representative is an actual member-coordinate medoid with stable candidate-ID
ties, selected without scores or inventory. Its measurements and score are its own;
no centroid relocation, score-max substitution or evidence boosting. All IDs, sources,
channel identities, coordinates, measurements and available scores are retained in
`member_evidence`, with an explicit representative-selection explanation. Raw RF and
ROUTE-Z share identical identity groups/representatives and only order them differently.

Geodesic NMS and geodesic DBSCAN(min_samples=1) are **unsafe diagnostics**, not accepted
site policies: they merge different roads. DBSCAN can chain locations across distances
larger than the intended neighborhood. The last column is a conservative lower-bound
flag based on an axis extent exceeding twice the radius, not a full diameter census.
Geodesic edges use WGS84 ellipsoidal distance after transforming coordinates; the
small native candidate-search envelope is only an acceleration, not the distance test.

## 5. Distinct-group ranking at every requested inspection budget

Each fold is a separate queue from its saved whole-route-held-out v2 model, with the
existing 100 m training purge. We do **not** pool different fold-model scores into a
fake global queue. Thus K=250 below means up to 750 total distinct group slots across
three tests, not one countywide Top-250. `ranking_by_fold.csv` gives each queue.

{table(ranks[rankcols])}

`inventory_match_fraction` is inventory matches per actual distinct inspection-group
slot. **Field Precision@K is unavailable**: unmatched sites are unlabeled, not verified
false positives. `field_precision_at_k` is explicitly null. Site recall is relative to
705 eligible inventory records; no field-confirmed accuracy is inferred from proximity.

1,000 fixed-seed 2 km local-block permutations per method/fold/policy preserve the
representative geometry, population and rank-slot block composition. All 18,000 draws
are saved, with nominal exploratory p-values. Full 100/250/500/1000/2000 slot results
are reported, including short queues and unfavorable outcomes. No new score was
computed. New 20/40/75 m locations lack compatible saved scores and are **not ranked
by invented/interpolated scores**. Dense ranking uses the 85,512 cached 10 m anchors
and strict crossing points; the 202 additional legacy mapped contacts lack saved
per-candidate scores and are explicitly excluded from this ranking-only comparison.

Route allocations, zero-slot roads, matches and largest shares are in
`ranking_by_route.csv` and `top250_largest_route_shares.csv`. Group identity is still
a research hypothesis; unknown adjacent road samples cannot be guaranteed distinct
physical structures merely because a grouping algorithm left them separate.

## 6. Working staging and measurement controls

[Open the updated map](http://127.0.0.1:8096/).

- Purple **↔ Measure A → B** is always on the map, not buried in the sidebar.
  First click places A, second places B. Independent drag and explicit replacement,
  snap-to-visible-sites, metres/feet, Reset/Clear/Exit/Escape remain supported.
- The header's original Measure distance remains available. The two controls share
  the same ruler state. The ruler never moves/reranks candidate sites or saves notes.
- Grouping selector compares the unchanged legacy baseline with conservative evidence
  identity. The raw/grouped selector reveals all raw source candidates without
  modifying queue ranks or exporting them as recommendations.
- One primary marker is drawn for each displayed recommended group. Optional source
  layers remain diagnostic; enabling them can intentionally show overlapping evidence.
- Details expose source member evidence and representative selection. Map/list/export
  use the same ranked group IDs and exact coordinates; exports include grouping policy.
- Mobile and desktop have separate actual screenshots and interaction evidence in
  `VERIFICATION.md`. No production UI change was made to the older screenshot map.

## 7. Reproduce, files and limitations

```sh
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/serve_feature_staging.py
# For a new immutable study, first copy FIXED_METHOD.md into <new-directory>:
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/evaluate_site_retrieval.py --output <new-directory>
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/report_site_retrieval.py --input <new-directory>
# Plot with the bundled image Python, not a newly installed dependency:
/Users/Carli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/plot_site_retrieval.py --input <new-directory>
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider
node --test scripts/test_hydrology_web.mjs scripts/test_hybrid_web.mjs scripts/test_feature_web.mjs scripts/test_ruler_adapter.mjs
```

`code_manifest.json` and `code_snapshot/` preserve the exact current implementation.
The previous feature-localization report, snapshots and results remain immutable;
its source hashes describe that earlier version, not the updated live source. Input,
inventory and frozen-artifact hashes remain unchanged. Stop only port 8096 to roll back.
No production edits, model writes, commits, pushes or deployment occurred.

The objective of a smaller **and** no-worse-recall physical queue is not fully achieved.
Evidence identity reduces redundant representations relative to raw proposals and
preserves more coverage than aggressive suppression, but does not dominate the smaller
legacy queue in review cost. Independent structure identities, small/unmapped drainage,
measurement availability and admission losses remain limiting. Do not force a cleaner
map by merging unsupported neighbors or claim all close records are one structure.

Recommendations require field verification.
"""
    (out / "REPORT.md").write_text(report)
    frames = {
        "All raw sources": gpd.read_parquet(HYBRID / "all_raw_proposals.parquet"),
        "Feature proposals": gpd.read_parquet(PREVIOUS_FEATURE / "feature_proposals.parquet"),
        "Legacy 20 m groups": gpd.read_parquet(PREVIOUS_FEATURE / "scorefree_sites.parquet"),
        "Evidence identity groups": gpd.read_parquet(out / "identity_feature_sites.parquet"),
    }
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    save_json(
        out / "spatial_plot_data.json",
        dict(
            bounds=roads.total_bounds.tolist(),
            frames={name: xy(frame).tolist() for name, frame in frames.items()},
            roads=[
                list(part.coords)
                for geometry in roads.geometry
                for part in (
                    list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
                )
            ],
        ),
    )
    snapshot = out / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in SOURCES:
        (snapshot / (path.replace("/", "__") + ".txt")).write_bytes((ROOT / path).read_bytes())
    save_json(out / "code_manifest.json", {p: sha256_file(ROOT / p) for p in SOURCES})
    save_json(
        out / "decision.json",
        dict(
            default="legacy20 retained",
            alternative="evidence identity 20 m, explicit research option",
            dominated=False,
            coverage_improved_vs_legacy=True,
            review_count_increased=True,
            independent_validation=False,
            field_precision=None,
            production_promoted=False,
            site_matches_old=285,
            site_matches_identity=301,
            site_count_old=5241,
            site_count_identity=6264,
        ),
    )
    if (out / "VERIFICATION.md").exists():
        save_json(
            out / "COMPLETE.json",
            dict(
                completed="2026-10-16",
                model_fits=0,
                evaluation_prediction_calls=0,
                production_unchanged=True,
                protected_artifacts=len(protected()),
                original_inputs=len(guards),
                artifact_sha256={
                    str(p.relative_to(out)): sha256_file(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "COMPLETE.json"
                },
            ),
        )
    print("Report and exact source snapshots saved.")


if __name__ == "__main__":
    main()
