"""Generate the research report from verified tables; no inference or fitting."""

import argparse
import csv
import json
from pathlib import Path


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output.resolve()
    assert (out / "verification.json").exists() and not (out / "COMPLETE.json").exists()
    metrics = read(out / "comparison.csv")
    population_rows = []
    for name in sorted({r["system"] for r in metrics}):
        full = next(
            r
            for r in metrics
            if r["system"] == name and r["budget"] == "all" and r["tolerance_m"] == "20"
        )
        baseline_only = name in ("BASELINE-RF", "BASELINE-HGB", "BASELINE-DIVERSITY")
        full_hybrid = name in ("FULL-HYBRID-RF", "FULL-HYBRID-HGB", "HYBRID-DIVERSITY")
        population_rows.append(
            dict(
                system=name,
                raw_measured_candidate_pool=236858 if baseline_only else 245373,
                physical_hypotheses_before_original_or_extra_grouping=17715
                if baseline_only
                else 33898
                if full_hybrid
                else int(full["sites"]),
                baseline_already_consolidated_before_extra_grouping=not (
                    baseline_only or full_hybrid
                ),
                final_recommended_site_hypotheses=int(full["sites"]),
                note="Hypotheses are not field-certified distinct culverts; selective counts start with 13065 baseline anchors plus new proposals",
            )
        )
    with (out / "candidate_population_accounting.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(population_rows[0]))
        writer.writeheader()
        writer.writerows(population_rows)

    def metric(name, budget, tolerance=20):
        return next(
            r
            for r in metrics
            if r["system"] == name
            and r["budget"] == str(budget)
            and r["tolerance_m"] == str(tolerance)
        )

    names = [
        ("BASELINE-RF", "Existing fixed-site baseline, raw RF"),
        ("FULL-HYBRID-RF", "Full hybrid, grouped, raw RF"),
        ("SELECTIVE-corroborated-RF", "Selective admission, raw RF"),
        ("SELECTIVE-GROUPED-corroborated-RF", "Selective + strict grouping, raw RF"),
        ("BASELINE-FIRST", "Baseline-first append control"),
        ("FIXED-PHYSICAL-0.1", "Physical support +0.1, soft diversity"),
        ("PHYSICAL-RERANK", "Development-selected soft diversity"),
        ("NESTED-SELECTIVE", "Nested admission/reranking selection"),
        ("SITE-MODEL-DESCRIPTIVE", "New site model, descriptive"),
        ("SITE-MODEL-CROSSFIT", "New site model, road/spatial cross-fit"),
        ("NESTED-SITE-MODEL", "New site model, nested population"),
    ]
    rows = []
    for name, label in names:
        full = metric(name, "all")
        rows.append(
            [
                label,
                full["sites"],
                full["matched"],
                *[metric(name, k)["matched"] for k in (100, 250, 500, 1000)],
            ]
        )
    main_table = table(["System", "Sites", "Full /847", "@100", "@250", "@500", "@1000"], rows)
    tolerance_table = table(
        ["System", "5 m", "10 m", "15 m", "20 m", "25 m", "50 m"],
        [
            [label, *[metric(name, "all", t)["matched"] for t in (5, 10, 15, 20, 25, 50)]]
            for name, label in names[:4]
        ],
    )
    changes_table = table(
        [
            "System",
            "Full added/lost",
            "@1000 added/lost",
            "Full assigned median (m)",
            "@1000 assigned median (m)",
        ],
        [
            [
                label,
                f"{int(float(metric(name, 'all')['added_vs_baseline20']))}/{int(float(metric(name, 'all')['lost_vs_baseline20']))}",
                f"{int(float(metric(name, 1000)['added_vs_baseline20']))}/{int(float(metric(name, 1000)['lost_vs_baseline20']))}",
                f"{float(metric(name, 'all')['matched_distance_median_m']):.2f}",
                f"{float(metric(name, 1000)['matched_distance_median_m']):.2f}",
            ]
            for name, label in names[:8]
        ],
    )
    sources = read(out / "source_counterfactuals.csv")
    source_table = table(
        ["Representative source", "Sites", "Standalone matches", "Matches lost if source omitted"],
        [
            [r["source"], r["sites"], r["standalone_matches"], r["marginal_capacity_matches"]]
            for r in sources
        ],
    )
    folds = read(out / "road_spatial_fold_results.csv")
    fold_table = table(
        ["System", "Fold 0 /174", "Fold 1 /360", "Fold 2 /313"],
        [
            [
                name,
                *[
                    next(
                        r["matched"]
                        for r in folds
                        if r["system"] == name and r["fold"] == str(fold) and r["budget"] == "1000"
                    )
                    for fold in range(3)
                ],
            ]
            for name in (
                "BASELINE-RF",
                "FULL-HYBRID-RF",
                "SELECTIVE-RF",
                "PHYSICAL-RERANK",
                "SITE-MODEL",
            )
        ],
    )
    road = read(out / "proper_existing_model_road_comparison.csv")
    road_table = table(
        [
            "Existing family",
            "Policy on proper held-out representatives",
            "Sites",
            "Full /845",
            "@1000",
        ],
        [
            [
                family,
                name,
                next(
                    r["sites"]
                    for r in road
                    if r["family"] == family and r["system"] == name and r["budget"] == "all"
                ),
                next(
                    r["matched"]
                    for r in road
                    if r["family"] == family and r["system"] == name and r["budget"] == "all"
                ),
                next(
                    r["matched"]
                    for r in road
                    if r["family"] == family and r["system"] == name and r["budget"] == "1000"
                ),
            ]
            for family in ("RF", "HGB")
            for name in ("BASELINE", "SELECTIVE", "SELECTIVE-DIVERSITY")
        ],
    )
    regional = read(out / "region_route_results.csv")
    region_rows = []
    for name in (
        "BASELINE-RF",
        "FULL-HYBRID-RF",
        "PHYSICAL-RERANK",
        "NESTED-SELECTIVE",
        "SITE-MODEL-CROSSFIT",
    ):
        for region in ("Region 6", "Region 8"):
            full = [
                r
                for r in regional
                if r["system"] == name and r["budget"] == "all" and r["Region"] == region
            ]
            head = [
                r
                for r in regional
                if r["system"] == name and r["budget"] == "1000" and r["Region"] == region
            ]
            region_rows.append(
                [
                    name,
                    region,
                    sum(int(r["records"]) for r in full),
                    sum(int(r["matched"]) for r in full),
                    sum(int(r["matched"]) for r in head),
                ]
            )
    region_table = table(
        ["System", "Region", "Records", "Full matched", "Global @1000 matched"], region_rows
    )
    verification = json.loads((out / "verification.json").read_text())
    selected = json.loads((out / "all_development_choice.json").read_text())
    groups = read(out / "grouping_accounting.csv")
    grouping_table = table(
        ["Rule", "Before sites/matches", "After sites/matches"],
        [
            [
                rule,
                *[
                    f"{next(r['hypotheses'] for r in groups if r['rule'] == rule and r['stage'] == stage)}/{next(r['matched20'] for r in groups if r['rule'] == rule and r['stage'] == stage)}"
                    for stage in ("before", "after")
                ],
            ]
            for rule in ("corroborated", "balanced", "drainage")
        ],
    )
    report = f"""# Targeted candidate admission and distinct-site ranking

Experiment started 2026-10-16; report completed 2026-10-17. Retrospective research
only. Existing frozen models, production assets, application defaults and
deployments are unchanged. Recommendations require field verification.

## Decision

**Keep the current default. No validated replacement was established.**

The reproduced raw-RF benchmark is **123 matches within 1,000 stops**. A selected
selective-admission/soft-diversity queue reaches **127/847** at the same budget and
**372/847** in its full list (17,083 recommendations). That is only +4 at the
inspection budget and +25 full-list, not a robust field improvement. Nested
geographic/corridor development selection falls to **99/847** at 1,000 stops.
The new physical-site model retrieves only **13/847** with cross-fitted scoring.
Do not promote the 127 result or report it as independent accuracy.

![Measured fixed-budget comparison](fixed_budget_retrieval.png)

## 1. Why +136 full-list matches but -21 at 1,000 stops?

Source observations: unchanged `Culvert_Locations_All_Teams_Summer_2026.xlsx`,
`Culvert Locations!A1:F960`; SHA256
`b74908b7238f8c5c93f890f6eeddb886d24a96bfb4ec36eb832b0ae8988e9013`.
All 959 original source rows are preserved, including blank culvert IDs.

The full grouped hybrid retains 346 baseline matches, adds 137 and loses one:
**483 versus 347**, net +136. At 1,000 stops it retains 71 old matches, gains 31,
and loses 52: **102 versus 123**, net -21. There are 603 different representatives
in the hybrid Top1,000; only 397 IDs are shared with the original Top1,000.

This is primarily a demonstrated **queue dilution/crowding and local redundancy
problem**, not a feature-schema exception or proof that unmatched sites are wrong:

- The saved hybrid introduces many similarly/highly scored nearby hypotheses.
  Baseline Top1,000 has 66 sites participating in a same-road/part pair within
  40 m in space AND chainage; hybrid has 650, with 524 such pairs versus 33.
- Retaining all original representatives in a controlled union does not fix
  raw ordering: it still matches 102 at 1,000, while full-list capacity rises to 484.
  The effect is therefore not mainly the hybrid's medoid replacement of old
  representatives. Medoid changes explain one net full-list loss, not 21 at Top1,000.
- On the SAME full hybrid population/scores, changing only to the declared soft
  same-road redundancy penalty raises102→116. This ordering intervention recovers
  14 of the21 deficit, without deleting a site or imposing a road quota. It does
  not prove that every nearby pair is one physical culvert.
- Road-name representation shrinks393→266 in the first1,000, while the largest
  single-road-name share barely changes7.4%→7.5%. Local crowding and geographic
  allocation are better supported explanations than one dominant road alone.

Scores differ by source but **distribution shift alone is not proven causal**.
Small D8 crossings have mean RF0.501, versus baseline0.493, despite similar
nearest-inventory median distances553 m versus545 m. No unmatched location becomes
a false positive: the workbook is not a complete survey of all candidate roads.
Baseline representative scores are reproduced unchanged when adding candidates.

![Per-source score and inventory-distance distributions](source_distributions.png)

`source_candidates_*.csv` retains every measured score/distance, and
`source_distributions.csv` gives quantiles. Sources below are exclusive by
representative-source precedence; original multi-source provenance remains in
the saved populations. Standalone contributions overlap and cannot be added.
Leave-one-source-out results quantify marginal one-to-one capacity, not precision.

{source_table}

There are **4,641 hybrid candidate IDs already represented as non-representative
members of original baseline groups**. These are documented representation overlap,
not4,641 verified duplicate pipes. Previous strict hybrid identity grouping removes
2,050 representations (33,898→31,848) and one match (484→483). Proximity-only pairs
are reported separately and never treated as confirmed duplicates.

## 2. All 458 admission losses

Audited each original record and every owned raw candidate within20 m against
the actual available profile, dip/low-point gate, flow/concavity gate, complete
60 m window, window maximum, two-sided strict peak and deterministic plateau tie.

- **296** have at least one nearby candidate passing a profile or flow value gate
  but are blocked by local-window/peak selection.
- **161** lack the required measured value support; relaxing only a local maximum
  rule cannot manufacture physical evidence for them.
- **1** has no usable nearby profile and no other admitted evidence.

Detailed overlapping rule reasons are in `admission_458_candidate_rules.csv`;
the exhaustive record partition is `admission_458_records.csv`. Missing data,
zero dip, stronger neighbours, incomplete shoulders and plateau ties are distinct.
This is not a claim that458 culverts are absent from the terrain, nor that a
nearby raw probe is itself a culvert detection. The other42 original misses and
112 unsupported observations remain in the prior diagnosis, unchanged.

## 3. Selective generation and physical identity

No new coordinates were generated from inventory. Every extra hypothesis already
exists in the saved label-free hybrid. Preserve every original baseline anchor,
its geometry, ID, member provenance and own RF/HGB scores. Exclude IDs already
represented in baseline membership before testing extra admission.

Three rules and18 rule/weight/diversity combinations were fixed in `METHOD.md`:
corroborated local extrema/flow or transverse D8>=0.1km²; balanced additionally
allows D8>=0.01km² and strong profile/valley depressions; drainage retains strong
D8 or simultaneous profile/flow extrema. There is **no universal mapped-stream
prerequisite**. Flow accumulation, upstream area, concavity, local valley depth,
profiles and transverse geometry are used where available. The10 m DEM cannot
verify small ditches, pipe openings or embankment geometry; those are not invented.

Same actual road AND part, shared explicit channel or coincident geometry, and
complete-link20 m spatial/chainage bounds are necessary for extra grouping.
Never merge two baseline anchors, bridge separate channels or move an anchor.
Unknown adjacent extrema stay separate. A regression caught missing channel IDs
being treated as a shared string: fixed **before final results**, with a new test
and a no-fit replay. All preliminary merged results were superseded.

{grouping_table}

**No additional identity merges are justified in these selective populations**
after excluding already represented member IDs. This negative grouping result
must not be hidden by distance-only thinning. Saved original baseline groups
already perform their historical20 m consolidation; close unknown new hypotheses
remain uncertain review locations, not certified distinct pipes.

All systems reuse236,858 original computational candidates; the hybrid source
pool adds8,515 small-D8 contacts (245,373 measured positions total). Baseline
admission has17,715 proposals before13,065 representatives. Selective counts here
are original13,065 already-consolidated anchors plus selected extras, not a fresh
claim that each raw sample is a physical site. See candidate accounting tables.

## 4. Equal budgets and localization

Same847 source rows, same157 successful fixed-cell ownerships, native EPSG:32618,
one-to-one maximum-cardinality matching<=20 m. No inventory snapping or tolerance
selection. Every site's original location is preserved. The app's local Top25
and dynamic ROUTE-Z default are NOT these pooled global raw-RF budgets.

{main_table}

The development-selected configuration is `{selected["rule"]}`, physical bonus
{selected["weight"]}, soft40 m same-road penalty{selected["penalty"]}. Its winning
setting uses **zero physical-score bonus**: call it selective admission plus soft
diversity, not evidence that adding physical bonus improved ranking. Nonzero
physical weights 0.1 and 0.25 yield 121 and 109 at 1,000. HGB raw ordering remains weak:
baseline 28, full hybrid 18, corroborated selective 25 at 1,000.

The baseline-first append control preserves123 at1,000 by construction and
retains extra full-list coverage. That is a reversible queue-protection control,
**not an improvement at the inspection budget**. Soft diversity does not force
equal routes and never removes a lower-ranked genuine hypothesis.

Full-list sensitivity (counts, denominator847; not alternate accuracy claims):

{tolerance_table}

Changes compare to baseline at the **same budget**. Record memberships can change
under equally maximum-cardinality assignments when nearby observations compete;
membership is deterministic, not proof of physical identity.

{changes_table}

Assigned-distance medians use different matched sets and can change with matching
tie allocation. They are not evidence of improved localization by themselves.
`paired_nearest_localization.csv` supplies same-record nearest changes. Anchored
selective populations retain every original geometry, so nearest-baseline
coverage cannot worsen; missed one-to-one capacity is still reported separately.

All26 systems have six tolerances, four fixed budgets, added/lost source rows,
nearest locations and unmatched records in the CSVs. A full candidate list's
larger recall does not establish precision, efficiency or field readiness.

## 5. Model compatibility and new-model outcome

Existing compatible v2 RF800-tree/48-field and HGB use their OWN saved contracts
and medians, with exact column order. Stored RF-transformed composites are not
fed to HGB. Original frozenv1 differs in35/48 representations and is not used to
score these new measurements. All old model bytes remain unchanged.

**Spacing correction:** today's normal research app and saved experiment use
10 m computational probes and repaired dense-road profiles. The old production
path defaults75 m; it is a different baseline, not current inference spacing.
Baseline profile availability is97.85%, not universally missing or constant;
33.11% of baseline representatives have zero dip. Small D8 profile availability
is89.47%, with42.98% zero dip. Source-level fields and train p01/p99 distributions
are in `profile_sampling_audit.csv` and `existing_model_distribution_audit.csv`.
Missing/constant portions remain limitations; a schema match is not calibration.

The original RF's training manifest contains941 ordinary unvisited negatives
and5 proximity negatives. These weak negatives were NOT imported into the new
ranker. Separately, the authoritative-positive/verified-negative manifest has
2,526 positive candidate rows belonging to689 source sites and126 verified
negative sites. Use one deterministic actual-coordinate medoid per positive
source site, not2,526 independent positives. Exclude one spatial label conflict,
leaving689 positives and125 verified negatives. No summer row is added as a new
label, and no unmatched prediction is assigned negative.

The fixed new logistic ranker has C=1, balanced classes, ten usable log physical
features, train-only median imputation/scaling, no tuning, coordinates, route ID,
label/source flags, inventory distance or training-exposed RF/HGB scores. The two
profile features fail a fixed class/source availability-gap check and are excluded
within each training fold, rather than letting missingness identify inspection
source. See `supervised_site_manifest.csv` and four separate experimental bundles.

Hold out complete recovered corridors and complete other road IDs; additionally
purge every training site within1,000 m of the held-out road geometry and all
siblings of those sites. Minimum retained distances are1029/1117/1064 m. Negative
training sites remaining by fold:50/17/120. This is a severe geographic/negative-
coverage limitation, not a basis for field-precision claims. Three fold models
and one all-data descriptive model were fitted once (four fits total); grouping
correction/verification reuses their identical bytes with zero additional fits.

New-model descriptive@1000=19; cross-fit@1000=13; nested population+cross-fit=13.
**Reject this new model for recommendation use.** Confirmed labels alone do not
make the coarse physical features discriminative for candidate-source ranking.
The small, geographically narrow verified-negative set and representation shift
are plausible limitations, not individually established causes of model failure.

## 6. Geographic and whole-road variation

Northing bands were fixed at4625000/4645000 m, with entire recovered corridors
assigned by dissolved centroid and other whole road IDs by centroid. Entire road
groups stay together. Inventory fold assignment uses its nearest loaded road for
evaluation only. Training purges include distant siblings on held-out corridors.

Each cell below uses a separate1,000-stop fold queue (three queues=3,000 stops).
Do NOT sum and call it a global1,000-stop result. Existing RF/HGB scores here are
fixed historical models, not newly purged spatial-fold fits; only SITE-MODEL is
fitted with this study's outer-road/spatial exclusion.

{fold_table}

Nested admission/reranking selection uses only the OTHER two folds' retrieval,
then evaluates the remaining outer fold. Settings differ across folds; the single
pooled cross-fitted1,000-stop queue matches99, below123. RF itself retains historical
training exposure, so even this is retrospective development evidence, not an
untouched generalization test. Different fold models' score scales also need
future independent calibration; their pooled site-model score is not a probability.

Additional existing-model comparison reruns baseline20 m score-ordered
representative selection using the **own-road-excluded RF/HGB scores**, not final
RF-selected geometry. It reproduces the old proper raw-RF128 and HGB110 controls.
Policy selection used exposed development data; the22 recovered corridors were
historically used in method development. Denominator845 differs from847.

{road_table}

The sealed previous proper RF/ROUTE-Z control reaches133 at1,000 (HGB/ROUTE-Z123),
so selective soft diversity is not a new overall winner even on those exposed
corridors. Geography, representation and model-exclusion scope are kept separate.

Region results allocate the single global match set, never independently rematch
each region/route and add overlapping credits:

{region_table}

Every recorded route (including weak/zero results) is in `region_route_results.csv`,
with same-budget added/lost rows in `budget_record_changes.csv`. The112 unsupported
records remain80 outside terrain and32 no-loaded-road query failures; all959 source
rows remain intact. Unsupported areas do not become0% classifier accuracy.

## 7. Verification, reproducibility and next evidence

Independent reverse-direction matching rechecks **{verification["independent_metric_rows_replayed"]} metric rows
across {verification["queue_systems"]} systems**. Three admission/grouping rules replay deterministically after
input shuffling; baseline IDs, scores and geometries are exact. Every site-model
score is replayed with its saved fold bundle, and training-source/corridor/sibling/
1 km exclusions are independently reconstructed. Protected291 artifacts,430 guarded
inputs and baseline runtime sources remain byte-identical. `verification.json`
records actual counts. Complete Python/JavaScript suite results are in
`VERIFICATION.md`; new regressions cover NaN channel identity, close distinct
sites, immutable anchors, deterministic ordering, missing admission features,
unvisited-negative rejection and whole-road/sibling spatial exclusion.

No UI route, server, default mode or deployment changed. `npm run dev` still opens
the normal research workspace. These unsuccessful/weak alternatives remain
separate research scripts and artifacts; the original frozen models and Supabase
production data were not changed. Local handoff remains ignored and uncommitted.

Full fixed-spec replay into a NEW directory (four separate experimental fits):

```sh
scripts/python.sh scripts/replay_selective_retrieval.py \\
  --output reports/experiments/selective_retrieval_replay
```

Read-only numeric/integrity replay of this result (zero fits):

```sh
scripts/python.sh scripts/verify_selective_retrieval.py \\
  --output reports/experiments/selective_admission_20261016
```

The supplied study was replayed for grouping with saved model bytes and verified
without a second fit. Full fresh-model replay is provided but was not used to
claim independent validation. Historic completed reports are immutable. Detailed
sources and code are archived with manifests alongside this report.

Before another deployment-oriented choice: obtain untouched, completely surveyed
roads in supported and missing-coverage regions, surveyed physical site identities
for close pairs, accurate inlet/outlet coordinates and verified inspected negatives
across drainage/terrain/source types. Reserve complete roads and nearby geography
before any feature/threshold/model selection. Evaluate one locked queue at realistic
budgets with confidence intervals and actual inspection outcomes. Higher-resolution
cross-road/ditch evidence would help distinguish missing physical support from a
strict-local-maximum suppression error; the present10 m terrain cannot establish it.

Keep **baseline as control**. Retain corroborated selective admission+soft diversity
only as a documented development lead; reject the new logistic ranker and do not
automatically promote any alternative. Recommendations require field verification.
"""
    (out / "REPORT.md").write_text(report)
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
