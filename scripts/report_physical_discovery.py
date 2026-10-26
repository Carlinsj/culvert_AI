"""Build the measured research report from completed, verified result tables."""

import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/physical_discovery_20261016"


def md(frame):
    def value(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v).replace("|", "/")

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
            *[
                "| " + " | ".join(value(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None)
            ],
        ]
    )


def main():
    verify = json.loads((OUT / "verification.json").read_text())
    coverage = pd.read_csv(OUT / "candidate_and_site_coverage.csv")
    primary = coverage.loc[coverage.tolerance_m.eq(20)].copy()
    budgets = pd.read_csv(OUT / "fixed_budget_retrieval.csv")
    global_budgets = budgets.loc[budgets.budget_scope.eq("global_fixed_sites")]
    top = global_budgets.pivot(
        index=["variant", "family", "policy"], columns="requested_budget", values="matched"
    ).reset_index()
    top.columns = [str(v) for v in top.columns]
    stages = pd.read_csv(OUT / "baseline_stage_coverage.csv")
    diagnosis = pd.read_csv(OUT / "all_959_record_diagnosis.csv")
    misses = diagnosis.loc[diagnosis.query_success & ~diagnosis.full_list_matched20]
    categories = [
        "No road coverage",
        "No raw candidate generated",
        "Candidate poorly localized",
        "Candidate removed by grouping/filtering",
        "Candidate retained but ranked too low",
        "Search/query limitation",
        "Coordinate or matching ambiguity",
        "Other/undetermined",
    ]
    failure = pd.DataFrame(
        {
            "Primary category": categories,
            "Supported full-list misses": [
                int(misses.primary_failure_category.eq(c).sum()) for c in categories
            ],
        }
    )
    rock = pd.read_csv(OUT / "rock_record_diagnosis.csv")
    rock_metrics = pd.read_csv(OUT / "rock_variant_metrics.csv")
    exposure = json.loads((OUT / "provenance_audit.json").read_text())
    folds = pd.read_csv(OUT / "proper_complete_road_holdout_comparison.csv")
    fold_table = folds.loc[
        folds.budget.astype(str).isin(["100", "250", "1000", "all"])
        & folds.variant.isin(["BASELINE", "HYBRID-GROUPED"])
    ][["variant", "family", "policy", "budget", "recommendations", "records", "matched"]]
    region_routes = pd.read_csv(OUT / "region_route_performance.csv")
    regions = (
        region_routes.groupby(["variant", "region"])[["records", "matched20"]].sum().reset_index()
    )
    regions["recall_percent"] = 100 * regions.matched20 / regions.records
    distribution = pd.read_csv(OUT / "feature_compatibility_distribution.csv")
    shifts = (
        distribution.loc[distribution.variant.eq("HYBRID-GROUPED")]
        .sort_values("outside_training_p01_p99_fraction", ascending=False)
        .groupby("family", sort=True)
        .head(5)
    )
    burden = []
    for name in ["BASELINE", "HYBRID", "HYBRID-GROUPED"]:
        f = pd.read_csv(OUT / f"duplicate_burden_{name}.csv")
        burden.append(
            dict(
                variant=name,
                records_with_multiple_nearby_recommendations=int(
                    f.candidates_within20m.gt(1).sum()
                ),
                excess_nearby_recommendations=int(f.excess_nearby_recommendations.sum()),
            )
        )
    paired = pd.read_csv(OUT / "paired_localization.csv")
    common = paired.loc[paired.baseline_matched & paired.hybrid_matched]
    h = primary.set_index("variant").loc["HYBRID-GROUPED"]
    concentration = global_budgets.loc[
        global_budgets.variant.isin(["BASELINE", "HYBRID-GROUPED"])
        & global_budgets.family.eq("RF")
        & global_budgets.requested_budget.isin([100, 250, 1000])
    ][
        [
            "variant",
            "policy",
            "requested_budget",
            "represented_routes",
            "maximum_route_share",
            "routes_with_no_recommendation",
        ]
    ]
    negatives = "The regional full-list gain does not establish inspection efficiency: the grouped hybrid loses under raw RF at every equal budget below. Rock Ledge does not improve. Retain the unchanged baseline control, not a new winning detector."
    report = f"""# Physical culvert discovery: failure diagnosis and controlled comparison

2026-10-16. Research only; no model fitting, architecture change, production write,
Supabase update, default-policy promotion or deployment. All 959 summer rows retained.

## Outcome

Exact current baseline reproduced: **347/847 full-list matches** and **155/847
default Top25/query matches**. The main observed bottleneck is physical-evidence
admission: **458 of the 500 full-list misses**. Dense raw geometry can match
**833/847** records, but dense samples are not culvert detections.

The new fixed HYBRID-GROUPED comparison retrieves **{int(h.matched)}/847
({100 * h.recall:.1f}%)** from **{int(h.candidates):,}** site hypotheses, versus
**347/847 (41.0%)** from **13,065** current baseline sites. This is a retrospective
inventory-recall result, not verified precision or classifier accuracy.
{negatives}

## 1. Scope, assumptions and reproduced baseline

Read-only source: `Culvert_Locations_All_Teams_Summer_2026.xlsx`, sheet
`Culvert Locations!A1:F960`; SHA256
`b74908b7238f8c5c93f890f6eeddb886d24a96bfb4ec36eb832b0ae8988e9013`.
All959 coordinates finite and distinct;396 populated IDs unique,563 blank IDs.
Stable source-row identities preserve every record, including60 nearby pairs
within20m (eight within5m). These may be separate structures/endpoints/repeated
observations; no physical deduplication is justified by this workbook alone.

Latitude/longitude interpreted as WGS84 decimal degrees; survey datum/accuracy
not supplied. Explicit EPSG:32618 native geometry for matching. Primary20m;
sensitivity5/10/15/20/25/50m, inclusive. Deterministic maximum-cardinality one-to-one
matching; one prediction/source row gets at most one credit. Not a nearest-neighbor
count and not a route-by-route rematch added together.

Original167 fixed2km cells, anchored at0,0; unchanged1.5km query circles,157
successes containing847 records. A query contains its cell corners. Keep a point
only in its half-open owning cell. DefaultTop25 is selected before clipping;
raw-score global budgets are a different experiment and are labeled separately.
Nearby points outside the tested-cell union remain excluded. No favorable centers,
inventory snapping, route whitelist, road quotas or coordinate correction.

All167 queries were freshly recomputed, not merely loaded from the prior report.
Every successful query reproduced saved site IDs/ranks/coordinates and scores
(score tolerance1e-12). Native measurements, full48 inputs and each stage are saved.

## 2. All500 supported-search full-list misses

{md(failure)}

Grouping/filtering decomposes into458 admission losses and13 score-ordered
representative losses. Coordinate/matching ambiguity is15 one-to-one capacity
competitions: a retained site is nearby but cannot be credited twice. The13 poorly
localized cases have raw points beyond20m; the remaining road-coverage case is
over100m from the loaded roads. No missing raw point within100m or query-boundary
primary loss was found under the fixed diagnostic order.

**Zero of these500 are attributed solely to Top25.** A separate192 baseline
full-list-matched records are not recovered by defaultTop25. Membership of a
maximum matching can change among competing records, so this flag is not a proof
of unique physical identity or a simple additive ranking waterfall.

`all_959_record_diagnosis.csv` includes original row/ID/coordinates/route/region,
nearest raw/proposal/group/site IDs and native distances, raw scores and physical
measurements, profile/flow/ML-peak availability, road separation, mapped/DEM
drainage distance, training proximity, mutually exclusive primary category and
secondary hypotheses. `failure_secondary_evidence_counts.csv` retains overlapping
secondary counts; do not add them as disjoint failures.

Of458 admission losses,457 have at least one available nearby profile;109 have
some nearby dip>=0.15m,266 some positive flow/concavity evidence,80 some one-hectare
area evidence and225 an ML local peak.449 have the nearest raw point over20m from
mapped water. These are evidence gaps/selection conflicts, **not proof of a ditch,
GPS error or an absent culvert**. Neither a high RF score nor a nearby label is
used to admit a physical proposal.

Baseline stage accounting (unranked, before considering inspection budgets):

{md(stages.loc[stages.tolerance_m.eq(20), ["stage", "candidates", "covered_any", "matched_one_to_one", "recall"]])}

Nearest coverage and maximum matching differ. The primary per-record failure
partition is tied to the reproduced final matching, so counts need not equal
differences between separately reoptimized stage match sets.

## 3. All112 unsupported records

-80 outside the loaded terrain bounds (all western Region6 records in this set).
-32 inside terrain but queries have no loaded roads:26 NY9G records (5 Region6,
  21 Region8) and6 Region8 NY209 records.
-All112 stay in the service denominator. Full baseline347/959=36.2%; unsupported
  records are not successful zero-detection tests or evidence of0% classifier accuracy.

Finite coordinates do not establish correct survey location/route labeling.
Road-name conventions alone do not determine support. No new road/terrain download,
invented geometry or truth relocation extends the tested area.
See `unsupported_112_breakdown.csv` for each route and source-row audit.

## 4. Fixed physical hypotheses and feature compatibility

Parameters fixed in `FIXED_METHOD.md` after baseline diagnosis and before new
regional matching. Add actual road intersections with one-hectare conditioned
D8 paths; require>=30degree acute angle and opposite-side endpoints. Native pointer
encoding follows [official Whitebox documentation](https://www.whiteboxgeo.com/manuals/legacy/tool_help.html#d8_pointer),
checked against the saved Whitebox-native metadata. No ESRI encoding assumption.
The existing10m DEM and D8/area rasters are only read, never regenerated or edited.

Keep baseline proposals and add20m-shoulder road-elevation minima with>=0.15m
bilateral rise and local one-hectare flow maxima with positive9x9 concavity.
These are physically guided selections of measured10m probes, not uniform samples
presented as detections. Missing shoulders/terrain and flat ordinary roads do not
automatically admit candidates. No mapped stream is required for DEM/profile sites.
Ten-metre terrain cannot establish pipe locations, roadside ditch dimensions or
embankment openings; cross-road embankment evidence remains unavailable.

Only existing hydrology-sites-v2 RF/HGB models are scored. Each uses its **own**
saved48-field transform, imputation and original dense baseline reference context.
No aggregate/group-average vector, member-maximum score, new field or new calibration.
RF parity was checked at every original candidate. Original frozenv1 is preserved
and **not used** on the incompatible v2 feature representation.

Both saved model families can legally consume the exactv2 fields; this does not
establish distributional validity on new physical sites. Largest descriptive
training-p01/p99 shift flags for the grouped hybrid are:

{md(shifts[["family", "feature", "candidate_missing_fraction", "outside_training_p01_p99_fraction"]])}

All48 fields/families/variants are in `feature_compatibility_distribution.csv`.
These are warnings, not calibrated uncertainty or new thresholds selected by results.

## 5. Stage A: candidate and site coverage

BASELINE is the exact current recommended geometry; BASELINE-RAW is its complete
raw population. DENSE is a computational ceiling, not a usable detector. Today's
baseline already probes at10m; DENSE is not falsely called a denser improvement
over today's baseline.75m is only the historical sampling control.
STREAM is strict mapped crossing hypotheses; DEM includes existing networks plus
the new one-hectare paths; HYBRID combines physical evidence; HYBRID-GROUPED freezes
identity-aware representatives before ranking.

Primary20m one-to-one results on the same847 records/owned-cell scope:

{md(primary[["variant", "candidates", "unique_coordinates", "covered_any", "matched", "missed", "recall", "nearest_median_m", "matched_median_m"]])}

Full tolerance sensitivity (matched distinct source rows):

{md(coverage.pivot(index="variant", columns="tolerance_m", values="matched").reset_index().rename(columns={5: "5m", 10: "10m", 15: "15m", 20: "20m", 25: "25m", 50: "50m"}))}

Raw candidates / unique coordinates are computational counts, not confirmed
physical culverts. Matched-error medians condition on different retrieved subsets;
`paired_localization.csv` separately compares nearest distances for common records.
Among{len(common)} records matched by both baseline and grouped hybrid, median
nearest-distance change is{common.nearest_change_m.median():.3f}m (negative=closer).

## 6. Stage B: grouping and inspection burden

Grouping requires the same actual road/part and a shared nonconflicting channel
or coincident representation. Every pair must satisfy20m spatial and chainage
bounds; no chaining, cross-road merging or visual jitter. Representative is an
actual-coordinate medoid with stable-ID ties, independent of model score/inventory.
Retain member IDs/coordinates/sources/measurements/scores and grouping rationale.
Unknown neighboring extrema are kept separate. Shared channel is a hypothesis,
not definitive culvert identity; very close structures on one channel remain a risk.

{md(pd.DataFrame(burden))}

This table measures excess recommendations near a recorded point; it does not
prove every such pair is the same culvert. HYBRID versus HYBRID-GROUPED counts and
matches above expose any representative losses. Lower marker count alone is not
a success criterion. Separate channels/roads/parts are explicitly regression-tested.

## 7. Stage C: distinct matches at equal global inspection budgets

Same fixed site population per variant, separately ordered by existing RF/HGB raw
score or dynamic ROUTE-Z. Global ROUTE-Z is recomputed on the pooled owned-site
population; local query z-scores are not pooled. No probabilities, tuned cutoffs,
road quotas or performance-based representative moves. BASELINE here keeps its
current final-model-selected coordinates; reordering is not a new raw/HGB NMS.
Columns below are distinct matched inventory records at the indicated maximum
recommendation budget (short queues may contain fewer recommendations):

{md(top)}

Exact recommendation counts, per-query raw budgets and full population sizes are
in `fixed_budget_retrieval.csv`. The unchanged current default155/847 uses local
ROUTE-Z Top25 before clipping and is **not** any globalTop25 row here.

Route concentration (RF; route-name grouping, not guaranteed unique road identity):

{md(concentration)}

For example, final raw RF Top100 selects19 sites on State Rte32, but none on
US Hwy209 (96 available baseline sites), State Rte208 (121 available), or I-87
(301 available). This is real route neglect under global ordering, not proof
these roads lack culverts. Names can cover multiple physical segments; no quotas
or post-result boosts were added to disguise this limitation.

`route_budget_concentration.csv` lists every available route, selected count and
neglected flag at every budget. Zero selected does not mean absent culverts.
`ranked_region_route_performance.csv` allocates the one global matching to each
recorded region/route; no separate rematching inflation.

## 8. Geographic consistency and model-only whole-road holdouts

Full-list geographic allocation at20m:

{md(regions.loc[regions.variant.isin(["BASELINE", "STREAM", "DEM", "HYBRID", "HYBRID-GROUPED"])])}

Full original route-label detail: `region_route_performance.csv`. Improvements
must be checked across routes, not inferred from pooled totals or Rock Ledge.

All959 records were exposed in prior evaluation; RF positive-training proximity
within25m is{exposure["RF_within25m_training_positive"]}/959, HGB proximity is
{exposure["HGB_within25m_training_positive"]}/959. This is spatial exposure, not
verified exact record identity. The new physical hypothesis was designed after
examining failure diagnostics, so even the nonproximal subset is not untouched.

Additionally reuse the six **existing**, whole-corridor-excludedv2 fold models,
checking their training manifests exclude each evaluated corridor. No refitting.
`proper_complete_road_holdout_comparison.csv` regenerates baseline admission/NMS
from the owned raw population using those excluded-road scores, avoiding geometry
selected by the exposed final RF. Other physical representatives remain label-free.
The evaluated extent is the same summer cells on recovered road IDs, not a new
complete survey of every road. Raw measurements/normalization context stay fixed.

{md(fold_table)}

These are **retrospective development holdouts**, not independent generalization:
old labels/routes/policies/features were previously exposed. The additional
`existing_complete_road_holdout_comparison.csv` is only a fixed-current-site score
audit and retains final-RF-selected baseline geometry; do not call it a clean
end-to-end road holdout. Preserve this distinction when interpreting differences.
Reserve future untouched NYSDOT data/new field roads for a locked final test before
any further candidate/feature/threshold/policy selection.

## 9. Rock Ledge, not an optimization target

Exact original1.5km circle, seven records, all NY/US209 labels. Baseline full95
recovers4/7; the road-filtered23 retains the same4 and their original ranks.
The filter removes72 unrelated-road recommendations; it does not discover a new
culvert, rerank the detector, or establish meaningful model-based site selection.
It is useful geographic restriction, not higher detector accuracy.

{md(rock_metrics.loc[rock_metrics.tolerance_m.eq(20) & rock_metrics.budget.astype(str).isin(["25", "all"]), ["variant", "budget", "recommendations", "matched", "records", "recall"]])}

All three baseline full-list misses (rows33,205,214) are admission losses, not
missing raw geometry, grouping loss or merely low rank. Their nearest raw samples
are5.561m/8.851m/3.605m away, have available profiles but zero60m depression; sampled
upstream areas400/4800/100m²; nearest mapped water729/966/289m. Scores approximately
0.542/0.642/0.622 are **not probabilities**, and cannot substitute for physical
support. New one-hectare/short-profile proposals do not recover these three.
See `rock_three_misses_nearest_evidence.csv` for exact values.

{md(rock[["source_excel_row", "baseline_raw_nearest_m", "baseline_admitted_nearest_m", "hybrid_hypotheses_nearest_m", "hybrid_grouped_nearest_m", "baseline_failure"]])}

The new grouped full223-site list stillrecovers4/7, so **Rock Ledge efficiency is
worse**, not better. Allseven are near training positives. The map can display
raw candidates, DEM paths, physical groups and original summer locations with
nearest-distance popups; none of those inventory overlays enters inference.

![Measured Rock Ledge comparison](rock_predicted_vs_recorded.png)

## 10. Recommendation: retain the baseline control, not a new winning detector

The grouped hybrid raises full-list recall by136 records (347→483), but requires
18,783 additional hypotheses (13,065→31,848). Under raw final RF ordering it loses
at **every equal budget**: baseline versus grouped hybrid is11/8 at25 stops,
20/14 at50,32/25 at100,55/47 at250,79/71 at500 and123/102 at1,000. Therefore this
does not meet the requested combined recall/inspection-efficiency success criteria.

Within the new method, conservative grouping removes2,050 proposals (33,898→31,848)
but loses one maximum match (484→483). That is not a demonstrated absence of
physical-site merging errors. Common-record nearest localization is essentially
unchanged in the median; better full-list coverage alone is insufficient.

The proper whole-road-excluded comparison also rejects a universal win: at1,000
stops, existing RF/ROUTE-Z baseline finds133/845 versus grouped hybrid91/845.
Final-model raw RF looks strongest in the exposed fixed-site comparison, but the
whole-road-excluded RF comparison instead favors ROUTE-Z slightly (133 versus128
at1,000). Do not automatically promote raw ordering or HGB based on exposed rows.

**Retain the unchanged feature/RF/ROUTE-Z/legacy20 configuration as the research
baseline control.** Keep physical discovery opt-in as a reproducible diagnostic
ablation, not a field-deployable replacement. Retain the useful inventory overlay,
per-record evidence and existing ruler. No further threshold search, model fitting
or architecture experiment was begun after these negative efficiency results.
The next evidence need is better drainage/ditch/road-location information and
untouched complete-road survey data, not hiding markers or claiming70% accuracy.

## 11. Implementation, safety, verification and limitations

-New `staging/discovery_sites.py`: label-free D8 crossing/adaptive physical
  proposals, strict physical grouping and deterministic fixed-score ordering.
-New `staging/discovery_service.py`: separate compatiblev2 research path.
-Normal root app adds an explicit Advanced comparison mode and an unchecked
  summer-inventory diagnostic overlay. Default feature/RF/ROUTE-Z/legacy20 remains.
  No separate app/route to launch; use `npm run dev`. Restart after Python edits.
-A/B ruler, draggable pins, metres/feet, road filters, original ranks, exports,
  diagnostic layers and simple collapsed research controls remain. No inventory
  features are added to prediction/export recommendations.
-258 Python tests and33 JavaScript tests pass, plus existing security/product-UI
  checks. New tests cover D8 encoding/geometry, unseen IDs, duplicate IDs,
  separate close channels/roads, one-to-one capacity, missing/flat terrain,
  deterministic ties, feature guards, failure accounting and explicit API mode.
-Independent reverse-direction augmenting-path matcher verified
  {verify["independent_stage_metric_rows"]} stage/tolerance rows. All167 fresh
  baseline query replays agree. All291 protected artifacts plus349 baseline
  inputs and savedv2 models remain unchanged. Zero fits and no production writes.
-Browser root→analysis→223 sites→optional inventory popup→nearest distances verified;
  console clean. A/B drag changed1420.95m to1776.24m; feet conversion5827.58ft.
  Existing mouse/touch/keyboard adapter tests pass. This is not field-device/GPS,
  offline, adverse-weather or production-load validation.

Remaining scientific limits: no surveyed-negative/completeness population; unknown
survey datum/GPS/inlet-outlet offsets; incomplete roads/terrain;10m conditioned
DEM may hide small ditches or create artificial flow paths; bridges/open channels
can resemble crossings; physical identity is not certified; inherited label and
development exposure; route/geographic concentration; query-boundary convention;
changed candidate distributions; no calibrated probability or validated70% accuracy.
Independent field readiness is **not established**.

## Reproduction and artifacts

Cache-only independent verification (does not overwrite historical inputs):

```sh
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/verify_physical_discovery.py
```

Full expensive no-fit replay into a new unused directory:

```sh
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh scripts/replay_physical_discovery.py --output reports/experiments/physical_discovery_replay
```

The runner refuses existing output, replays baseline first, repeats the fixed
physical hypotheses and saved-model comparisons, then diagnoses/verifies. It is
not an npm startup hook. Do not rerun old training-script mains. Native per-cell
measurements, transformed features, IDs and original geometry are retained for
auditing; CSV files give human-readable results. Source workbook is never edited.

Key datasets: `all_959_record_diagnosis.csv`, `candidate_and_site_coverage.csv`,
`fixed_budget_retrieval.csv`, `region_route_performance.csv`,
`ranked_region_route_performance.csv`, `proper_complete_road_holdout_comparison.csv`,
`feature_compatibility_distribution.csv`, `paired_localization.csv`,
`rock_record_diagnosis.csv`, and `verification.json`. Source/as-run hashes are saved.

Recommendations require field verification.
"""
    # CommonMark bullets require a space and a blank line before the list.
    report = report.replace("\n-", "\n- ")
    adaptive_path = OUT / "adaptive_sampling_coverage.csv"
    if adaptive_path.exists():
        supplemental = pd.read_csv(adaptive_path)
        report = report.replace(
            "## 6. Stage B",
            "Supplemental adaptive computational coverage control (75 m coarse probes plus\n10 m refinement within 75 m chainage of physical evidence, same road/part):\n\n"
            + md(supplemental[["tolerance_m", "candidates", "matched", "inventory", "recall"]])
            + "\n\nThese probes are not detections. Seeds currently require the full 10 m physical\nscan, so reduced retained sample count does not demonstrate runtime savings.\nThe saved model reference context is not changed to this subset. Method fixed\nin `ADAPTIVE_CONTROL_METHOD.md`; separate independent matching check passed.\n\n## 6. Stage B",
        )
    # Readable prose without altering code, filenames, URLs or inline identifiers.
    prose = []
    fenced = False
    for line in report.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and "`" not in line and "http" not in line and not line.startswith("|"):

            def separate_number(match):
                word, number = match.groups()
                if word in {"D", "WGS", "EPSG", "v", "p", "q", "e", "x", "RF", "HGB"}:
                    return word + number
                if word.endswith("v") and len(word) > 1:
                    return word[:-1] + " v" + number
                return word + " " + number

            line = re.sub(r"([A-Za-z]+)(\d+(?:\.\d+)?)", separate_number, line)
            line = re.sub(r"(?<=[A-Za-z])([,;:.])(?=\d)", r"\1 ", line)
            line = line.replace("defaultTop", "default Top").replace("DefaultTop", "Default Top")
            line = re.sub(
                r"\b(All|all|Of|of|these|those|Only|only|original|default|within|some|Among|among|require|requires|Zero|zero|below|above|full|over|under|saved|separate|no|from|versus|reproduced|with|old|new|allseven)(?=\d)",
                r"\1 ",
                line,
            )
            line = re.sub(
                r"(?<=\d)(m²|m|km|degree|fields|field|rows|records|source|coordinates|cells|queries|proposals|models|sites|cases|are|have|has|remain|remains|stay|finds|recovers|within|positive|negative|full|near|Python|JavaScript|protected|baseline|fresh|stage|physical|admission|one)(?=\b|[A-Z])",
                r" \1",
                line,
            )
            line = line.replace("stillrecovers", "still recovers").replace("v2 RF/HGB", "v2 RF/HGB")
        prose.append(line)
    report = "\n".join(prose) + "\n"
    (OUT / "REPORT.md").write_text(report)
    print(f"Saved measured report: {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
