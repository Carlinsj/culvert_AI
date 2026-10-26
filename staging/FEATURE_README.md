# Research workspace controls

Run the normal local research application:

```sh
npm run dev
```

`npm start` opens the same app. Use one command, not both at once, and open the
root address printed in the terminal. There is no separate staging route or
comparison server to launch. The shared `scripts/serve_web.sh` launcher calls
`scripts/serve_research.py` using this checkout's Python environment. Startup does
not train, regenerate data, run experiments or deploy. Stop with Ctrl+C.
Refresh after UI edits and restart after Python edits; there is no automatic hot
reload. An explicit `--port` override is supported when the default is occupied.

Current phase: research and validation. Field deployment is planned for summer
2027, subject to independent validation and operational readiness. Internal source
folders retain their historical names for reproducibility; they are not separate
applications that users must navigate between.

Feature mode admits strict crossings or genuine local profile/flow-concavity
extrema. Ordinary road anchors are hidden unless explicitly enabled and are
never described as detected culverts. No map backfill, road quotas, jitter or
outcome-tuned thresholds. Compatible experimental v2 RF orders admitted sites;
the frozen original RF is disabled because 35/48 feature semantics differ.

The app starts with a four-area map overview; it does not run inference until
you choose a road area, select Show sites, or use your location.
The everyday sidebar now contains road-area selection, radius, **Show sites**,
a road filter and the site list. **Explore other road areas** restores the map
overview with Beaver Kill, Rock Ledge Ln, Beech Hill Rd and Braden Rd. The **Road**
filter lists roads within the current search, including roads with zero proposals.
It filters before the display cap and keeps original global ranks/scores; map,
list and export share the same selected-road subset.

Coordinates, model/ranking/grouping settings and all diagnostic layers are under
the collapsed **Advanced** section. Analysis settings apply only with **Show sites**;
changing controls does not fire repeated model requests. Technical site details
are also collapsed. No candidate-generation or ranking default was changed.

## Actual vs Predicted and Model Performance (2026-10-19)

The header separates three tasks. **Predictions** is the uncluttered field-review
map and leaves the recorded-inventory diagnostic off by default. **Actual vs
Predicted** loads recorded culverts and saved predictions automatically for Rock
Ledge, Beaver Kill, Region 6, Region 8, an exact route, a coordinate radius, or
all 847 supported records. Controls switch the saved model queue, Top-K budget,
one-to-one match tolerance, actual/predicted visibility, fit-to-recorded view and
the next missed record. Low-zoom marker aggregation is visual only; server-side
counts and matching always use individual saved sites.

**Model Performance** reports the fixed 20 m retrospective benchmark at equal
budgets and a separate full-population table. The original RF has 13,065 sites
and 123 matches at 1,000. v3 has 68,302 hypotheses and reaches 672/847 only when
all 68,302 are used; it has 236 at an equal 13,065-stop budget. The new v4
challenger has the baseline's 13,065 coordinates but only 23 matches at 1,000,
so the original RF remains the comparison default. See the
[v4 report](../reports/experiments/physical_ranker_v4_20261019/REPORT.md).

The model selector also exposes the v5 baseline-preserving reranker. It keeps the
same 13,065 coordinates and adds a training-selected 10% physical residual to the
original score order. It retrieves 113/847 at 1,000, below the original RF's 123,
so the original remains first and default. The
[v5 report](../reports/experiments/baseline_preserving_v5_20261019/REPORT.md)
documents exact repaired-RF replay, historical pseudo-negatives, ablations,
route/spatial folds, saved models and label-collection needs.

The selector additionally exposes the experimental v6 compact physical queue.
It preserves original-RF ordering for baseline-containing groups, removes
uncorroborated added internal probes, and consolidates compatible proposals from
different physical generators without moving coordinates. It has 50,826 sites,
preserves 123 matches at 1,000, covers 638/847 when exhausted, and needs 42,140
recommendations to reach retrospective 70% recall. This is a coverage diagnostic,
not a promoted model; the original RF remains first and selected by default. See
the [v6 report](../reports/experiments/final_physical_optimization_20261019/REPORT.md).

## Physical-discovery comparison (2026-10-16)

Advanced → Analysis settings → **Small drainage + adaptive sites** uses a separate
research pipeline: transverse one-hectare D8 drainage crossings plus measured
short-profile/flow extrema, then conservative physical-identity grouping. The
baseline grouping selector does not change this mode. No frozen model is edited
or retrained; live scoring uses the existing compatible v2 RF. Existing v2 HGB
and whole-road-excluded RF/HGB models are compared offline with their own saved
48-feature transformations.

This is **not a promoted replacement**. Across 847 supported summer records,
grouped hybrid full-list recall is 483/847 versus baseline 347/847, at a queue cost
of 31,848 versus 13,065. Raw RF global Top100 finds 25 versus baseline 32; Top1000
finds 102 versus 123. Rock Ledge stays 4/7 while the full queue grows to 223 from 95.
The proper existing whole-road-excluded RF/ROUTE-Z comparison also loses at 1000
stops (91 versus baseline 133, on 845 eligible records). Keep the current default
as the research control. These are retrospective, exposed-inventory results.

Advanced → Map layers & diagnostics → **Summer 2026 recorded locations** adds an
unchecked, read-only diagnostic overlay. Popups show nearest raw/recommended
distances, not one-to-one match credit. All original source records are retained
in the report; the map shows only those in the current search circle. Inference
does not read the inventory and exports do not mix it into recommendations.

See the [complete physical-discovery study](../reports/experiments/physical_discovery_20261016/REPORT.md),
including all 500 misses, 112 unsupported records, six variants, fixed budgets,
feature distributions, provenance and reproduction commands.

## Existing map and ruler controls

Use the purple **↔ Measure A → B** button on the map (or **Measure distance**
in the header) to place A then B. The explicit **Place marker A / B** buttons
choose which endpoint to replace. Drag either pin to update geographic
straight-line metres/feet. Candidate clicks snap endpoints rather than open
details while measuring. **Measure from this site** sets A. Reset starts over,
Clear removes both pins, Exit/Escape pauses. Expand endpoint coordinates/snapping
for manual entry or disable snapping. Pin arrow keys move 1 m, Shift+arrow 10 m.
Ruler operations do not move candidates, change ranks, save field notes or call
analysis. Mobile advanced controls collapse to preserve map access.

The normal research app includes this ruler. Retained legacy UI files are not
started by either normal npm command.

Under **Advanced → Analysis settings**, **Grouping policy** compares the existing 20 m baseline with **Preserve distinct
evidence — Experimental**. The new option only combines coincident representations
or shared explicit drainage identities on the same road/part, with bounded diameter
and chainage. Details expose all member evidence and the representative-selection
reason. Nearby distinct extrema/roads/channels are not merged just for proximity.

Under **Advanced → Map layers & diagnostics**, **Primary map representation** switches between grouped recommendations and all
raw candidates. Raw view is diagnostic, not an expanded recommendation queue;
queue/export remain grouped. Optional source layers can intentionally overlap.
The current road filter also applies to raw candidate and source diagnostics.

Seven core layers plus the separate inventory overlay distinguish samples, crossings, proposals,
consolidated sites, recommendations and drainage networks. Spacing diagnostics
use same-road/part chainage and actual road geometry, not geographic neighbours.
Hover/tap any spacing line for its distance. Persistent labels are decluttered
without hiding points or line segments. Field notes stay in this browser session
and exports; they are not production records or model training data.

Executed research and limitations:
[latest grouping/retrieval audit](../reports/experiments/site_retrieval_20261016/REPORT.md),
[browser and regression verification](../reports/experiments/site_retrieval_20261016/VERIFICATION.md),
[localization report](../reports/experiments/feature_localization_20261015/REPORT.md).
The fixed grouping study uses `scripts/evaluate_site_retrieval.py` and
`scripts/report_site_retrieval.py`; copy its fixed plan into a new output directory
before reproducing. It reduces 7,244 proposals to 6,264 conservative groups with
301/705 matches, versus the smaller legacy queue's 5,241 groups and 285 matches.
This is not a uniform retrieval-efficiency improvement; legacy remains default.

Fixed evaluation can be rerun into a **new** output directory with
`scripts/evaluate_feature_sites.py`, then `scripts/validate_feature_spatial_null.py`
and `scripts/report_feature_sites.py`. See the report for exact commands.
It reuses compatible saved held-out scores and does not fit/predict models.
Live map inference is separate and uses the existing final experimental v2 copy.

Recommendations require field verification. Retrospective physical localization
enrichment is measurable, but coverage and local RF ranking remain insufficient
for a field-ready detector. No automatic deployment or field-model promotion.
