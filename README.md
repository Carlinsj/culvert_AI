# Culvert AI

A local research workspace for finding and ranking potential culvert locations.
The current geographic data covers an Ulster County, New York pilot, not all of
New York State. Arbitrary road IDs are supported by the inference code; geographic
coverage and accuracy still depend on the available data.

**Current phase: research and validation. Field deployment target: summer 2027,
subject to independent validation and field-readiness checks.** Nothing is being
deployed automatically.

Recommendations require field verification. Scores are uncalibrated ranking
signals, not probabilities. No 70% accuracy or precision claim is validated.

## Physical crossing recovery v7 — 2026-10-20

The label-free v7 generator raises the sealed candidate ceiling on the complete
NY214 and NY28 survey intervals from **20/31 at 117 sites** to **26/31 at 142
consolidated physical sites**. It adds multiscale D8 road crossings and
road-profile terrain lows, then conservatively groups duplicate representations
without using inventory coordinates. The unchanged frozen RF first reaches the
development target at **119 recommendations (22/31, 71.0%)**.

This is a retrospective, development-exposed result, not independent validation.
NY214 reaches 15/15, while NY28 remains below target at 11/16; no classifier was
trained and no production or frozen-model artifact changed. See the
[v7 report](reports/experiments/v7_physical_crossings_final_20261020/REPORT.md).

## Latest experimental v3 checkpoint — 2026-10-17

The isolated 48-hour sprint checkpoint reaches **672/847 (79.34%) retrospective
full-list recall at 20 m** using 68,302 physical-site hypotheses, while preserving
the original RF's 123 matches in the first 1,000 through a baseline-first queue.
This meets the exposed-inventory recall target, but it does **not** establish
generalization, accuracy, or precision: the inventory has prior development
exposure and the candidate list is 5.23x the baseline size. New RF/HGB/PU rankers
all underperform the original top-of-queue ordering and are not promoted.

See `reports/experiments/model_improvement_sprint_20261017/REPORT.md` for the
candidate-cost tradeoff, feature-parity audit, grouped validation, saved models,
exact inference replay, regional results, and remaining 175 matching failures.
Production models, defaults, and deployment remain unchanged.

## Final compact physical queue v6 — 2026-10-19

The final coverage/ranking audit separates the v3 headline from operational
retrieval. The 13,065-site original population covers at most 347/847 records;
the 68,302-site v3 population covers 672/847 but needs 51,814 saved-rank sites to
reach 70%. A sealed, label-free consolidation policy keeps baseline anchors,
multiscale terrain extrema and explicit mapped/DEM crossings, then merges only
compatible cross-generator proposals within one 10 m DEM cell. It never moves a
coordinate or merges different known channel identities.

The experimental v6 queue has 50,826 sites, covers 638/847 (75.32%) when
exhausted, preserves 123/847 at 1,000, and first reaches 593/847 (70.01%) at
42,140 recommendations. That is a 25.6% reduction from v3, but still not a
practical inspection budget. The original RF remains the default and no
independent 70% claim is made. See the
[final optimization report](reports/experiments/final_physical_optimization_20261019/REPORT.md).

## Physical-site ranker v4 — 2026-10-19

The new isolated RF challenger was selected using both complete-route and fixed
5 km spatial-block folds, then evaluated on the unchanged 13,065-site baseline
population. It retrieves **23/847 at 1,000 stops**, versus **123/847** for the
original RF, so it is saved for diagnosis but **not promoted**. Its 815-row
training manifest contains 689 authoritative positives and 126 field-verified
negatives; unlabeled candidates are never treated as negatives. Route/class
confounding remains severe: 11 of 14 observed routes contain only one class.

See the [v4 report](reports/experiments/physical_ranker_v4_20261019/REPORT.md),
saved queue, exact split definitions, OOF predictions, feature importance,
provenance hashes, and serialized model. No frozen artifact or default changed.

## Baseline-preserving reranker v5 — 2026-10-19

The controlled same-population audit shows that v4's failure is not caused by
using more candidate points: v4 retrieves 23/847 at 1,000 on the original 13,065
sites and 19/847 on the 31,848-site hybrid, while the original RF retrieves 123
and 102. An exact repaired-feature replay reproduces every original RF score and
123/847, but its historical training data include 941 ordinary unlabeled
candidates treated as negatives plus 696 development-exposed official matches.

The new verified-label-only baseline-preserving v5 reranker uses whole-route-
excluded original RF scores and a training-selected 10% physical residual. It
retrieves **113/847 at 1,000**, so it is saved and selectable for research but not
promoted. Only 126 verified negatives exist, and 11 of 14 training routes contain
one class. The next useful input is complete-road field survey data with verified
absences reserved by route, not another expansion of uniformly sampled points.

See the [v5 report](reports/experiments/baseline_preserving_v5_20261019/REPORT.md).

## Anti-saturation validation and map comparison — 2026-10-17

The equal-budget control study confirms that the existing RF queue is doing more
than saturating the road network. At 1,000 stops it matches **123/847** searchable
records, versus **23** for the v3 physical-feature ranking, **15** for fixed
uniform road spacing, and a **19.5** mean across 1,000 seeded random-road trials
(11–29 seeded 95% interval). At 13,065 stops the results are 347, 236, 231 and
222.0 respectively. These are retrospective one-to-one 20 m matches, not
precision or independent validation; unmatched predictions remain unverified.

The live research map now defaults to Top 250 and shows recorded matched/missed
locations, predicted-site pins, dashed match lines, an accessible collapsible
legend, and active-budget performance. Its county benchmark clearly separates
the exact 847 successful searches from 112 fixed-cell query failures while
preserving all 959 workbook records. Full-list and raw-candidate views are marked
diagnostic. See `reports/experiments/anti_saturation_validation_20261017/REPORT.md`.

## Run the research app

From this repository, using the existing installed environment:

```sh
npm run dev
```

`npm start` starts the same application. Use one command, not both at once.
Open the address printed in the terminal. There is one research workspace at the
root URL; no separate staging route or comparison server is required.

The shared launcher is `scripts/serve_web.sh`, which uses this checkout's Python
environment and `scripts/serve_research.py`. It binds only to this computer
(loopback), using port 8096 by default. Startup does **not** train models,
regenerate datasets, import observations, run experiments or deploy.

Stop with Ctrl+C. If an older server is already running, stop it before restarting
so the updated launcher is used. An explicit port override is available:

```sh
npm run dev -- --port 8097
```

Refresh the browser after UI edits. Restart after Python edits. This server does
not provide automatic hot reload. The same launcher also works directly with
`bash scripts/serve_web.sh`, including from a different working directory.

## Using the workspace

- Start on the four-area overview: Beaver Kill, Rock Ledge Ln, Beech Hill Rd and
  Braden Rd. Choose an area, select Show sites, or use your location. Opening the
  app alone does not run inference. Geolocation needs browser permission.
- Use the road filter and site budget to narrow the map, list and export together.
  Road filtering happens before the display cap and preserves original ranks,
  scores and coordinates. It does not improve or rerank the detector.
- Use Measure distance to place independent A/B pins, drag either endpoint, and
  read straight-line metres or feet. It is not road travel distance.
- Coordinates, discovery/ranking/grouping comparisons and diagnostic layers stay
  under Advanced in the same app. No additional server is needed to compare the
  available fixed policies.
- Advanced → Analysis settings includes an opt-in **Small drainage + adaptive
  sites** research comparison. It uses identity-based physical grouping; the
  baseline grouping selector does not alter that mode. This is not a promoted
  detector. The existing feature-driven baseline remains the default.
- **Predictions** is the normal recommendation map; its Summer 2026 inventory
  overlay is off by default. **Actual vs Predicted** is a separate workflow for
  Rock Ledge, Beaver Kill, Region 6, Region 8, route, coordinate-radius, or all
  supported records. It exposes model, Top-K budget, tolerance, visibility,
  fit-to-recorded and next-missed controls. **Model Performance** separates
  equal-budget results from full-population coverage. The saved inventory never
  enters inference or recommendation exports.
- The site budget defaults to Top 250, with 25/50/100/250/500/1,000 choices.
  **Full list — diagnostic** and **Raw candidates — Diagnostic only** are explicit
  workload/diagnostic views, not ordinary field-review recommendations.
- Notes are browser-session-only research annotations. They are not durable field
  records, synced observations or training labels. Export them before leaving the
  session if needed.

An empty or unsupported area does not establish that no culverts exist.

## What the app actually runs

The normal launch uses `staging/feature_service.py` and `staging/feature_web/`.
The historical folder name is an internal implementation detail, not a separate
application to launch.

- Existing **hydrology-sites-v2 RF experimental** model: 800 trees, 48 ordered
  features, with its compatible preprocessing and training-only imputation.
- Default discovery: feature-supported crossings and local road-profile /
  flow-concavity extrema. Dense road anchors are computational samples, not
  automatically displayed detections. No inventory is read during inference.
- Default ordering: dynamic ROUTE-Z; existing road-aware, score-ordered 20 m
  consolidation. Raw RF ordering and conservative evidence-identity grouping
  remain explicit research comparisons, not promoted winners.
- Analysis distances use EPSG:32618 metres; map/export coordinates use WGS84.
  Representatives keep actual candidate coordinates; no inventory snapping,
  visual jitter or road quotas.

The original `models/flow_c_frozen_v1.joblib` remains an unchanged historical
baseline. It is **not the model used by normal startup**: 35/48 v1/v2 feature
representations differ, so repaired v2 measurements must not be fed to it.
Legacy UI/server code remains for provenance and reproducibility, but is not
started by `npm run dev` or `npm start`. Changing the research entry point is not
a field-model promotion.

## Latest ranking research — completed 2026-10-17

**Keep the existing default.** Targeted admission and ranking did not establish a
robust replacement. All comparisons are retrospective; nothing was deployed.

| Global queue | Sites | Full matches /847 | Matches within 1,000 stops |
| --- | ---: | ---: | ---: |
| Existing fixed-site baseline, raw RF | 13,065 | 347 | 123 |
| Full grouped hybrid, raw RF | 31,848 | 483 | 102 |
| Selective admission, raw RF | 17,083 | 372 | 109 |
| Selective + development-selected soft diversity | 17,083 | 372 | 127 |
| Nested geographic/corridor selection | 18,075 | 374 | 99 |
| Separate experimental site model, cross-fitted | 17,083 | 372 | 13 |

The 127 result is a small exposed-development gain, not validated field accuracy.
Nested selection does not preserve it. The new site ranker is rejected for
recommendation use. Baseline raw RF here is a research benchmark, not a promotion
over the app's unchanged dynamic ROUTE-Z default or its local Top25 queue.

The hybrid displaces 603 original Top1,000 representatives: 52 previously matched
records are lost and 31 gained. Nearby same-road hypotheses crowd the queue;
unmatched predictions remain unverified, not false positives. Of 458 admission
losses, 296 involve local-window/peak selection, 161 lack the required measured
support, and one lacks a usable profile. No inventory coordinates enter admission.

Three selective rules preserve original IDs, locations, scores and provenance.
Conservative grouping finds no further defensible merges in these selective
populations; unknown nearby channels are kept separate. Four separate experimental
site-model fits use confirmed positive sites and verified negatives only, with
complete-road/corridor and 1 km spatial exclusions. Frozen models remain unchanged.

[Full report, all budgets, per-route results and limitations](reports/experiments/selective_admission_20261016/REPORT.md)
and [comparison data](reports/experiments/selective_admission_20261016/comparison.csv).
The normal app remains available with `npm run dev`; no new route is needed.

Zero-fit numeric/integrity verification:

```sh
scripts/python.sh scripts/verify_selective_retrieval.py \
  --output reports/experiments/selective_admission_20261016
```

## Previous physical-discovery diagnosis — 2026-10-16

The physical-discovery study freshly replayed all 167 baseline queries and
reproduced the saved site IDs, geometry, scores and ranks. It diagnosed every
supported-search full-list miss:

| Primary observed loss | Records |
| --- | ---: |
| Physical-evidence admission | 458 |
| Score-ordered representative selection | 13 |
| Raw localization beyond 20 m | 13 |
| One-to-one matching competition | 15 |
| Far from the loaded roads | 1 |
| Total supported-search full-list misses | 500 |

The separate 112 unsupported records comprise 80 outside the terrain extent and
32 in queries without loaded roads. Raw geometry can match **833/847** supported
records at 20 m, but these **236,858 computational candidates are not culvert
detections**. Admission is the largest observed loss. Another 192 full-list-matched
records are not recovered by default Top25; they are not part of the 500 misses.

The new Rock Ledge physical comparison still finds **4/7**, with **223** grouped
recommendations versus the baseline's **95**. That is not an improvement. No
threshold is tuned to recover these seven. Across the supported regional searches,
the grouped hybrid finds **483/847 (57.0%)** versus the baseline's **347/847
(41.0%)**, but needs **31,848 sites versus 13,065**. Under the same raw RF ordering,
it loses at every tested equal inspection budget:

| Global inspection budget | Baseline distinct matches | Grouped hybrid distinct matches |
| --- | ---: | ---: |
| 25 | 11 | 8 |
| 50 | 20 | 14 |
| 100 | 32 | 25 |
| 250 | 55 | 47 |
| 500 | 79 | 71 |
| 1,000 | 123 | 102 |

This does **not** justify replacing the baseline. These global queues are separate
from the app's per-query default Top25, and raw RF is not automatically promoted.
The proper existing whole-road-excluded RF/ROUTE-Z comparison also favors the
baseline at 1,000 stops: **133/845 versus 91/845** for grouped hybrid. These roads
were historically exposed during development; this is not independent validation.
Regional variants, fixed-budget
RF/HGB comparisons, physical grouping, distribution shift and existing
whole-road-excluded models are documented in the
[physical-discovery report](reports/experiments/physical_discovery_20261016/REPORT.md).
This remains retrospective research, not independent validation or field readiness.

Detailed records: [all 959 diagnoses](reports/experiments/physical_discovery_20261016/all_959_record_diagnosis.csv),
[candidate/site coverage](reports/experiments/physical_discovery_20261016/candidate_and_site_coverage.csv),
[fixed-budget retrieval](reports/experiments/physical_discovery_20261016/fixed_budget_retrieval.csv).

### Unchanged current-app baseline

The summer workbook has 959 records. Fixed 2 km geographic cells were evaluated
through unchanged 1.5 km app queries. Of 167 cells, 157 queries succeeded and
contained 847 records; 112 records were outside supported searches. Overlapping
queries were clipped to owning cells to avoid duplicate predictions and matches.

Primary matching is one-to-one within **20 m**:

| Scope | Queue | Recorded culverts found | Inventory recall |
| --- | --- | ---: | ---: |
| Rock Ledge area, 1.5 km | Default top 25 | 2/7 | 28.6% |
| Rock Ledge area, 1.5 km | Full 95-site list | 4/7 | 57.1% |
| Rock Ledge area, US 209 filter | 23 sites | 4/7 | 57.1% |
| Region 6, supported searches | Full lists | 16/40 | 40.0% |
| Region 8, supported searches | Full lists | 331/807 | 41.0% |
| All supported regional searches | Top 25 per query, before cell clipping | 155/847 | 18.3% |
| All supported regional searches | Full lists | 347/847 | 41.0% |
| All 959 input records, including unsupported locations | Full lists | 347/959 | 36.2% |

Rock Ledge's default list matches 2 of 25 predictions (8.0% inventory-supported
fraction); its full list matches 4 of 95 (4.2%). These are **not verified
precision**: the workbook does not certify every local road or provide a complete
surveyed-negative population. Recall is the fraction of recorded culverts found,
not the percentage of predictions known to be correct.

All seven Rock Ledge records are close to positive training locations; 553/959
summer records are within 25 m of such locations. This is a retrospective check,
not independent generalization accuracy. Beech Hill and Braden navigation circles
contain no summer records and cannot be assessed from this workbook. Geographic
grid results can differ from individual map views because the search context
changes. WGS84 source coordinates are assumed; survey datum/accuracy are not
documented in the workbook.

[Summer inventory report, per-route results and limitations](reports/experiments/summer_inventory_20261016/REPORT.md)

Current research priorities are physical-evidence admission losses, ranking
quality at realistic review budgets, coverage outside the loaded roads, and
prospective evaluation on independently surveyed roads with verified negatives.
No threshold, model or field-deployment decision should be selected from these
exposed records and advertised as independent validation.

## Historical research

These reports describe their own models, candidate populations, scopes and source
snapshots. Their scores and counts are not interchangeable with today's app:

- [Feature localization](reports/experiments/feature_localization_20261015/REPORT.md)
- [Grouping and physical-site retrieval](reports/experiments/site_retrieval_20261016/REPORT.md)
- [High-recall hybrid comparison](reports/experiments/high_recall_hybrid_20261015/REPORT.md)
- [Hydrology feature repair and crossing proposals](reports/experiments/hydrology_driven_sites_20261015/REPORT.md)
- [Retrospective held-out road audit](reports/experiments/real_inventory_heldout_roads_20261015/REPORT.md)
- [Matched-population ranking control](reports/staging/ranking_policy_correction_20261015/REPORT.md)
- [Complete surveyed-route supervision](reports/experiments/surveyed_route_supervision_v4_20261020/REPORT.md)
- [v7 physical crossing recovery](reports/experiments/v7_physical_crossings_final_20261020/REPORT.md)

Reports and scientific assets are retained locally outside Git. Preserve old
reports and hash manifests as historical records; do not regenerate them merely
because documentation or startup code has changed.

## Setup and required local assets

Use the working environment already installed in this checkout. For a fresh
environment, the declared minimums are Node.js 18+ and Python 3.10+:

```sh
npm ci --ignore-scripts
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Serialized estimators require compatible Python/library versions; dependency
lower bounds do not guarantee artifact compatibility. `scripts/python.sh` selects
this checkout's `.venv` and source tree. Normal `npm install` also invokes the
Python bootstrap hook; it is unnecessary just to restart an installed app.

Required inference assets include:

```text
models/flow_c_frozen_v1.joblib.schema.json
reports/experiments/hydrology_driven_sites_20261015/hydrology-sites-v2_RF_experimental.joblib
data/raw/roads.gpkg
data/raw/streams.gpkg
data/processed/hydrology/usgs_3dep_10m/
```

The hydrology directory supplies the projected DEM, accumulation/upstream-area,
edge-flow and drainage/crossing layers. Models, GIS data, rasters and research
outputs are local assets, not included in a Git clone. The inventory workbook
is an evaluation input, not a live inference dependency. Do not delete a model
merely because its historical storage directory is named `reports/`.

## Verification

```sh
npm test
npm run test:research
```

`npm test` includes Python, security, retained legacy UI checks, and research
startup/map/ruler tests. The second command runs just the JavaScript research
suites. Tests requiring real models/GIS need those local assets.

Latest verification on 2026-10-17: all 280 Python tests and 35 JavaScript tests
passed, along with security, retained UI and focused lint checks. A live browser
pass verified the Top 250 default, marker states, active matching metrics and
847-record baseline/random comparison through the normal launcher.

The latest summer comparison independently checked 80 local and 40 regional
metric rows, replayed Rock Ledge inference, passed 38 relevant regression tests,
and verified 291 protected artifacts plus the current model/workbook unchanged.
Those numbers describe that completed audit, not a field-accuracy certification.

`npm run build` remains the retained legacy static/API asset check. It does not
package or deploy this Python/GIS research workspace. Commands such as
`predict:actual`, `actual`, `pipeline:ulster`, `retrain:from-vercel`, `demo`,
model-promotion scripts and historical experiment runners can fit models or
overwrite outputs; they are **not** startup or repair commands. Run them only
when the intended research change is explicitly authorized.

## Field deployment boundary

Summer 2027 is the planned field-deployment window, not a claim that the system
is ready today. Independent road validation, verified precision/recall at a
declared tolerance and review budget, geographic coverage, durable/offline field
records and operational checks remain to be established before deployment.

Existing remote/static configuration and legacy observation/retraining endpoints
are historical code, not a working deployment of the current Python/GIS app.
No remote service is launched, promoted or changed by the normal research command.

## Repository guide

- `scripts/serve_web.sh`, `scripts/serve_research.py`: normal research startup.
- `staging/feature_*`, `staging/retrieval_sites.py`: current research map/service
  and reversible comparison logic; see [workspace controls](staging/FEATURE_README.md).
- `src/culvert_ai/`: GIS, features, frozen-baseline inference and retained
  historical training/label/export utilities.
- `web/`, `server/`, `api/`: retained legacy application and remote API code.
- `models/`, `data/`, `reports/`: scientific assets and provenance.
- `scripts/`, `tests/`: setup, evaluations and regressions.

`handoff.md` remains local-only, ignored by Git and uncommitted.
