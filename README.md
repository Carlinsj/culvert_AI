# Culvert AI

Geospatial research software for ranking potential culvert locations and planning
field inspections. The current data footprint is an Ulster County, New York pilot;
the inference code accepts arbitrary road IDs, but that does not provide data or
validated accuracy outside the supplied geographic coverage.

**Recommendations require field verification.** Scores are uncalibrated ranking
signals, not probabilities. No 70% accuracy or precision tier is validated.

## Current system — 2026-10-15

| Component | Implementation | Status |
| --- | --- | --- |
| Main application | `web/`, `server/dev-server.js`, `src/culvert_ai/product_service.py` | Map-first location search, field checks and exports; frozen RF inference |
| Ranking-policy staging | `staging/ranking_policy.py`, `staging/ranking-policy.json` | Reversible raw RF versus dynamic ROUTE-Z on identical scored candidates; offline only |
| Hydrology staging | `staging/hydrology_sites.py`, `staging/hydrology_service.py`, `staging/web/` | Separate crossing-based map/API and experimental RF/HGB models; **not promoted** |

### Frozen main application

- Model: `models/flow_c_frozen_v1.joblib`, an 800-tree Random Forest with 48 ordered
  label-free GIS/terrain/hydrology features. Its manifest, schema and training-record
  sidecars are required; loading fails closed if their integrity checks fail.
- Inputs: server-side roads, mapped water, a DEM and precomputed hydrology layers.
  Road/route IDs group scores and identify outputs; they are not model features.
- Analysis CRS: EPSG:32618, metre units; display/export geometry: WGS84.
- Candidate geometry: first point at 37.5 m, then every 75 m along each road part.
  Ranking: dynamic ROUTE-Z, RF-score and candidate-ID tie-breaks, then global
  score-ordered 20 m suppression. No equal-road quotas or fitting during inference.
- Known defects remain in this unchanged baseline: 75 m proposals leave the
  20–60 m road-profile shoulders empty, producing three zero-valued features;
  the map draws all returned representatives even when its list is limited.
  These representatives are not independently confirmed sites.
- Map explores recommendations; Field checks records inspection results.
  Saving a field result does not retrain or modify the frozen estimator.

### Experimental hydrology map

The opt-in `hydrology-sites-v2` path separates dense internal road search from
physical-site proposals. It uses actual mapped/DEM drainage-road intersections,
transverse crossing angles, two-sided drainage geometry, and road/channel-aware
20 m consolidation. A nearby parallel river alone does not qualify a site.
Representatives retain actual intersection coordinates and traceable evidence.

Road profiles are measured independently at 10 m resolution on complete road
parts. Missing measurements remain unavailable until explicit training-only
imputation. Corrected feature definitions require a v2 contract and matching
experimental model; they are never fed into the original frozen estimator.

The map defaults to ranked physical sites, applies the same display budget to
map/list/export, and offers hidden-by-default internal candidates plus an optional
hydrology layer. Details include raw ranking score, model version and available
drainage/terrain evidence. This diagnostic map does not replace the main app's
field-observation workflow.

## Validation and limitations

The latest comparison reuses three fixed geographically held-out road groups and
preserved training manifests. These are retrospective experiments: all 714
inventory records had prior development exposure, and 701 directly influenced
the original RF training. There is no untouched independent test set.

Inventory matches in **three separate Top-250 budgets** (750 total slots):

| Pipeline | RF raw | RF dynamic ROUTE-Z | HGB raw |
| --- | ---: | ---: | ---: |
| Existing 75 m feature contract, held-out refits | 35 | 31 | 36 |
| Hydrology v2 consolidated crossing sites, held-out refits | 58 | 59 | 44 |

These are one-to-one inventory matches within 20 m, **not verified prediction
accuracy**. Different fold-model scores are not pooled into a single Top-250 list.

Candidate coverage is the deployment blocker. Within their held-out groups, the
75 m control reaches 344/705 in-footprint inventory records; dense internal search
reaches 700/705; strict crossing proposals reach only 63/705. Across all 714
records, native-geometry union coverage is respectively 348, 700 and 63. Nine
records outside the recovered road footprint remain in the audit.

The crossing queue misses many small/unmapped drainage locations. The smallest
derived network uses a 0.1 km² catchment threshold; 10 m DEM resolution,
road/stream registration, bridges, embankments and fragmented channels also limit
interpretation. An empty map does not mean no culverts exist. **Do not promote
this experiment to field deployment on the strength of its higher match count.**

Local reports, retained outside Git:

- [Hydrology implementation, ablations and before/after maps](reports/experiments/hydrology_driven_sites_20261015/REPORT.md)
- [Regression and browser/API verification](reports/experiments/hydrology_driven_sites_20261015/VERIFICATION.md)
- [Real-inventory held-out road audit](reports/experiments/real_inventory_heldout_roads_20261015/REPORT.md)
- [Matched-population ranking-policy control](reports/staging/ranking_policy_correction_20261015/REPORT.md)

Historical 41/250 POINTWISE-GB results, older 650/714 representative coverage,
and saved raw RF 33/250 matched-control results describe different score sources
or candidate populations. They are not interchangeable with the table above or
validation of today's main application. See the reports for lineage.

## Setup and required local assets

Declared minimums: Node.js 18+, Python 3.10+. The latest local regression run used
Python 3.13.7 and scikit-learn 1.9.0. Serialized estimators need a compatible
Python/library environment; dependency lower bounds alone do not guarantee that.

For a fresh environment:

```sh
npm ci --ignore-scripts
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Use the existing environment when it is already working. `scripts/python.sh`
selects this checkout's `.venv` and source tree. Normal `npm install` additionally
runs the Python bootstrap hook; it is not necessary just to start an installed app.

Models, inventory, rasters, raw GIS and experiment outputs are local assets,
not provided by a Git clone. For inference, provision:

```text
models/flow_c_frozen_v1.joblib
models/flow_c_frozen_v1.joblib.manifest.json
models/flow_c_frozen_v1.joblib.schema.json
models/flow_c_frozen_v1.joblib.training_records.csv
data/raw/roads.gpkg
data/raw/streams.gpkg
data/processed/hydrology/usgs_3dep_10m/
```

The hydrology directory supplies `dem_projected.tif`, flow accumulation, upstream
area, edge-flow flags, and drainage/crossing layers for 0.1, 0.5, 2 and 10 km²
thresholds. Hydrology staging also requires its separate RF/HGB v2 bundles under
`reports/experiments/hydrology_driven_sites_20261015/`.
The official inventory workbook is read-only research input, not a live inference
dependency. `web/data/` holds required static/API compatibility exports; it is not
the source of truth for frozen-model validation.

## Run without training

Main location-first application:

```sh
PORT=8093 npm start
```

Open `http://127.0.0.1:8093/`. The server defaults to 8080 when not configured and
may try higher ports if busy; check its startup message. Geolocation requires
browser permission. Coordinate entry, map selection and supported-area browsing
provide alternatives. Model/coverage errors disable analysis rather than invent
predictions.

Separate, read-only experimental hydrology map:

```sh
scripts/python.sh scripts/serve_hydrology_staging.py --port 8094
```

Open `http://127.0.0.1:8094/`. The first analysis measures real GIS features;
repeated identical requests are cached. The main app is not switched to this API.
See [staging usage](staging/README.md) for both experimental paths.

**Avoid training hooks for routine use.** `npm run dev` invokes `predev`, which
can run the legacy training/export pipeline when outputs are missing. Commands
such as `predict:actual`, `actual`, `pipeline:ulster`, `retrain:from-vercel`,
`demo`, `scripts/promote_flow_c.py`, and experiment runners can fit models
or overwrite generated artifacts. They are not normal startup/repair commands.
Do not retrain, promote, refresh protected data, or deploy without explicit approval.

## Tests and packaging

```sh
npm test
node --test scripts/test_hydrology_web.mjs
npm run build
```

Latest verified results: 174 Python tests passed (153 previous + 21 added), three
hydrology web tests passed, and existing UI/export, security and static/API build
checks passed. Twelve Python warnings are existing `unary_union` deprecations.
`npm run build` verifies assets/imports; it does not train, validate field accuracy,
or deploy. Models and full local GIS data are needed for all relevant tests.

Recreate the server-only frozen-model ZIP when needed:

```sh
scripts/python.sh scripts/package_flow_c_runtime.py
```

This generates ignored `dist/flow_c_frozen_v1-runtime.zip` from the audited model
and sidecars. It is a reproducible export, not authoritative model storage.

## Storage and deployment boundaries

Local field observations are stored in `data/processed/field_observations.geojson`.
The Vercel APIs support durable Blob storage when credentials are provisioned;
without it, do not assume deployed writes persist. Deployed observation writes
require `CULVERT_FEEDBACK_WRITE_TOKEN`; `.env.example` lists basic configuration.
Never put credentials in the README, handoff, browser exports or Git.

`vercel.json` packages static `web/` assets and JavaScript APIs, not the Python/GIS
nearby-analysis service. There is no Vercel JavaScript handler for
`/api/model-status` or `/api/analysis/nearby`. An equivalent Python-capable backend
and appropriate routing/assets are required for that full live flow;
a successful static build does not establish a working remote deployment.

The legacy cron retraining endpoint/dispatcher is still configured in source.
That is distinct from saving field observations, which no longer calls retraining
in `api/observations.js`. Remote configuration was not inspected or changed.

## Repository guide

- `src/culvert_ai/`: frozen inference, feature extraction, GIS utilities and retained
  historical training/label/export code.
- `web/`, `server/`, `api/`: main UI, local Python bridge and static/feedback APIs.
- `staging/`: reversible ranking adapter and isolated hydrology v2 implementation.
- `scripts/`, `tests/`: setup, audits, reproducible experiments and regression tests.
- `configs/`: geographic-source settings and historical pipeline inputs.
- `models/`, `data/`, `reports/`: preserve local scientific assets and provenance.
- `docs/product_application.md`: application/storage/deployment contracts.
- `docs/route_general_inference.md`: frozen API/schema details.
- `docs/model.md`, `docs/research_notes.md`: historical pipeline context, not current
  performance claims or instructions to retrain the frozen model.

`handoff.md` is intentionally local-only and ignored by Git. Caches, packaged
exports and temporary screenshots are not authoritative assets. Old research
scripts remain when needed for reproducibility; unused by the live server does
not mean safe to delete.
