# Culvert AI: Ulster County Pilot

Starter research repo for predicting likely culvert locations before field inspections in the
Ulster County side of the Poughkeepsie/Hudson Valley region.

The practical field problem is simple: inspection teams already have a way to record culvert
condition once a culvert is found, but they lose time locating the culvert. This project builds a
repeatable GIS and machine-learning workflow that ranks likely culvert locations using road-stream
crossings, topography, land cover, and previously registered culverts. Poughkeepsie itself is in
Dutchess County, so this repo scopes the pilot to nearby Ulster County communities and corridors west
of the Hudson River.

## What This Repo Does

- Generates candidate culvert points where roads cross or nearly cross streams/drainage lines.
- Builds features from DEM/topography, terrain roughness, road-stream crossing angle, land cover
  rasters, road density, stream density, and known culvert inventories.
- Compares multiple model families and selects the strongest non-baseline model by cross-validated
  average precision.
- Reports random holdout metrics, spatial holdout metrics, top-k field utility, and feature
  importance.
- Produces a ranked map layer that field teams can open in QGIS, ArcGIS, Google Earth, or export to
  a field app.
- Includes an Ulster-style synthetic demo dataset so the full workflow can run before real NYSDOT
  data is added.
- Provides a region filter command and boundary file for the Ulster County/Poughkeepsie-area pilot.
- Supports a no-known-culvert workflow for the real field problem: ranking likely locations before
  crews walk the road.

## Quick Start

Recommended local workflow:

```bash
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:8080
```

`npm install` creates/updates the Python virtual environment. On first run, `npm run dev` creates
actual Ulster County predictions from Census TIGER/Line roads and linear-water data if they are
missing, then starts one local server that serves the map frontend and backend API endpoints from the
same port.

Useful npm commands:

```bash
npm run predict:actual    # download actual Ulster TIGER roads/linear-water and refresh web/data
npm run dev               # ensure actual predictions exist, then start backend/frontend
npm run import:reports    # extract field-report culvert coordinates from the Team 3 ZIP
npm run pipeline:ulster   # run the real no-inventory Ulster workflow after data/raw is populated
npm test                  # run Python tests
```

`npm run demo` still exists only as a software smoke test. Do not use it as evidence for the
research proposal or field plan.

The direct Python workflow is still available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python -m culvert_ai.cli download-osm --output-dir data/raw
```

Or use the included Makefile:

```bash
make setup
make demo
```

The actual Census TIGER/Line prediction workflow writes:

- `data/raw/roads.gpkg`
- `data/raw/streams.gpkg`
- `data/raw/osm_download_metadata.json`
- `data/interim/actual_ulster_candidates.gpkg`
- `data/processed/actual_ulster_unlabeled_features.gpkg`
- `data/processed/actual_ulster_unlabeled_predictions.gpkg`
- `data/processed/actual_ulster_unlabeled_predictions.csv`
- `data/processed/actual_ulster_google_earth_review.kml`
- `web/data/findings.geojson`
- `web/data/summary.json`

If you add a real DEM at `data/raw/dem.tif`, the same command includes slope, terrain roughness,
local relief, topographic position, and valley-depth features automatically.

The synthetic demo writes:

- `data/ulster_demo/raw/roads.gpkg`
- `data/ulster_demo/raw/streams.gpkg`
- `data/ulster_demo/raw/known_culverts.gpkg`
- `data/ulster_demo/raw/demo_dem.tif`
- `data/ulster_demo/interim/candidates.gpkg`
- `data/ulster_demo/processed/training_features.gpkg`
- `data/ulster_demo/processed/unlabeled_predictions.gpkg`
- `data/ulster_demo/processed/unlabeled_predictions.csv`
- `data/ulster_demo/processed/google_earth_review.kml`
- `data/ulster_demo/processed/predictions.gpkg`
- `data/ulster_demo/models/culvert_model.joblib`
- `data/ulster_demo/reports/metrics.json`
- `data/ulster_demo/reports/feature_importance.csv`

## Actual Prediction Workflow When Culverts Are Unknown

Use this first when you do not have a NYSDOT or county culvert inventory:

```bash
npm install
npm run dev
```

This downloads actual Census TIGER/Line roads and linear-water features for Ulster County, generates
road-drainage crossing candidates, ranks likely culvert locations, writes Google Earth review files,
and refreshes the website map. It is not synthetic.

## Real Data Workflow With Project GIS Files

This is the workflow to use when there is no reliable culvert inventory for the target area:

```bash
scripts/run_ulster_unlabeled_pipeline.sh
```

It creates:

- `data/processed/ulster_unlabeled_features.gpkg`
- `data/processed/ulster_unlabeled_predictions.gpkg`
- `data/processed/ulster_unlabeled_predictions.csv`
- `data/processed/ulster_google_earth_review.kml`

Open the KML in Google Earth, inspect the top-ranked points, and send field crews to the strongest
evidence locations first.

## Web Dashboard

Start the combined backend/frontend dev server:

```bash
npm run dev
```

Then open `http://127.0.0.1:8080`.

Export the latest ranked findings for the browser dashboard:

```bash
culvert-ai export-web \
  --predictions data/processed/ulster_unlabeled_predictions.gpkg \
  --output-dir web/data
```

For the synthetic demo data:

```bash
make web-data
```

Start the local website:

```bash
scripts/serve_web.sh
```

Then open:

```text
http://127.0.0.1:8080
```

The dashboard uses Leaflet with OpenStreetMap tiles. It shows ranked candidates, priority filters,
evidence summaries, Google Earth links, and Google Maps links.

## Vercel Deployment

This repo deploys to Vercel as a static dashboard from `web/` plus lightweight API functions in
`api/`. The checked-in `vercel.json` skips the local Python bootstrap, validates the static web
assets with `npm run build`, and lets the Vercel Functions serve feedback-adjusted findings from
Vercel Blob.

Create a Vercel Blob store for the project and set `BLOB_READ_WRITE_TOKEN` in Vercel. The defaults in
`.env.example` store:

- field observations at `culvert-ai/field_observations.geojson`
- feedback-adjusted findings at `culvert-ai/findings.geojson`
- feedback-adjusted summary stats at `culvert-ai/summary.json`

Before deploying, refresh the predictions locally if needed and commit the generated dashboard data:

```bash
npm run predict:actual
npm run build
git add .gitignore .env.example README.md package.json package-lock.json vercel.json api scripts/verify_web_build.js scripts/pull_vercel_observations.js web/app.js web/data/findings.geojson web/data/summary.json
```

On Vercel, saving field feedback writes to Blob, refreshes the served `/api/findings` ranking, and
marks confirmed/denied culverts so the dashboard reflects the update immediately. The full Python
supervised model retrain still runs locally because it depends on the geospatial pipeline. To fold
deployed feedback into that model and refresh the deployable `web/data` files:

```bash
BLOB_READ_WRITE_TOKEN=... npm run retrain:from-vercel
```

If another region has verified culvert locations, use transfer learning:

```bash
scripts/run_transfer_from_external_region.sh
```

That trains on `data/external/source_*` files and predicts on the Ulster feature table.

## Supervised Workflow If Known Culverts Become Available

Put your project data under `data/raw/`:

```text
data/raw/
  roads.gpkg              # road centerlines
  streams.gpkg            # streams, drainage lines, or hydrography
  known_culverts.gpkg     # existing culvert inventory with point locations
  dem.tif                 # optional USGS 3DEP or LiDAR-derived DEM
  landcover.tif           # optional land cover raster
```

The repo includes a rough working boundary at
`configs/regions/ulster_poughkeepsie_pilot.geojson`. Replace it with an official Ulster County or
NYSDOT project boundary when one is available.

Then run the Ulster-specific pipeline:

```bash
scripts/run_ulster_pipeline.sh
```

That script runs these steps:

```bash
culvert-ai filter-region \
  --input data/raw/roads.gpkg \
  --output data/interim/ulster_roads.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

culvert-ai filter-region \
  --input data/raw/streams.gpkg \
  --output data/interim/ulster_streams.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

culvert-ai filter-region \
  --input data/raw/known_culverts.gpkg \
  --output data/interim/ulster_known_culverts.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

culvert-ai build-candidates \
  --roads data/interim/ulster_roads.gpkg \
  --streams data/interim/ulster_streams.gpkg \
  --output data/interim/ulster_candidates.gpkg

culvert-ai build-features \
  --candidates data/interim/ulster_candidates.gpkg \
  --known-culverts data/interim/ulster_known_culverts.gpkg \
  --roads data/interim/ulster_roads.gpkg \
  --streams data/interim/ulster_streams.gpkg \
  --dem data/raw/dem.tif \
  --landcover data/raw/landcover.tif \
  --output data/processed/ulster_training_features.gpkg

culvert-ai train \
  --features data/processed/ulster_training_features.gpkg \
  --model-output models/ulster_culvert_model.joblib \
  --metrics-output reports/ulster_metrics.json \
  --importance-output reports/ulster_feature_importance.csv \
  --model-family auto \
  --spatial-block-size-m 2500

culvert-ai predict \
  --features data/processed/ulster_training_features.gpkg \
  --model models/ulster_culvert_model.joblib \
  --output data/processed/ulster_predictions.gpkg \
  --csv-output data/processed/ulster_predictions.csv
```

Open `data/processed/ulster_predictions.gpkg` in QGIS or ArcGIS and sort by
`culvert_probability` or `priority_rank`.

## Tests

```bash
npm test
```

Or run Python directly:

```bash
pip install -e ".[dev]"
pytest tests
```

## Current Status

What is built:

- A research repo scoped to the Ulster County side of the Poughkeepsie/Hudson Valley field area.
- A live Census TIGER/Line-based actual prediction workflow: `npm run predict:actual`.
- A no-known-culvert ranking workflow for when the team does not have local inventory labels.
- Candidate generation from road-stream crossings and near crossings.
- Topographic/terrain features from DEMs.
- Optional supervised training when verified culvert labels become available.
- KML export for Google Earth review.
- A Leaflet web dashboard served by `npm run dev`.
- A local ignored `PROJECT_TRACKING.md` for scratch progress notes.
- A scoring improvement plan at `docs/scoring_improvement_plan.md`.
- Field report integration notes at `docs/field_report_integration.md`.

What remains:

- Run `npm run predict:actual` for a first-pass actual map from Census TIGER/Line data.
- Add a real DEM at `data/raw/dem.tif` to improve topographic ranking.
- Replace Census roads/linear-water with NYSDOT or county GIS files when the team provides them.
- Review top-ranked points in Google Earth before field visits.
- Record field-confirmed culverts and false positives.
- Retrain/evaluate a supervised model once verified labels exist.

## Data Sources To Ask For

For the professor or hydraulics team, ask whether they can provide:

- Verified culvert inventory points for Ulster County, especially the eastern/southern side near
  the Poughkeepsie-area field work.
- Road centerlines for Ulster County.
- Hydrography or drainage lines for Ulster County.
- DEM/topographic data for Ulster County, preferably LiDAR-derived where available.
- Field observations from the current summer internship in or near Ulster County.
- Any existing inspection app exports, including photos/video metadata and coordinates.

## Suggested Pilot

Start with the Ulster County side of the Poughkeepsie/Hudson Valley field area. Focus first on
Highland/Lloyd, Esopus, New Paltz, Marlboro, Plattekill, Rosendale, and south/east Kingston approach
corridors. Use existing known culverts as positive examples, generate road-stream crossing
candidates, train the model, and validate whether high-ranked points match field observations.

## Important Limits

This is a field-planning and research workflow, not a final engineering decision system. When there
are no known culvert labels, the output is a ranked evidence score, not a trained local classifier.
It should help inspectors prioritize where to look. It still needs field validation, QA by the
project team, and review by the hydraulics group.
