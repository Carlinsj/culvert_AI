# Culvert AI: Ulster County Pilot

Last updated: 2026-06-22

Culvert AI is a geospatial machine-learning workflow for ranking likely culvert
locations in Ulster County, New York. It extracts coordinates from field reports,
filters them with GIS quality checks, trains supervised models from valid points,
and publishes a mobile Leaflet map for field review.

The goal is field-useful prediction: crews should be able to open the map, inspect
the highest-ranked locations first, add or delete their own observations, and feed
confirmed points back into the next model run.

## What This Is

- A reproducible geospatial ML pipeline for culvert discovery.
- A mobile-first field review UI backed by static GeoJSON plus observation APIs.
- A way to turn verified report coordinates and user-added observations into
  training labels.
- A research prototype that keeps validation honest with spatial holdout metrics.

## What This Is Not

- It is not an LLM-based location predictor.
- It does not invent coordinates, scores, or labels.
- It does not prove perfect unseen-culvert discovery from the current metrics.
- It is not yet a final engineering-grade culvert inventory.

LLMs can help review messy report text or flag uncertain extracted rows, but they
should not be used as the primary geospatial predictor. The predictor should learn
from coordinates, road and drainage geometry, density context, and terrain or
hydrology rasters when those rasters are available.

## Current Data State

Current dashboard export:

- `1,327` map rows in `web/data/findings.geojson`.
- `1,000` undiscovered discovery candidates.
- `327` known field matches.
- Bounds: Ulster County working extent from `web/data/summary.json`.

Current field-report extraction:

- `176` deduped report coordinate rows from `34` source files.
- `96` Team 4 rows extracted.
- `128` in-bound report coordinates used as exact candidates and QC positives.
- `48` coordinates rejected from training because they fall outside the current
  Ulster analysis extent.

Current training artifacts:

- `data/processed/field_report_culverts.gpkg`: extracted report coordinates.
- `data/processed/extracted_points_analysis.geojson`: coordinate QC output.
- `data/processed/high_confidence_training_points.gpkg`: report-derived positive labels.
- `data/processed/field_observations.geojson`: local user/ABU observations.
- `models/actual_ulster_field_report_model.joblib`: current supervised model bundle.

Generated markdown reports are intentionally written outside the repo at
`/private/tmp/culvert_extracted_points_analysis.md` so project documentation stays
capped at three Markdown files.

## Model Summary

The pipeline compares multiple model families and selects the best non-baseline
model by spatial holdout average precision first, then cross-validated average
precision and F1 as tie-breakers.

Model families currently compared:

- `baseline_prior`
- `regularized_logistic`
- `random_forest`
- `extra_trees`
- `spatial_regularized_extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`
- `balanced_hist_gradient_boosting`

Current selected model:

- Model: `hist_gradient_boosting`
- Rows: `5,173`
- Positive labels: `327`
- Negative labels: `4,846`
- Feature count: `28`
- QC coordinate training positives: `128`
- Random holdout AP: `0.719`
- Random holdout ROC AUC: `0.926`
- Spatial holdout AP: `0.563`
- Spatial holdout ROC AUC: `0.859`
- Spatial holdout P@10: `1.000`

Metric interpretation: spatial holdout is the metric to trust because field-report
labels are geographically clustered. The latest metrics improved because exact
field-report coordinates were added as candidate rows and labels. That is useful,
but it does not prove that every future high-ranked unknown point is a culvert.
The next major accuracy jump requires confirmed negative labels and better GIS data.

## Features Used

The model uses numeric geospatial features from candidate points, including:

- candidate source flags such as road-stream intersection, route sample, or
  field-report observed point,
- road-stream proximity,
- crossing angle and perpendicularity,
- road density around the point,
- stream density around the point,
- nearest road and nearest stream distance,
- matched route or named-road indicators,
- optional DEM-derived terrain features,
- optional flow-accumulation features,
- optional drainage-area features.

The feature builder automatically samples these optional raster files when present:

- `data/raw/dem.tif`
- `data/raw/flow_accumulation.tif`
- `data/raw/drainage_area.tif`

## Ranking Logic

The exported field-review ranking blends two signals:

- Interpretable GIS evidence from `score-unlabeled`.
- Supervised model probability from the selected classifier, when enough labels
  exist to train it.

Known field matches remain visible, but the discovery ranking prioritizes
not-yet-observed candidates first. This keeps the dashboard useful for field crews
instead of simply redisplaying already-known culverts at the top.

## Field UI

The web app is a Leaflet field-review dashboard.

Implemented UI behavior:

- Mobile map-first layout.
- List, Add, and Locate controls on mobile.
- Ranked candidate drawer.
- Candidate detail panel with coordinates and map links.
- ABU marker labels for confirmed user-added culverts.
- ABU tab listing all user-added confirmed culverts.
- Add user observation from the map.
- Delete incorrect user-added observations.
- Removed the old tracking-status text overlay.

Observation statuses supported by the API:

- `confirmed_culvert`
- `no_culvert`
- `uncertain`

Confirmed user-added culverts are displayed as `ABU` for "Added By User".

## Persistence

Local development stores observations in:

```text
data/processed/field_observations.geojson
```

The Vercel API supports Vercel Blob persistence when the environment is configured.
Relevant environment variables:

```text
BLOB_READ_WRITE_TOKEN
VERCEL_OIDC_TOKEN
BLOB_STORE_ID
CULVERT_OBSERVATIONS_BLOB_PATH
CULVERT_FINDINGS_BLOB_PATH
CULVERT_SUMMARY_BLOB_PATH
CULVERT_FEEDBACK_MATCH_RADIUS_M
```

If Blob is not configured, deployed feedback can be handled in memory/static mode,
but it will not be durable. To fold deployed observations back into local training,
pull them and retrain:

```bash
npm run retrain:from-vercel
```

## How The Pipeline Runs

The main production-like workflow is:

1. Download or reuse Census TIGER/Line roads and linear water.
2. Extract coordinates from configured field reports.
3. Build road-stream candidate points.
4. Add route-sampled candidates when extracted rows include route information.
5. Add valid field-report coordinates as exact candidates.
6. Analyze extracted points against roads, streams, candidates, and boundary.
7. Build high-confidence training positives.
8. Merge local or pulled ABU/user observations into training labels.
9. Build features from candidates, GIS layers, labels, and optional rasters.
10. Score all candidates with interpretable evidence.
11. Train and compare supervised models when enough positives and negatives exist.
12. Blend evidence and supervised probability into discovery rankings.
13. Export `web/data/findings.geojson`, `summary.json`, and `model_summary.json`.

## Setup

Requirements:

- Node.js `>=18`
- Python `>=3.10`
- npm

Install dependencies:

```bash
npm install
```

The `postinstall` script runs `scripts/bootstrap_python.sh`, which creates or
updates the local Python environment used by `scripts/python.sh`.

## Common Commands

Refresh the current Ulster model and web data:

```bash
npm run predict:actual
```

Run with explicit field-report paths:

```bash
FIELD_REPORTS_PATHS="/path/to/team2:/path/to/team4-report-1.pdf:/path/to/team4-report-2.pdf" npm run predict:actual
```

Refresh Census input files before predicting:

```bash
REFRESH_CENSUS_INPUTS=1 npm run predict:actual
```

Start the local app:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:8080
```

Run tests:

```bash
npm test
```

Verify deployable static/API assets:

```bash
npm run build
```

Pull Vercel Blob observations, retrain, and rebuild:

```bash
npm run retrain:from-vercel
```

## Important Scripts

- `scripts/run_actual_ulster_census_pipeline.sh`: main current pipeline.
- `scripts/ensure_actual_predictions.sh`: ensures web data exists before dev server startup.
- `scripts/bootstrap_python.sh`: creates the local Python environment.
- `scripts/write_model_summary.py`: writes the UI model summary JSON.
- `scripts/verify_web_build.js`: checks static web data and API imports.
- `scripts/pull_vercel_observations.js`: downloads Vercel Blob observations for retraining.

## Important Source Files

- `src/culvert_ai/field_reports.py`: PDF/DOCX coordinate extraction.
- `src/culvert_ai/point_analysis.py`: coordinate QC and training-point filtering.
- `src/culvert_ai/candidates.py`: road-stream, route, and merged candidate generation.
- `src/culvert_ai/features.py`: feature table construction and raster sampling.
- `src/culvert_ai/model.py`: model comparison, training, validation, and prediction.
- `src/culvert_ai/scoring.py`: evidence scoring and discovery ranking.
- `src/culvert_ai/observations.py`: local field observation merging.
- `src/culvert_ai/web_export.py`: GeoJSON and summary export for the UI.
- `server/dev-server.js`: local web server and observation endpoints.
- `api/observations.js`: Vercel observation API.
- `api/findings.js`: Vercel findings API.
- `api/summary.js`: Vercel summary API.
- `web/app.js`: Leaflet UI behavior.
- `web/styles.css`: responsive/mobile UI styling.

## Documentation Files

Project Markdown is intentionally limited to three files:

- `README.md`: main project handbook.
- `track.md`: current status and handoff notes.
- `docs/research_notes.md`: research framing and external communication notes.

## Accuracy Bottlenecks

The main bottleneck is not model choice. It is label and GIS quality.

Highest-priority improvements:

- Add confirmed `no_culvert` field checks as negative labels.
- Add official NYSDOT or county road centerlines.
- Add official hydrography, drainage, ditch, and structure layers.
- Add LiDAR-derived DEM, flow accumulation, and drainage area rasters.
- Validate by field route/day with precision at 10, 25, and 50.
- Track false positives and false negatives after each field session.

Why an LLM alone is the wrong model:

- Culvert prediction is spatial, not language-only.
- The model needs geometry, topology, terrain, hydrology, and measured labels.
- LLMs can hallucinate coordinates unless tightly constrained to extraction/QA.
- Research-grade performance needs reproducible geospatial features and validation.

Where an LLM can safely help:

- Extract candidate coordinates from messy reports with human/QC review.
- Flag coordinate rows with ambiguous context.
- Summarize field notes after the coordinates are already parsed.
- Draft reports about verified model outputs.

## Research-Grade Path

To make the system competitive with serious academic work, the next experiments
should focus on data and validation:

1. Build a route/day validation set with both positives and confirmed negatives.
2. Add higher-resolution GIS and hydrology rasters.
3. Compare spatial holdout, route holdout, and time-based holdout.
4. Report precision at field-review budgets such as top 10, 25, and 50.
5. Keep a strict audit trail from source report to extracted coordinate to label.
6. Treat ABU observations as training data only after they are confirmed and synced.

## Current Limitations

- Current Census road and water layers are coarse compared with engineering GIS.
- Out-of-bound extracted points are retained in analysis outputs but excluded from
  the Ulster model run.
- User observations are local unless Vercel Blob is configured.
- Current labels are clustered by field routes, so random validation is optimistic.
- Optional terrain/hydrology rasters are supported but not committed in this repo.

## Quick Health Check

After a fresh run, these files should exist and be non-empty:

```text
web/data/findings.geojson
web/data/summary.json
web/data/model_summary.json
reports/actual_ulster_field_report_metrics.json
models/actual_ulster_field_report_model.joblib
```

Use this command to verify the web export:

```bash
npm run build
```
