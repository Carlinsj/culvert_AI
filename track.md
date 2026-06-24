# Culvert AI Track

Last updated: 2026-06-24

## Current Goal

Build a field-ready culvert discovery workflow for Ulster County:

- extract verified field-report coordinates,
- use valid coordinates as model training points,
- rank likely undiscovered culvert locations,
- let users add/delete ABU points from the map,
- keep model metrics honest with spatial validation.

## Current Data State

- Field-report coordinate rows: 176.
- Source files: 34.
- Team 4 rows extracted: 96.
- In-bound coordinates used as exact candidates/training positives: 128.
- Out-of-bound coordinates rejected: 48.
- Candidate universe after 20 m route sampling, report coordinates, and field observations: 14,565 rows.
- Dashboard export: 1,210 rows.
- Discovery candidates shown: 1,000.
- Known field matches shown: 210.
- Local ABU/user observations: `data/processed/field_observations.geojson`.
- Deployed observation pull on 2026-06-24 returned 19 observations:
  13 confirmed culverts, 6 no-culvert labels, and 12 unique confirmed positives
  after duplicate field IDs are deduped.
- User-confirmed ABU positives are included in retraining by default now that field
  crews are intentionally adding missed culverts. Set
  `INCLUDE_FIELD_OBSERVATIONS_AS_POSITIVES=0` only for a questionable batch.
- `scripts/pull_vercel_observations.js` can pull deployed observations through
  Blob credentials or the public `/api/observations` endpoint.
- Deployed observation uploads now queue automatic retraining when
  `CULVERT_RETRAIN_WEBHOOK_URL` or `GITHUB_RETRAIN_TOKEN`/`GITHUB_REPOSITORY` is
  configured. The Python training still runs in an external worker with
  `npm run retrain:from-vercel`, then deploys rebuilt `web/data` files.
- Current strict field match radius is 10 m. A 50 m miss is not counted as correct.
- A confirmed ABU point farther than 10 m from its nearest predicted candidate is
  now stored as the true positive location and can also mark that predicted
  candidate as a missed/negative training label.

## Current Model

Selected model: `hist_gradient_boosting`.

Model comparison includes:

- baseline prior
- regularized logistic regression
- random forest
- extra trees
- spatial-regularized extra trees
- gradient boosting
- histogram gradient boosting
- balanced histogram gradient boosting

Current measured metrics:

- Spatial holdout average precision: 0.635.
- Spatial holdout P@10: 1.000.
- Training point rows: 140.
- Class counts: 210 positives, 14,355 negatives.

Interpretation: the metrics are much stronger after adding exact field-report coordinate candidates.
That is useful for learning from known points, but it does not prove perfect discovery of unseen
culverts. The next honest improvement requires field-confirmed negatives and better GIS/terrain data.

## UI State

Implemented:

- Mobile-friendly map controls: List, Add, Locate.
- Removed visible tracking text overlay.
- User-added confirmed culverts display as `ABU`.
- ABU tab in the drawer lists all confirmed user-added points.
- ABU points can be selected, opened in Google Maps, or deleted.
- User-added observation deletion works through `/api/observations?id=...`.
- Deleting an observation removes its deployed feedback and refreshes the served
  ranking; the deleted point is also absent from the next Blob-backed retrain.
- Location tracking now updates the GPS marker in place, throttles nearby-list rerenders,
  and uses a single smooth first-location fly-to.
- Mobile prediction and known-point markers now render through one Leaflet canvas layer,
  not hundreds of DOM markers.
- Mobile prediction circles now show score labels again through a separate canvas
  overlay, keeping the score display without reintroducing DOM marker lag.
- Mobile camera movement now uses short pan/fly animations and defers full list rerenders
  while the drawer is closed.
- Browser-local observations now attempt to sync to the server after Blob is configured.
- Default report inputs now come from `configs/field_report_inputs.txt` and are combined into
  one report-derived training set; team number is not used as a model feature.
- Observation save/delete responses include a `retraining` object showing whether
  automatic retraining was queued, debounced, unchanged, failed, or not configured.

## Bottlenecks

- Need negative labels from field checks.
- Need Vercel Blob attached and `BLOB_READ_WRITE_TOKEN` set so ABU/no-culvert marks persist.
- Need official NYSDOT/county road and drainage layers.
- Need DEM and hydrology rasters:
  - `data/raw/dem.tif`
  - `data/raw/flow_accumulation.tif`
  - `data/raw/drainage_area.tif`
- Current Census road/water data is good enough for prototype ranking but not final research quality.
- Team 4 includes out-of-bound points; they are retained in extracted files but excluded from Ulster training.
- Spatial holdout remains the main metric because field reports are route-clustered.

## How To Resume

Install:

```bash
npm install
```

Refresh predictions/model:

```bash
npm run predict:actual
```

Run with explicit report sets:

```bash
FIELD_REPORTS_PATHS="/Users/Carli/Downloads/Team No. 2-selected (1):/path/to/team4.pdf" npm run predict:actual
```

Start app:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:8080
```

Verify:

```bash
npm test
npm run build
```

## Next Best Work

1. Collect confirmed `no culvert` labels and train with negatives.
2. Add the missing third team's report path to `configs/field_report_inputs.txt` once available.
3. Configure Vercel Blob and retraining worker env vars, then reopen the app on the same phone
   to sync any browser-local marks.
4. Add DEM/flow/drainage rasters.
5. Replace Census roads/water with NYSDOT/county GIS.
6. Add route/day field validation reports: precision at 10, 25, 50 using 10 m match radius.
7. Add a simple UI summary for last retrain date, model name, training positives, ABU count, and spatial AP.
