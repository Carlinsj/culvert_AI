# Culvert AI Track

Last updated: 2026-06-23

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
- Candidate universe after 20 m route sampling and report coordinates: 14,553 rows.
- Dashboard export: 1,324 rows.
- Discovery candidates shown: 1,000.
- Known field matches shown: 324.
- Local ABU/user observations: `data/processed/field_observations.geojson`.
- User-confirmed ABU positives are excluded from training by default because some field-added
  known points were inaccurate and not present in the report files.
- Live deployed `/api/observations` returned 0 persisted observations on 2026-06-23;
  the Vercel project still needs Blob persistence configured.
- Current strict field match radius is 20 m. A 50 m miss is not counted as correct.

## Current Model

Selected model: `spatial_regularized_extra_trees`.

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

- Spatial holdout average precision: 0.495.
- Spatial holdout P@10: 1.000.
- QC coordinate training positives: 128.

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
- Location tracking now updates the GPS marker in place, throttles nearby-list rerenders,
  and uses a single smooth first-location fly-to.
- Browser-local observations now attempt to sync to the server after Blob is configured.
- Default report inputs now come from `configs/field_report_inputs.txt` and are combined into
  one report-derived training set; team number is not used as a model feature.

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
3. Configure Vercel Blob, then reopen the app on the same phone to sync any browser-local marks.
4. Add DEM/flow/drainage rasters.
5. Replace Census roads/water with NYSDOT/county GIS.
6. Add route/day field validation reports: precision at 10, 25, 50 using 20 m match radius.
7. Add a simple UI summary for last retrain date, model name, training positives, ABU count, and spatial AP.
