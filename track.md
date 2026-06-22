# Culvert AI Track

Last updated: 2026-06-22

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
- Candidate universe after adding report coordinates: 5,173 rows.
- Dashboard export: 1,327 rows.
- Discovery candidates shown: 1,000.
- Known field matches shown: 327.
- Local ABU/user observations: `data/processed/field_observations.geojson`.

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

- Spatial holdout average precision: 0.563.
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

## Bottlenecks

- Need negative labels from field checks.
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
2. Add DEM/flow/drainage rasters.
3. Replace Census roads/water with NYSDOT/county GIS.
4. Add route/day field validation reports: precision at 10, 25, 50.
5. Add a simple UI summary for last retrain date, model name, training positives, ABU count, and spatial AP.
