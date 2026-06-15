# Team 3 Field Report Integration

Input archive:

- `/Users/Carli/Downloads/Team No. 3.zip`

What the importer extracts:

- PDF and DOCX daily field reports.
- Route, report date, latitude, longitude, and culvert IDs where available.
- Coordinates are interpreted as New York latitude/longitude even when the table header says
  longitude/latitude, because values such as `42.081240N 74.319991W` are latitude first.

Current extracted labels:

- 57 unique field-observed points from 11 reports.
- 21 points fall inside the Ulster County boundary used by this project.
- The enhanced pipeline builds route-sampled candidates every 75 m along reported routes such as
  NY28 and NY32, then uses the field points as labels.
- The current training table has 4,048 candidates: 36 positive matches and 4,012 negatives.

Model interpretation:

- This is now a field-report-enhanced model, not just a generic road-water crossing score.
- The model is still early because there are only 36 positive matched candidates.
- Spatial holdout performance is the honest metric to watch; it is much more important than random
  holdout on a small, route-clustered dataset.
- The next quality jump should come from adding DEM-derived drainage/terrain features and recording
  false positives from field searches.

Useful commands:

```bash
npm run import:reports
npm run predict:actual
npm run dev
```

Key outputs:

- `data/processed/field_report_culverts.gpkg`
- `data/processed/field_report_culverts.csv`
- `data/interim/actual_ulster_route_candidates.gpkg`
- `models/actual_ulster_field_report_model.joblib`
- `reports/actual_ulster_field_report_metrics.json`
- `reports/actual_ulster_field_report_feature_importance.csv`
