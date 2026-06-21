# Field Report Integration

Default input folder:

- `/Users/Carli/Downloads/Team No. 2-selected (1)`

What the importer extracts:

- PDF and DOCX daily field reports.
- Route, report date, latitude, longitude, and culvert IDs where available.
- Coordinates are interpreted as New York latitude/longitude even when the table header says
  longitude/latitude, because values such as `42.081240N 74.319991W` are latitude first.

Current extracted labels:

- 80 unique Team 2 field-observed points from 13 reports.
- The enhanced pipeline builds route-sampled candidates every 75 m along reported routes such as
  NY9G, NY32, NY42, NY212, and NY375, then uses the field points as labels.
- The current training table has 3,785 candidates and 83 positive matches after merging the
  dashboard-confirmed observation.

LLM-assisted label QA:

- Use `npm run prepare:llm-label-review` to write
  `data/processed/field_report_llm_review_queue.jsonl`.
- Give that JSONL to an LLM for validation/correction of extracted coordinates, route IDs, and
  culvert IDs. The LLM should only accept rows supported by the report context.
- Import reviewed labels with `culvert-ai import-llm-reviewed-labels`.
- If `data/processed/field_report_llm_reviewed_culverts.gpkg` exists, `npm run predict:actual`
  uses those reviewed labels as the cleaner training source.

Model interpretation:

- This is now a field-report-enhanced model, not just a generic road-water crossing score.
- The model is still early because there are only 36 positive matched candidates.
- Spatial holdout performance is the honest metric to watch; it is much more important than random
  holdout on a small, route-clustered dataset.
- The next quality jump should come from adding DEM-derived drainage/terrain rasters and recording
  false positives from field searches.

Useful commands:

```bash
npm run import:reports
npm run prepare:llm-label-review
npm run predict:actual
npm run dev
```

Key outputs:

- `data/processed/field_report_culverts.gpkg`
- `data/processed/field_report_culverts.csv`
- `data/processed/field_report_llm_review_queue.jsonl`
- `data/processed/field_report_llm_reviewed_culverts.gpkg`
- `data/interim/actual_ulster_route_candidates.gpkg`
- `models/actual_ulster_field_report_model.joblib`
- `reports/actual_ulster_field_report_metrics.json`
- `reports/actual_ulster_field_report_feature_importance.csv`
