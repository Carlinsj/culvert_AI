# Culvert AI product application

## Architecture

The browser is a presentation client. It starts on a full-screen map, requests
the user's location, displays the returned final physical sites, stores field
observations separately, and exports an explicit safe-field subset. It contains
no prediction or ranking logic.

The primary endpoint is `POST /api/analysis/nearby`. It accepts latitude,
longitude, and a search radius, selects and clips roads from the server-side
road network, validates terrain coverage and the frozen bundle, and invokes the
existing `predict_culverts()` boundary. It returns only the globally ranked,
20 m-consolidated physical sites. No raw candidates or evaluation labels cross
the API boundary. `GET /api/model-status` verifies availability before the UI
enables analysis.

Users do not upload road files in the primary UI. Location may come from browser
geolocation, a clicked map point, or entered latitude/longitude. The server-side
road layer is transformed to EPSG:32618 for analysis and returned in WGS84 for
display. The lower-level upload service remains available for research tooling,
where GeoJSON must explicitly declare its CRS and GeoPackage or zipped
shapefiles are also supported.

## Frozen-model deployment

`models/*` is intentionally excluded from Git. A deployment must provision the
model and its integrity sidecars on the server before startup. Build the
validated server-only package with:

```bash
scripts/python.sh scripts/package_flow_c_runtime.py
```

The resulting ignored archive at `dist/flow_c_frozen_v1-runtime.zip` includes:

- `flow_c_frozen_v1.joblib`
- `flow_c_frozen_v1.joblib.manifest.json`
- `flow_c_frozen_v1.joblib.schema.json`
- `flow_c_frozen_v1.joblib.training_records.csv`

The training-record sidecar is retained for provenance and integrity checking,
but it is never served to ordinary UI users. The model-status endpoint fails
closed when any file is missing or altered.

The current static Vercel deployment cannot execute this Python/GDAL/raster
analysis service or securely ship the large server-only artifacts. Production
analysis therefore requires a Python-capable service/container with the model,
DEM, streams, and hydrology assets mounted. The static UI can call that service
through the same API contract.

## Research-rule distinction

The 2026-10-14 precision-policy audit found no supported 70% tier in the saved
ranking behind the historical 41/250 benchmark. The UI labels recommendations
as field review; exported CSV/GeoJSON rows carry `recommendation_tier: review`.
Existing priorities and ordering are unchanged. No high-confidence threshold
is enabled. See `reports/research/precision_policy_20261014/REPORT.md`.

The provenance audit also established that historical ROUTE-Z scores normalize
V18 POINTWISE-GB predictions, whereas production normalizes frozen Random
Forest FLOW-C predictions. The historical 41/250 result must not be presented
as validation of the production score. The older 650/714 figure measures
distinct matches after FLOW-C-ordered 20 m consolidation, not raw candidate
coverage. Previously published summaries using those labels are superseded
by the precision-policy provenance audit; prior research files are retained.

The selected historical FLOW-C training specification used a positive-label
rule of **25 m or less**. The deployed physical-site consolidation and reported
candidate-coverage evaluation use **20 m**. These are different rules.

The final frozen model retains the selected hard-negative training records from
the historical specification. Those records are not all field-confirmed
negatives and must not be described that way.
