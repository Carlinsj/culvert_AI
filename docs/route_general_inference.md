# Route-General FLOW-C Inference

## Purpose and fixed decisions

This is the production-facing inference architecture for applying Culvert AI to
an arbitrary road, a previously unseen route, or several roads at once. It does
not fit, tune, or select a model during prediction.

The following decisions are fixed:

- Candidate generation uses the frozen 75 m interval procedure. The historical
  650/714 figure describes score-ordered representatives from a different saved
  benchmark, not current candidate coverage. The October 15 audit measured
  348/714 native-geometry union coverage for current 75 m candidates; see
  [current validation and limitations](../README.md#validation-and-limitations).
- FLOW-C remains the predictive score.
- ROUTE-Z remains the route-context score and is recomputed from the current
  input roads at inference time.
- Consensus ranking (27 of 250) is rejected and is not part of this path.
- There are no route quotas. All physical sites compete in one global queue.
- Route filtering is a presentation step after scoring and global ranking.
- Whole-route holdout remains the correct validation design. This interface
  does not launch a new experiment or report new validation results.

## End-to-end flow

1. **Accept road geometry.** The input may contain one road feature, one route
   split into several features, or a multi-road network. A caller may name an
   opaque route-ID column. If no column is supplied, stable row identifiers are
   used.
2. **Generate candidates.** Every supplied road is sampled with the frozen
   75 m interval procedure and zero lateral offset. Broadening the eligible
   input from numbered routes to all supplied roads does not change candidate
   spacing or point placement.
3. **Build generic features.** Terrain, drainage, mapped crossing, density, and
   other label-free GIS fields are computed at each candidate. Route identity,
   known-culvert distance, labels, field feedback, and other leakage fields are
   forbidden model inputs.
4. **Score with frozen FLOW-C.** The predictor loads an audited model bundle and
   calls it in inference mode only. No route-specific fitting or retraining is
   performed.
5. **Compute dynamic ROUTE-Z.** FLOW-C scores are normalized within each route
   using candidates in the current request. Stored known-route parameters are
   never looked up.
6. **Create physical sites.** Candidates are sorted globally, then greedily
   suppressed within the frozen 20 m radius. This suppression is global, so
   nearby candidates assigned to different route IDs still become one site.
7. **Return one global queue.** Output is ordered by ROUTE-Z, with FLOW-C as the
   deterministic tie-break. A route may contribute many sites or none; no
   allocation quota is imposed.

## Required inputs

The high-level interface is `culvert_ai.inference.predict_culverts`:

```python
predictions = predict_culverts(
    road_geometry=roads,
    streams=streams,
    dem_path="data/raw/dem.tif",
    flow_c_model="models/flow_c_frozen_v1.joblib",
    route_id_column="corridor_id",       # optional grouping/output field
    flow_accumulation_path=None,          # optional generic raster
    drainage_area_path=None,              # optional generic raster
    landcover_path=None,                  # optional generic raster
    dem_crossings=crossing_layers,        # optional precomputed generic layers
)
```

Inputs are:

- road centerlines with a valid CRS;
- mapped streams/water with a valid CRS;
- a DEM raster;
- the exact generic rasters or crossing layers required by the frozen feature
  list; and
- an audited frozen FLOW-C bundle.

The production bundle contract is versioned and strict. The following abbreviated
example omits provenance and integrity metadata only for readability:

```python
{
    "model": fitted_flow_c_estimator,
    "feature_columns": ["... generic GIS/DEM fields ..."],
    "fill_values": {"feature_name": "full-training median"},
    "route_z_global": {"center": 0.42, "scale": 0.17},
}
```

The actual bundle must also contain the exact schema and its hash, model class
and parameters, preprocessing policy, training/code provenance, CRS and version
requirements, site/candidate definitions, training date, and ROUTE-Z fallback
provenance. The loader fails closed if any required metadata is absent, if the
bundle asks for route identity or an inference-forbidden label/leakage field, or
if the arbitrary-road feature pipeline cannot produce every frozen feature. It
does not silently substitute a different model.

## Promoted FLOW-C bundle

The selected specification was promoted on 2026-10-13 by fitting exactly one
final estimator. No alternative model, feature set, parameter set, or ranking
was evaluated during promotion.

- Bundle: `models/flow_c_frozen_v1.joblib`
- Specification: `hydro_c_plus_flow_c__nysdot_ulster_dem_hydrology_v8_20261005`
- Estimator: `sklearn.ensemble.RandomForestClassifier`
- Parameters: 800 trees, `min_samples_leaf=2`, `max_features="sqrt"`,
  `class_weight="balanced_subsample"`, seed 42, one worker; all remaining
  parameters are persisted explicitly in the bundle.
- Training records: 2,021 exact preserved rows: 946 positives and 1,075
  negatives. These comprise 696 official matches at the selected experiment's
  <=25 m training-label rule, 250 field-confirmed positives, 129 verified field
  negatives, five frozen proximity-derived negatives, and 941 frozen ordinary
  unlabeled hard negatives. The deployment site-consolidation rule remains 20 m.
- Features: the ordered 48-field FLOW-C schema in
  `models/flow_c_frozen_v1.joblib.schema.json`.
- Preprocessing: numeric float64 conversion, infinity-to-missing conversion,
  full-training feature medians, no scaling, and no categorical features.
- Feature CRS: EPSG:32618 with metre linear units; the audited hydrology grid is
  10 m USGS 3DEP-derived terrain/drainage.
- Candidate version: unchanged V13 interval baseline: 37.5 m first point, then
  every 75 m on each road line part, with zero lateral offset.
- Bundle SHA-256:
  `bdefae5bc12723e17487001826eea706fb48065c33ec1b13b15e892ad7347486`.
- Integrity manifest:
  `models/flow_c_frozen_v1.joblib.manifest.json`.

The bundle stores the exact ordered candidate IDs and their hash, label/source
counts, source-artifact hashes, estimator class and parameters, environment and
Git version, preprocessing contract, expected dtypes, CRS requirements, and
candidate/hydrology versions. The external schema and training-record copy are
also hashed. The loader checks these artifacts before deserializing and then
checks the embedded schema, fitted estimator feature order, class order,
parameters, preprocessing, and fallback statistics.

For statistically unstable routes, the bundle contains a global FLOW-C center
of `0.6095600570243888` and population scale of `0.22200026849259258`. These
were calculated from 51,417 unlabeled whole-route out-of-fold FLOW-C scores.
Ground-truth matches and match distances were not used to compute them.

## Dynamic ROUTE-Z and fallbacks

ROUTE-Z never uses a stored identity-specific parameter. For each route in the
current request:

| Current-route condition | Normalization |
| --- | --- |
| At least 8 finite scores and nonzero standard deviation | Current-route mean and population standard deviation |
| 4–7 finite scores, or unstable standard deviation, with a usable IQR | Current-route median and `IQR / 1.349` |
| Too few scores or zero variance, with frozen global statistics in the bundle | Frozen global FLOW-C center and scale |
| Too few scores or zero variance, without usable global statistics | Neutral ROUTE-Z of 0; FLOW-C breaks ties |

Every output row records the method, candidate count, center, and scale used.
Missing or non-finite individual scores normalize to zero rather than producing
an invalid rank.

The robust and neutral paths keep short roads processable without inventing
route knowledge. Frozen global fallback statistics, when present, are shared
statistics from the approved training population—not values keyed to a named
route.

## Multi-road behavior

All roads are scored before any route filter is applied. Dynamic ROUTE-Z is
computed independently within each current route, then the candidates are
combined, globally sorted, and deduplicated into 20 m physical sites. This
ordering matters: filtering a route first would change its competitive context
and could create duplicate sites where roads meet.

`filter_predictions_by_route` may be used after prediction. It preserves the
original `global_rank`; it does not rerank the subset.

## Output contract

The field-ready result includes:

- `global_rank`
- `route_id`
- `latitude`, `longitude`
- `candidate_id`
- `physical_site_id`
- `flow_c_score`
- `route_z_score`
- `final_priority` (the ROUTE-Z value used for ordering)
- ROUTE-Z method, count, center, and scale audit fields
- geometry and available road provenance fields

The command-line equivalent is:

```bash
scripts/python.sh -m culvert_ai.cli predict-route-general \
  --roads path/to/roads.gpkg \
  --streams path/to/streams.gpkg \
  --dem path/to/dem.tif \
  --model models/flow_c_frozen_v1.joblib \
  --route-id-column corridor_id \
  --output data/processed/route_general_predictions.gpkg \
  --csv-output data/processed/route_general_predictions.csv
```

Optional generic raster and DEM-crossing arguments are exposed by the command.
If short-route global fallback values are not stored in the bundle, they may be
supplied as a center/scale pair. Supplying only one is rejected.

## Invariants

- The prediction path contains no `fit`, parameter search, or model selection.
- Route IDs are grouping and output metadata only.
- Known culvert inventories and field feedback are not prediction inputs.
- The candidate interval and 20 m site definition are frozen.
- FLOW-C and dynamic ROUTE-Z are the only ranking signals in this interface.
- A route-specific result is always a view of the globally ranked output.
