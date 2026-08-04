# Culvert Prediction Model

Last updated: 2026-07-06

This file explains how the culvert prediction model is built, how field labels
are used, and how the final map score is calculated.

## Short Version

The system does not predict culverts from text. It predicts from geospatial
evidence around candidate map points.

The workflow has three scoring layers:

1. Candidate generation creates places worth checking, mostly road-stream
   crossings, route samples, field-report points, and field-observed points.
2. A supervised machine-learning model estimates `culvert_probability` from
   numeric GIS features when enough labels exist.
3. A field-review ranking combines supervised probability with interpretable GIS
   evidence into `discovery_score`, then pushes already-known culverts behind
   unchecked candidates.

The deployed map is a review queue, not a proof that a point is definitely a
culvert. High scores mean "check here first."

## Current Run

The current rebuilt model artifacts report:

- Selected model: `random_forest`
- Candidate rows scored: `80,620`
- Explicit positive labels: `244`
- Explicit negative labels: `237`
- Unreviewed rows excluded from training: `80,139`
- Training point rows: `250`
- Feature count: `34`
- Spatial holdout average precision: `0.926`
- Spatial holdout precision at 10: `1.000`
- Spatial holdout precision at 25: `1.000`
- Candidate coverage before field-facing filtering: `102 / 122` confirmed
  culverts have a candidate within `15 m`

The source of truth for the latest run is `web/data/model_summary.json`.

## Training Labels

The model target column is `is_culvert`.

Positive labels come from:

- verified field-report culvert coordinates,
- confirmed CBU/user-added observations pulled from Vercel,
- confirmed field observations within the strict match radius.

Negative labels come from explicit field decisions:

- field observations marked `no_culvert`,
- missed-prediction labels when a confirmed field culvert proves that a specific
  predicted candidate was outside the hit radius.

Unreviewed candidates are unlabeled, not negative. They are scored after training
but excluded from model fitting and holdout metrics. This prevents tens of
thousands of unchecked locations from overwhelming the comparatively small set
of verified field decisions.

The current strict match radius is `10 m`. A prediction 50 m from a confirmed
field culvert is a miss, not a correct prediction.

Confirmed CBU/user-added positives are included by default in retraining. Set
`INCLUDE_FIELD_OBSERVATIONS_AS_POSITIVES=0` only for a questionable field batch
that should be displayed on the map but not learned yet.

## Candidate Generation

The model does not search every coordinate in the county. It first builds a
candidate table of plausible locations:

- exact or nearest road-stream crossing candidates,
- points sampled along named routes when field reports mention routes,
- valid field-report coordinates inserted as exact candidate rows,
- confirmed field-observation coordinates inserted as exact candidate rows,
- existing candidates that can be matched to user observations.

This candidate table is important: the model ranks candidates that exist in the
table. If a real culvert location is not represented by a candidate point, the
supervised model cannot rank that exact location until the candidate-generation
step is improved or a field/user point adds it.

## Feature Table

For every candidate, `src/culvert_ai/features.py` builds numeric features. The
main feature groups are:

- road-stream distance and road-stream proximity,
- crossing angle and perpendicularity,
- road density around the point,
- stream density around the point,
- nearest-road and nearest-stream distance,
- road and stream tag flags such as bridge, tunnel, or culvert,
- whether road, stream, or route names are present,
- field-report support and distance to known culvert labels,
- optional DEM terrain features, including a composite
  `dem_culvert_terrain_score`,
- approved-known culvert context from document-approved and CBU/user-confirmed
  positives, including same-route and nearby-corridor support,
- optional flow-accumulation features,
- optional drainage-area features.

The actual Ulster pipeline downloads a USGS 3DEP 1 arc-second DEM to
`data/raw/dem.tif` when it is missing. Use `REFRESH_DEM=1` to rebuild it or
`DEM_RESOLUTION=13` to request larger 1/3 arc-second USGS tiles.

Raster files are sampled when these files exist:

```text
data/raw/dem.tif
data/raw/flow_accumulation.tif
data/raw/drainage_area.tif
```

Missing numeric model inputs are filled with `-9999.0` before prediction.

## Supervised Model

Training happens in `src/culvert_ai/model.py`.

The training code compares several model families:

- `baseline_prior`
- `regularized_logistic`
- `random_forest`
- `extra_trees`
- `spatial_regularized_extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`
- `balanced_hist_gradient_boosting`
- `soft_voting_ensemble`

The actual Ulster pipeline defaults to `auto`, which compares all candidate
families using the spatial holdout before fitting the final model. Set
`CULVERT_MODEL_FAMILY` to a specific family only for a controlled experiment.

Only numeric feature columns are used. The code excludes target, coordinate,
label, rank, and already-computed score columns so the model does not train on
the answer or on UI ranking outputs. Columns beginning with `approved_known_`
or `nearest_known_` are also excluded from supervised training because they are
derived from approved positive labels. They are used only by the interpretable
evidence ranking.

Model selection uses this priority:

1. Highest spatial holdout average precision.
2. Highest cross-validated average precision.
3. Highest cross-validated F1.

The spatial holdout splits labels by 2,500 m grid blocks. That is more honest
than a random split because field reports and field work are geographically
clustered. Random holdout can look too optimistic when nearby points from the
same route appear in both train and test sets.

After a model family is selected, the final model is trained on all labeled rows
and saved to:

```text
models/actual_ulster_field_report_model.joblib
```

## Supervised Probability

For prediction, the saved model returns:

```text
culvert_probability = model.predict_proba(features)[:, 1]
```

That value is a probability-like score from `0` to `1`, where higher means the
model thinks the candidate looks more like labeled culverts. It is not the final
field map score by itself.

The supervised prediction output also creates:

- `priority_rank`, sorted by `culvert_probability`,
- `priority_percentile`,
- probability bucket: `low`, `medium`, `high`, or `very_high`.

## Interpretable GIS Evidence Score

The pipeline also computes a non-ML evidence score in
`src/culvert_ai/scoring.py`. This keeps the ranking explainable and useful even
when supervised labels are sparse.

The component weights are:

| Component | Weight |
| --- | ---: |
| `road_stream_proximity_score` | `0.16` |
| `drainage_strength_score` | `0.16` |
| `valley_position_score` | `0.15` |
| `crossing_geometry_score` | `0.05` |
| `terrain_break_score` | `0.12` |
| `road_context_score` | `0.05` |
| `dem_route_drainage_score` | `0.18` |
| `osm_culvert_tag_score` | `0.04` |
| `field_corridor_support_score` | `0.08` |

Each component is normalized to `0..1` where possible. The evidence score is:

```text
culvert_likelihood_score =
  100 * weighted_average(component_scores)
  - 20 * non_culvert_structure_penalty
```

Then it is clipped to `0..100`.

Special rules:

- field-denied candidates are forced to `0`,
- known culvert labels are forced to `0` in the field-review queue so they do
  not consume discovery priority,
- the evidence summary names the strongest visible signals.

`field_corridor_support_score` comes from approved-known culvert pattern
columns built in `features.py`. It is intentionally bounded and small: a
candidate near an approved CBU or document-confirmed culvert on the same route
gets support, but it still needs drainage, DEM, crossing geometry, or model
agreement to become a top discovery candidate.

## Final Discovery Score

The final map ranking uses `discovery_score`, not raw model probability alone.

First the code converts values to `0..1`:

```text
evidence_score = culvert_likelihood_score / 100
model_probability = culvert_probability
model_rank_score = percentile_rank(culvert_probability)
```

The model rank percentile is used because absolute model probabilities can be
poorly calibrated when labels are sparse or clustered. Rank still tells the app
which candidates the model prefers most.

For candidates with a supervised model output:

```text
weighted_signal =
  0.40 * evidence_score
  + 0.60 * model_rank_score

agreement_signal =
  sqrt(evidence_score * model_rank_score)

discovery_score =
  100 * (
    0.55 * agreement_signal
    + 0.25 * evidence_score
    + 0.20 * weighted_signal
  )
```

Route-sampled candidates also get a bounded field-recall check. If a route
candidate has strong `field_corridor_support_score`, the ranking can lift it
with `field_recall_score`, capped at `70`, using corridor support, model rank,
GIS evidence, and DEM route-drainage evidence. This is meant to keep learned
field corridors visible on the map instead of dropping them below the web export
cutoff.

If no supervised model output is available, the discovery score falls back to the
evidence score.

Denied field observations force `discovery_score` to `0`.

Known field matches remain visible, but sorting puts unchecked candidates first.
That prevents the field queue from being dominated by culverts the team already
confirmed.

## Web Export

The web export writes:

```text
web/data/findings.geojson
web/data/summary.json
web/data/model_summary.json
```

The current map export keeps:

- declustered unchecked discovery candidates up to the configured web limit,
- summary counts for known field matches.

Known and denied rows are filtered from `findings.geojson` so the browser map
stays focused on field-review targets. Use the full discovery layer in
`data/processed/actual_ulster_discovery_predictions.gpkg` for validation and
known-match audit work.

The Leaflet app displays those rows and uses Vercel observations to refresh the
served ranking immediately after field feedback.

## Continuous Learning Behavior

The app does not update model weights inside the browser or inside the upload
request. Instead:

1. The user adds or deletes an observation on Vercel.
2. `/api/observations` saves the observation to Vercel Blob.
3. The served ranking refreshes immediately from the saved feedback.
4. The retraining trigger queues an external worker when configured.
5. The worker runs `npm run retrain:from-vercel`.
6. The rebuilt `web/data` files are deployed.

This is continuous retraining in batches. It is the right shape for this project
because the full model needs Python, geospatial libraries, source GIS data, and
more runtime than a normal Vercel request should use.

## Why Today's Missed Road Matters

If a field road had no predicted culverts but the team found several, that means
one or more of these things is true:

- candidate generation did not create enough candidate points on that road,
- the current GIS layers did not expose the drainage evidence,
- the model has too few examples like that road,
- the field culverts are in a pattern not yet represented by the training set.

Adding the confirmed CBU points helps in two ways:

- the exact culvert locations become positive labels,
- nearby missed predicted candidates can become negative or missed labels.
- nearby and same-route candidates receive bounded approved-known pattern
  support in the evidence ranking.

The next retrain can then rank similar geography higher, but only for candidate
locations that the pipeline creates. If the issue is missing candidates, the fix
is to improve candidate generation as well as model training.

## Limitations

- Field labels are still clustered by routes and work days.
- Census roads and water layers are coarse compared with engineering GIS.
- The default USGS 3DEP DEM is useful but still coarser than project-specific
  LiDAR-derived terrain products.
- Flow accumulation and drainage area rasters are optional and currently depend
  on local files.
- A high score is a field-review priority, not a verified culvert inventory.
- Better negatives are as valuable as better positives; `no_culvert` checks teach
  the model what to avoid.
