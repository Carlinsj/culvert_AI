# Culvert Prediction Model

Last updated: 2026-06-24

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

- Selected model: `hist_gradient_boosting`
- Candidate rows: `14,565`
- Positive labels: `210`
- Negative labels: `14,355`
- Training point rows: `140`
- Feature count: `70`
- Spatial holdout average precision: `0.645`
- Spatial holdout precision at 10: `1.000`

The source of truth for the latest run is `web/data/model_summary.json`.

## Training Labels

The model target column is `is_culvert`.

Positive labels come from:

- verified field-report culvert coordinates,
- confirmed ABU/user-added observations pulled from Vercel,
- confirmed field observations within the strict match radius.

Negative labels come from:

- candidate points that are not within the positive match radius of a known
  culvert,
- field observations marked `no_culvert`,
- missed-prediction labels when a confirmed field culvert proves that a specific
  predicted candidate was outside the hit radius.

The current strict match radius is `10 m`. A prediction 50 m from a confirmed
field culvert is a miss, not a correct prediction.

Confirmed ABU/user-added positives are included by default in retraining. Set
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
- optional DEM terrain features,
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

Only numeric feature columns are used. The code excludes target, coordinate,
label, rank, and already-computed score columns so the model does not train on
the answer or on UI ranking outputs.

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
| `road_stream_proximity_score` | `0.25` |
| `drainage_strength_score` | `0.20` |
| `valley_position_score` | `0.16` |
| `crossing_geometry_score` | `0.10` |
| `terrain_break_score` | `0.13` |
| `road_context_score` | `0.10` |
| `osm_culvert_tag_score` | `0.06` |
| `field_report_support_score` | `0.08` |

Each component is normalized to `0..1` where possible. The evidence score is:

```text
culvert_likelihood_score =
  100 * weighted_average(component_scores)
  - 20 * non_culvert_structure_penalty
```

Then it is clipped to `0..100`.

Special rules:

- field-denied candidates are forced to `0`,
- known culvert labels are clipped to at least `95`,
- the evidence summary names the strongest visible signals.

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

The map export keeps:

- the top `1,000` unchecked discovery candidates,
- all known field matches needed for context and validation.

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

Adding the confirmed ABU points helps in two ways:

- the exact culvert locations become positive labels,
- nearby missed predicted candidates can become negative or missed labels.

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
