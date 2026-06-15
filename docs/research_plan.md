# Research Plan: Ulster County Culvert Location Prediction

## Working Title

Geospatial Prediction of Culvert Locations in Ulster County Using Road, Drainage, Topographic, and
Remote Sensing Data

## Problem

Field teams spend a large amount of time locating culverts before they can inspect them. The current
inspection workflow can record culvert information once the structure is found, but the search step
is still slow in rural, vegetated, or poorly mapped areas.

## Research Goal

Develop and validate a GIS and machine-learning workflow that produces a ranked map of likely
culvert locations before field visits in Ulster County. The project is tied to the Poughkeepsie-area
field assignment, but the model scope should be Ulster County west of the Hudson River.

Because the main field problem is that culvert locations are not known in advance, the first phase
should not depend on local culvert labels. The first phase should use weak supervision and
topographic evidence to rank likely locations, then use field discoveries as labels for later
supervised training.

## Initial Research Questions

1. Can road-stream crossing candidates explain most verified culvert locations in the Ulster County
   pilot area?
2. Which geospatial features best predict culvert presence: stream density, road density,
   topography, slope, land cover, or proximity to known drainage?
3. How much field search time could be reduced by visiting high-probability points first?

## Proposed Method

1. Select the Ulster County side of the Poughkeepsie/Hudson Valley field area as the pilot region.
   Candidate focus areas include Highland/Lloyd, Esopus, New Paltz, Marlboro, Plattekill,
   Rosendale, and south/east Kingston approach corridors.
2. Gather road centerlines, hydrography/drainage lines, DEM/topography, land cover, and field
   observations.
3. Generate candidate points at road-stream intersections, road-drainage crossings, and road-low
   point terrain features.
4. Extract geospatial features for each candidate.
5. Score candidates using a no-local-label evidence model based on topography, drainage, and road
   crossing context.
6. Export a ranked map layer, CSV, and Google Earth KML for field review.
7. Validate high-ranked points against field discoveries.
8. Use confirmed field discoveries and external culvert inventories from other regions for later
   transfer learning or supervised training.

## Baseline Model

The first baseline should be rule-based GIS:

- Generate all road-stream crossings.
- Rank exact crossings higher than near crossings.
- Add priority where slope, stream density, and terrain suggest drainage concentration.

The machine-learning model should then be compared against this baseline.

## Model Features

The starter code currently supports:

- Road-stream crossing distance.
- Road-stream crossing angle.
- Exact versus near road-stream crossing indicator.
- Road density near the candidate.
- Stream/drainage density near the candidate.
- Multi-scale density windows, including 50 m, 100 m, 250 m, and 500 m.
- DEM elevation.
- Local DEM slope.
- DEM relief and terrain roughness around the candidate.
- Land cover raster value.
- Optional stream order if the hydrography layer includes it.
- Optional road speed limit if the road layer includes it.

## Model Training Design

The project has two modes.

No-local-label mode:

- Use road-drainage candidates.
- Score candidates with a weak-supervision evidence model.
- Export ranked locations and Google Earth KML for review.
- Treat field discoveries as new labels.

Supervised/transfer mode:

- Use verified culvert locations from other regions or future field observations.
- Build the same feature table for source and target areas.
- Train on source/confirmed labels.
- Predict on Ulster County candidates.

When labels exist, the training pipeline compares several model families:

- Baseline prior model.
- Regularized logistic regression.
- Random forest.
- Extra trees.
- Histogram gradient boosting.

The selected model is chosen by cross-validated average precision, with the baseline included as a
sanity check. The pipeline also saves feature importance so the hydraulics team can inspect which
signals are driving the ranked output.

Additional future features:

- Flow accumulation.
- Topographic wetness index.
- Distance to nearest watershed boundary.
- Road class or functional class.
- Drainage area upstream of crossing.
- LiDAR-derived ditch/channel signatures.
- Google Earth/manual imagery review labels.

## Validation Metrics

- Precision at high-priority threshold.
- Recall of known culverts within a search radius.
- Average precision.
- ROC AUC.
- F1 score.
- Brier score for probability calibration.
- Top-k precision and recall for field crew use.
- Spatial holdout validation to test whether the model generalizes across different parts of the
  pilot area.
- Number of false positives per mile of roadway.
- Field time saved compared with unranked search.

## Semester Deliverables

1. Clean pilot dataset for the Ulster County pilot region.
2. Reproducible Python/GIS workflow.
3. Ranked prediction map layer.
4. Evaluation report comparing predicted and verified culvert locations.
5. Short presentation for the hydraulics team.

## Immediate Ask For Professor

Ask whether you can work with the hydraulics team next semester to turn the summer field observation
into an Ulster County pilot research project. The key request is access to data, technical guidance,
and an official Ulster County or NYSDOT project boundary.
