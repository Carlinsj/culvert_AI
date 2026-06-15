# Competitive Strategy For Getting Assigned To The Project

## Core Message

Do not present this as "I want to use AI." Present it as:

> I observed a real field bottleneck: inspectors lose time locating culverts before they can inspect
> them. I built a reproducible Ulster County pilot workflow that uses road-stream crossings,
> topography, drainage density, and known culvert inventories to rank likely culvert locations before
> field visits.

That framing is stronger because it starts with the infrastructure problem, not the technology.

## What Makes This More Serious Than A Basic AI Idea

- The model is scoped to a specific pilot area: Ulster County west of the Hudson River.
- It starts with a hydrologic baseline: roads crossing streams or drainage lines.
- It uses features a hydraulics team can understand: stream order, road-stream distance, crossing
  angle, slope, terrain roughness, road density, stream density, and elevation.
- It compares several models instead of assuming one algorithm is best.
- It includes spatial holdout validation, which reduces the risk of overfitting to nearby points.
- It produces field-ready ranked outputs, not just a model score.
- It saves feature importance so the team can inspect why the model ranks locations highly.
- It has a no-inventory mode for the real field problem: prediction before any local culvert labels
  exist.

## What To Say In The Meeting

1. "I am not trying to replace field inspection. I am trying to reduce search time before inspection."
2. "Because we do not know where the culverts are in the target area, the first model should be a
   weak-supervision ranking model using topography, drainage, and road-crossing evidence."
3. "I scoped the first version to Ulster County because a smaller pilot is easier to validate."
4. "I built the pipeline so it can rank candidates without local labels, then improve as field
   observations become confirmed labels."
5. "The project can produce a useful research result even if the final model is simple, because the
   evaluation will show which geospatial signals actually help inspectors."

## What To Ask For

- Official Ulster County or NYSDOT project boundary.
- Existing culvert inventory points for Ulster County.
- Road centerline layer with route IDs and road class.
- Hydrography or drainage line layer.
- DEM or LiDAR-derived terrain raster.
- Permission to use summer field observations for validation.
- Guidance from the hydraulics team on what field conditions usually indicate hidden culverts.

## 30/60/90 Day Plan

First 30 days:

- Obtain and clean Ulster County roads, hydrography, DEM, and known culvert data.
- Generate candidate road-stream crossings.
- Create a baseline GIS ranking.

Days 31-60:

- Build the training table.
- Compare regularized logistic regression, random forest, extra trees, and gradient boosting.
- Run random and spatial holdout validation.
- Produce feature-importance results for technical review.

Days 61-90:

- Export a ranked field map for validation.
- Compare predicted high-priority points with field observations.
- Write a short report showing precision, recall, top-k field utility, and time-saving potential.

## How To Avoid Overpromising

Say this clearly:

> The current repo is a research-grade prototype. The model will only be genuinely trained after we
> connect it to verified Ulster County culvert inventory and field observations. The value I bring
> now is that the pipeline, validation design, and field-use case are already structured.
