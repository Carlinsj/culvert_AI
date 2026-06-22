# Research Notes

## Pitch

Field crews lose time finding culverts before inspection. This project turns field-report coordinates
and geospatial evidence into a ranked mobile map so crews can inspect likely locations first.

This is not an LLM location model. The predictor is geospatial ML. LLMs are useful only for checking
messy report text and flagging uncertain extracted rows.

## Research Plan

- Use Ulster County as the pilot area.
- Generate candidates from road/drainage crossings, route samples, and valid field-report coordinates.
- Extract features from road proximity, stream proximity, crossing geometry, road density, stream
  density, and optional terrain/hydrology rasters.
- Train multiple model families and select by spatial holdout average precision.
- Validate by field route/day using precision at 10, 25, and 50.
- Add confirmed positives and confirmed no-culvert negatives after field checks.

## Data Needed

Highest priority:

- confirmed no-culvert checks,
- official NYSDOT/county road centerlines,
- official hydrography/drainage/ditch layers,
- LiDAR-derived DEM,
- flow accumulation raster,
- drainage area raster.

Useful later:

- wetlands/floodplain layers,
- road class and maintenance jurisdiction,
- inspection app exports,
- photo/video GPS metadata,
- existing culvert inventories from nearby regions.

## Professor Email Draft

Subject: Culvert Location Prediction Pilot For Ulster County Field Work

Dear Professor [Last Name],

During the NYSDOT/C2SMART field work, I noticed that a major bottleneck is not only inspecting
culverts, but first locating them efficiently.

I have started building a reproducible Ulster County pilot workflow for culvert location prediction.
The prototype extracts verified coordinates from field reports, filters them with geospatial QC,
adds valid coordinates as training points, compares multiple machine-learning models, and exports a
ranked mobile map for field review.

The current model is geospatial, not an LLM. It learns from road/drainage geometry, density features,
and verified field coordinates. An LLM may help check messy report text, but it should not invent
coordinates or replace the GIS/model workflow.

I would like to discuss whether this could become a research project next semester. The goal would
be to measure whether a ranked geospatial model can reduce field search time and improve route
planning for culvert inspection teams.

Best regards,

[Your Name]
