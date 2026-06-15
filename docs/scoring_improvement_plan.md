# Improving Culvert Prediction And Scoring

The current `npm run predict:actual` output is a real-data evidence ranking. It is not yet a
field-validated classifier because we do not have confirmed local culvert and non-culvert labels.

## Fastest Improvements

1. Add a DEM at `data/raw/dem.tif`.
   - Use LiDAR-derived DEM if NYSDOT, county GIS, or USGS 3DEP data is available.
   - This activates slope, terrain roughness, local relief, topographic position, and valley-depth
     features.

2. Replace Census roads/water lines with project-grade layers.
   - Prefer NYSDOT road centerlines and official hydrography/drainage layers.
   - Census TIGER is reliable for a first pass, but it is not as detailed as engineering GIS.

3. Collect labels during field work.
   - Confirmed culvert: point found in field.
   - False positive: high-ranked candidate inspected but no culvert found.
   - Unknown: not yet inspected.
   - False positives are as valuable as positives because they teach the model what not to rank.
   - Use `npm run import:reports` to extract field report coordinates from the Team 3 ZIP.

4. Retrain with verified labels.
   - Once labels exist, use `npm run pipeline:ulster` or the supervised `culvert-ai train` command.
   - Compare average precision, top-k hit rate, and spatial holdout performance.

## Scoring Features To Add Next

- Drainage accumulation from DEM flow direction/flow accumulation.
- Distance to mapped wetlands, floodplains, and stormwater infrastructure.
- Road embankment/valley crossing indicators from elevation profiles perpendicular to the road.
- Road class and maintenance jurisdiction from NYSDOT data.
- Historical inspection app exports and photo GPS metadata.
- Google Earth/manual review status: likely culvert, unlikely culvert, unclear.

## Field Validation Plan

- Start with the top 25 candidates on one route, such as State Route 28.
- Mark each as confirmed culvert, no culvert, inaccessible, duplicate, or uncertain.
- Re-run scoring after adding labels.
- Report precision at 10, 25, 50, and 100 candidates. This directly answers whether the model saves
  walking time.

## What To Tell The Professor

The first research milestone is not "perfect AI." It is a reproducible ranking system that uses real
roads, drainage, and topography to reduce search time. The scientific contribution is improving the
ranking with field feedback until the highest-priority route segments produce reliable culvert hits.
