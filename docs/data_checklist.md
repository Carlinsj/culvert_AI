# Data Checklist

Use this checklist when talking to the professor, NYSDOT, or the hydraulics team about the Ulster
County side of the Poughkeepsie/Hudson Valley field area.

## Required

- Official Ulster County boundary or NYSDOT-defined project boundary.
- Road centerline layer for Ulster County.
- Stream, drainage, or hydrography line layer for Ulster County.
- DEM/topography raster for Ulster County, preferably LiDAR-derived where available.
- Field-observed culvert locations from the internship, especially near Highland/Lloyd, Esopus,
  New Paltz, Marlboro, Plattekill, Rosendale, and south/east Kingston approach corridors.

## Strongly Recommended

- LiDAR-derived elevation or terrain products.
- Land cover raster.
- Road names, route IDs, and functional class.
- Culvert condition app export, if available.

## Useful Later

- Flow accumulation raster.
- Watershed or catchment boundaries.
- Ditch or drainage channel layers.
- Orthophotos or high-resolution aerial imagery.
- Maintenance history or flooding complaint locations.
- Existing culvert inventory points from other counties or regions for transfer learning.
- Any verified culvert inventory for Ulster County, if discovered later.

## Minimum Data Fields For Known Culverts

- Unique culvert ID.
- Latitude/longitude or projected point geometry.
- Source agency or inventory source.
- Observation date if available.
- Condition fields if available.

## Privacy And Data Handling

Do not commit restricted NYSDOT files to GitHub. Keep raw agency data in `data/raw/`, which is
ignored by git in this repo.
