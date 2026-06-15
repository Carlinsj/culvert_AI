# No-Inventory Prediction Plan

This is the correct workflow when the target area has no reliable culvert inventory and field crews
are walking long distances trying to find culverts.

## Core Strategy

Do not start with supervised training in the target area. Start with a no-local-label prediction
workflow:

1. Generate likely locations where roads cross streams, drainage lines, valleys, or low points.
2. Extract topographic evidence from DEM/LiDAR:
   - slope
   - terrain roughness
   - local relief
   - topographic position
   - valley depth
3. Score each candidate with an expert/weak-supervision model.
4. Export a ranked map, CSV, and KML for Google Earth review.
5. Field crews visit high-priority candidates first.
6. Every field result becomes a label for future training.

## Why This Works Without Known Culvert Points

Culverts are usually installed where water needs to pass under a road. Even when the structure is
not mapped, the landscape often leaves clues:

- road crosses a mapped stream or drainage line
- road crosses a small valley or swale
- local DEM cell is lower than surrounding terrain
- terrain relief changes near the road
- stream/drainage density increases near the road
- crossing angle looks like a road cutting across drainage flow

The first model is therefore a ranking model, not a final supervised classifier.

## Google Earth Review

The pipeline writes:

- `data/processed/ulster_unlabeled_predictions.csv`
- `data/processed/ulster_google_earth_review.kml`

Open the KML in Google Earth and review the top-ranked points. Look for:

- visible pipe openings or headwalls
- road embankments crossing a drainage channel
- vegetation lines indicating water flow
- ditches leading toward a road
- wet areas or sediment patterns near both sides of the road
- guardrails or shoulder changes near drainage crossings

Suggested review fields:

- `visible_culvert`: yes / no / unsure
- `visible_channel`: yes / no / unsure
- `road_embankment`: yes / no / unsure
- `field_priority`: high / medium / low
- `notes`

## Transfer Learning From Other Areas

If another county or region has verified culvert locations, use that as external training data:

1. Build the same features in the source region.
2. Train the supervised model on source-region labels.
3. Build the same features in Ulster County.
4. Predict on Ulster County candidates.
5. Compare the transfer model ranking with the no-label evidence score.

This is cross-region transfer learning. It is weaker than local labels, but it is much better than
starting from nothing.

The repo includes a starter script:

```bash
scripts/run_transfer_from_external_region.sh
```

Expected external files:

- `data/external/source_roads.gpkg`
- `data/external/source_streams.gpkg`
- `data/external/source_known_culverts.gpkg`
- `data/external/source_dem.tif`

## Field Feedback Loop

After each field day, add confirmed locations back into the dataset:

- confirmed culvert found
- no culvert found
- inaccessible/uncertain
- photo/video metadata
- notes about why the location was hard to see

Once there are enough field labels, the project can switch from weak scoring to supervised training.
