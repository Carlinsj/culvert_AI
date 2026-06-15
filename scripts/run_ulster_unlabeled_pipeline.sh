#!/usr/bin/env bash
set -euo pipefail

# Use this when there is no reliable culvert inventory in the target area.
# It ranks likely culvert locations from road-drainage crossings and topographic evidence.

scripts/python.sh -m culvert_ai.cli filter-region \
  --input data/raw/roads.gpkg \
  --output data/interim/ulster_roads.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

scripts/python.sh -m culvert_ai.cli filter-region \
  --input data/raw/streams.gpkg \
  --output data/interim/ulster_streams.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

scripts/python.sh -m culvert_ai.cli build-candidates \
  --roads data/interim/ulster_roads.gpkg \
  --streams data/interim/ulster_streams.gpkg \
  --output data/interim/ulster_candidates.gpkg \
  --snap-tolerance-m 35 \
  --min-spacing-m 20

scripts/python.sh -m culvert_ai.cli build-features \
  --candidates data/interim/ulster_candidates.gpkg \
  --roads data/interim/ulster_roads.gpkg \
  --streams data/interim/ulster_streams.gpkg \
  --dem data/raw/dem.tif \
  --density-radii-m 50 100 250 500 \
  --output data/processed/ulster_unlabeled_features.gpkg

scripts/python.sh -m culvert_ai.cli score-unlabeled \
  --features data/processed/ulster_unlabeled_features.gpkg \
  --output data/processed/ulster_unlabeled_predictions.gpkg \
  --csv-output data/processed/ulster_unlabeled_predictions.csv \
  --kml-output data/processed/ulster_google_earth_review.kml \
  --kml-max-points 250

scripts/python.sh -m culvert_ai.cli export-web \
  --predictions data/processed/ulster_unlabeled_predictions.gpkg \
  --output-dir web/data
