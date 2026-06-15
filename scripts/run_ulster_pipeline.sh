#!/usr/bin/env bash
set -euo pipefail

# This script scopes the analysis to the Ulster County side of the
# Poughkeepsie/Hudson Valley field area before generating culvert candidates.

scripts/python.sh -m culvert_ai.cli filter-region \
  --input data/raw/roads.gpkg \
  --output data/interim/ulster_roads.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

scripts/python.sh -m culvert_ai.cli filter-region \
  --input data/raw/streams.gpkg \
  --output data/interim/ulster_streams.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

scripts/python.sh -m culvert_ai.cli filter-region \
  --input data/raw/known_culverts.gpkg \
  --output data/interim/ulster_known_culverts.gpkg \
  --boundary configs/regions/ulster_poughkeepsie_pilot.geojson

scripts/python.sh -m culvert_ai.cli build-candidates \
  --roads data/interim/ulster_roads.gpkg \
  --streams data/interim/ulster_streams.gpkg \
  --output data/interim/ulster_candidates.gpkg

scripts/python.sh -m culvert_ai.cli build-features \
  --candidates data/interim/ulster_candidates.gpkg \
  --known-culverts data/interim/ulster_known_culverts.gpkg \
  --roads data/interim/ulster_roads.gpkg \
  --streams data/interim/ulster_streams.gpkg \
  --dem data/raw/dem.tif \
  --landcover data/raw/landcover.tif \
  --density-radii-m 50 100 250 500 \
  --output data/processed/ulster_training_features.gpkg

scripts/python.sh -m culvert_ai.cli train \
  --features data/processed/ulster_training_features.gpkg \
  --model-output models/ulster_culvert_model.joblib \
  --metrics-output reports/ulster_metrics.json \
  --importance-output reports/ulster_feature_importance.csv \
  --model-family auto \
  --spatial-block-size-m 2500

scripts/python.sh -m culvert_ai.cli predict \
  --features data/processed/ulster_training_features.gpkg \
  --model models/ulster_culvert_model.joblib \
  --output data/processed/ulster_predictions.gpkg \
  --csv-output data/processed/ulster_predictions.csv
