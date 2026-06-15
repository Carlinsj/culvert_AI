#!/usr/bin/env bash
set -euo pipefail

# Use this when another county/region has verified culvert locations.
# It trains on the external region and applies the learned pattern to Ulster County candidates.

scripts/python.sh -m culvert_ai.cli build-candidates \
  --roads data/external/source_roads.gpkg \
  --streams data/external/source_streams.gpkg \
  --output data/interim/source_candidates.gpkg \
  --snap-tolerance-m 35 \
  --min-spacing-m 20

scripts/python.sh -m culvert_ai.cli build-features \
  --candidates data/interim/source_candidates.gpkg \
  --known-culverts data/external/source_known_culverts.gpkg \
  --roads data/external/source_roads.gpkg \
  --streams data/external/source_streams.gpkg \
  --dem data/external/source_dem.tif \
  --density-radii-m 50 100 250 500 \
  --output data/processed/source_training_features.gpkg

scripts/python.sh -m culvert_ai.cli train \
  --features data/processed/source_training_features.gpkg \
  --model-output models/transfer_culvert_model.joblib \
  --metrics-output reports/transfer_source_metrics.json \
  --importance-output reports/transfer_source_feature_importance.csv \
  --model-family auto \
  --spatial-block-size-m 2500

scripts/python.sh -m culvert_ai.cli predict \
  --features data/processed/ulster_unlabeled_features.gpkg \
  --model models/transfer_culvert_model.joblib \
  --output data/processed/ulster_transfer_predictions.gpkg \
  --csv-output data/processed/ulster_transfer_predictions.csv
