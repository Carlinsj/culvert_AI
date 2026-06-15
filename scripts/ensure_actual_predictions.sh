#!/usr/bin/env bash
set -euo pipefail

if [ -f data/processed/actual_ulster_unlabeled_predictions.gpkg ] && \
   [ -f web/data/findings.geojson ] && \
   [ -f web/data/summary.json ]; then
  echo "Actual Ulster predictions already exist. Skipping download."
  exit 0
fi

echo "Actual Ulster predictions are missing. Running real Census TIGER/Line prediction workflow."
bash scripts/run_actual_ulster_census_pipeline.sh
