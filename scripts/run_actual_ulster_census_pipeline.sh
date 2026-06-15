#!/usr/bin/env bash
set -euo pipefail

# Real-data first-pass prediction workflow.
# Downloads actual Census TIGER/Line roads/linear water for Ulster County,
# predicts likely culvert locations, and refreshes the web dashboard data.
# If data/raw/dem.tif exists, it is used automatically.

scripts/python.sh -m culvert_ai.cli download-census \
  --output-dir data/raw \
  --statefp "36" \
  --countyfp "111"

FIELD_REPORTS_PATH="${FIELD_REPORTS_PATH:-/Users/Carli/Downloads/Team No. 3.zip}"
KNOWN_CULVERTS_PATH=""
if [ -e "$FIELD_REPORTS_PATH" ]; then
  scripts/python.sh -m culvert_ai.cli import-field-reports \
    --input "$FIELD_REPORTS_PATH" \
    --output data/processed/field_report_culverts.gpkg \
    --csv-output data/processed/field_report_culverts.csv
  KNOWN_CULVERTS_PATH="data/processed/field_report_culverts.gpkg"
fi

if [ -f data/processed/field_observations.geojson ]; then
  CONFIRMED_OBSERVATIONS="$(scripts/python.sh - <<'PY'
import json
from pathlib import Path

path = Path("data/processed/field_observations.geojson")
data = json.loads(path.read_text())
features = data.get("features", [])
print(sum(1 for feature in features if feature.get("properties", {}).get("status") == "confirmed_culvert"))
PY
)"
  if [ "$CONFIRMED_OBSERVATIONS" -gt 0 ] || { [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; }; then
    MERGE_OBSERVATIONS_ARGS=(
      --observations data/processed/field_observations.geojson
      --output data/processed/training_known_culverts.gpkg
      --csv-output data/processed/training_known_culverts.csv
    )
    if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
      MERGE_OBSERVATIONS_ARGS+=(--base-known "$KNOWN_CULVERTS_PATH")
    fi
    scripts/python.sh -m culvert_ai.cli merge-field-observations "${MERGE_OBSERVATIONS_ARGS[@]}"
    KNOWN_CULVERTS_PATH="data/processed/training_known_culverts.gpkg"
  fi
fi

scripts/python.sh -m culvert_ai.cli build-candidates \
  --roads data/raw/roads.gpkg \
  --streams data/raw/streams.gpkg \
  --output data/interim/actual_ulster_candidates.gpkg \
  --snap-tolerance-m 35 \
  --min-spacing-m 20

CANDIDATES_PATH="data/interim/actual_ulster_candidates.gpkg"
if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
  scripts/python.sh -m culvert_ai.cli build-road-candidates \
    --roads data/raw/roads.gpkg \
    --routes-from "$KNOWN_CULVERTS_PATH" \
    --interval-m 75 \
    --output data/interim/actual_ulster_route_candidates.gpkg

  scripts/python.sh -m culvert_ai.cli merge-candidates \
    --inputs data/interim/actual_ulster_candidates.gpkg data/interim/actual_ulster_route_candidates.gpkg \
    --output data/interim/actual_ulster_candidates_with_route_samples.gpkg
  CANDIDATES_PATH="data/interim/actual_ulster_candidates_with_route_samples.gpkg"
fi

if [ -f data/raw/dem.tif ]; then
  if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
    scripts/python.sh -m culvert_ai.cli build-features \
      --candidates "$CANDIDATES_PATH" \
      --known-culverts "$KNOWN_CULVERTS_PATH" \
      --positive-radius-m 75 \
      --roads data/raw/roads.gpkg \
      --streams data/raw/streams.gpkg \
      --dem data/raw/dem.tif \
      --density-radii-m 50 100 250 500 \
      --output data/processed/actual_ulster_unlabeled_features.gpkg
  else
    scripts/python.sh -m culvert_ai.cli build-features \
      --candidates "$CANDIDATES_PATH" \
      --roads data/raw/roads.gpkg \
      --streams data/raw/streams.gpkg \
      --dem data/raw/dem.tif \
      --density-radii-m 50 100 250 500 \
      --output data/processed/actual_ulster_unlabeled_features.gpkg
  fi
else
  if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
    scripts/python.sh -m culvert_ai.cli build-features \
      --candidates "$CANDIDATES_PATH" \
      --known-culverts "$KNOWN_CULVERTS_PATH" \
      --positive-radius-m 75 \
      --roads data/raw/roads.gpkg \
      --streams data/raw/streams.gpkg \
      --density-radii-m 50 100 250 500 \
      --output data/processed/actual_ulster_unlabeled_features.gpkg
  else
    scripts/python.sh -m culvert_ai.cli build-features \
      --candidates "$CANDIDATES_PATH" \
      --roads data/raw/roads.gpkg \
      --streams data/raw/streams.gpkg \
      --density-radii-m 50 100 250 500 \
      --output data/processed/actual_ulster_unlabeled_features.gpkg
  fi
fi

scripts/python.sh -m culvert_ai.cli score-unlabeled \
  --features data/processed/actual_ulster_unlabeled_features.gpkg \
  --output data/processed/actual_ulster_unlabeled_predictions.gpkg \
  --csv-output data/processed/actual_ulster_unlabeled_predictions.csv \
  --kml-output data/processed/actual_ulster_evidence_review.kml \
  --kml-max-points 500

SUPERVISED_PREDICTIONS_PATH=""
if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
  read -r POSITIVES NEGATIVES < <(scripts/python.sh - <<'PY'
import geopandas as gpd
features = gpd.read_file("data/processed/actual_ulster_unlabeled_features.gpkg")
if "is_culvert" not in features:
    print("0 0")
else:
    y = features["is_culvert"].astype(int)
    print(int(y.sum()), int((1 - y).sum()))
PY
)
  if [ "$POSITIVES" -ge 2 ] && [ "$NEGATIVES" -ge 2 ]; then
    scripts/python.sh -m culvert_ai.cli train \
      --features data/processed/actual_ulster_unlabeled_features.gpkg \
      --model-output models/actual_ulster_field_report_model.joblib \
      --metrics-output reports/actual_ulster_field_report_metrics.json \
      --importance-output reports/actual_ulster_field_report_feature_importance.csv \
      --model-family auto \
      --spatial-block-size-m 2500

    scripts/python.sh -m culvert_ai.cli predict \
      --features data/processed/actual_ulster_unlabeled_features.gpkg \
      --model models/actual_ulster_field_report_model.joblib \
      --output data/processed/actual_ulster_supervised_predictions.gpkg \
      --csv-output data/processed/actual_ulster_supervised_predictions.csv
    SUPERVISED_PREDICTIONS_PATH="data/processed/actual_ulster_supervised_predictions.gpkg"
  fi
fi

DISCOVERY_ARGS=(
  --evidence-predictions data/processed/actual_ulster_unlabeled_predictions.gpkg
  --output data/processed/actual_ulster_discovery_predictions.gpkg
  --csv-output data/processed/actual_ulster_discovery_predictions.csv
  --kml-output data/processed/actual_ulster_google_earth_review.kml
  --kml-max-points 500
)

if [ -n "$SUPERVISED_PREDICTIONS_PATH" ]; then
  DISCOVERY_ARGS+=(--supervised-predictions "$SUPERVISED_PREDICTIONS_PATH")
fi

scripts/python.sh -m culvert_ai.cli build-discovery-ranking "${DISCOVERY_ARGS[@]}"

scripts/python.sh -m culvert_ai.cli export-web \
  --predictions data/processed/actual_ulster_discovery_predictions.gpkg \
  --output-dir web/data \
  --limit 1000
