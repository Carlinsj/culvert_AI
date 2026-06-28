#!/usr/bin/env bash
set -euo pipefail

# Real-data first-pass prediction workflow.
# Downloads actual Census TIGER/Line roads/linear water for Ulster County,
# predicts likely culvert locations, and refreshes the web dashboard data.
# Downloads a USGS 3DEP DEM by default if data/raw/dem.tif is missing.

if [ "${REFRESH_CENSUS_INPUTS:-0}" = "1" ] || [ ! -f data/raw/roads.gpkg ] || [ ! -f data/raw/streams.gpkg ]; then
  scripts/python.sh -m culvert_ai.cli download-census \
    --output-dir data/raw \
    --statefp "36" \
    --countyfp "111"
else
  echo "Using existing Census inputs in data/raw. Set REFRESH_CENSUS_INPUTS=1 to download fresh copies."
fi

DEM_PATH="${DEM_PATH:-data/raw/dem.tif}"
if [ "${DOWNLOAD_DEM:-1}" = "1" ] && { [ "${REFRESH_DEM:-0}" = "1" ] || [ ! -f "$DEM_PATH" ]; }; then
  DOWNLOAD_DEM_ARGS=(
    --boundary data/raw/ulster_county_boundary.gpkg
    --output "$DEM_PATH"
    --source-dir "${DEM_SOURCE_DIR:-data/raw/sources/dem}"
    --resolution "${DEM_RESOLUTION:-1}"
  )
  if [ "${REFRESH_DEM:-0}" = "1" ]; then
    DOWNLOAD_DEM_ARGS+=(--overwrite)
  fi
  scripts/python.sh -m culvert_ai.cli download-dem "${DOWNLOAD_DEM_ARGS[@]}"
elif [ -f "$DEM_PATH" ]; then
  echo "Using existing DEM at $DEM_PATH. Set REFRESH_DEM=1 to rebuild it."
else
  echo "DEM download disabled. Set DOWNLOAD_DEM=1 to add terrain features."
fi

DEFAULT_FIELD_REPORTS_PATH="/Users/Carli/Downloads/Team No. 2-selected (1)"
FIELD_REPORTS_MANIFEST="${FIELD_REPORTS_MANIFEST:-configs/field_report_inputs.txt}"
LLM_REVIEWED_CULVERTS_PATH="${LLM_REVIEWED_CULVERTS_PATH:-data/processed/field_report_llm_reviewed_culverts.gpkg}"
BOUNDARY_PATH="${BOUNDARY_PATH:-data/raw/ulster_county_boundary.gpkg}"
EXTRACTED_POINTS_PATH=""
KNOWN_CULVERTS_PATH=""
DENIED_CULVERTS_PATH=""
TRAINING_POINTS_SUMMARY_PATH="data/processed/high_confidence_training_points.csv"
INCLUDE_FIELD_OBSERVATIONS_AS_POSITIVES="${INCLUDE_FIELD_OBSERVATIONS_AS_POSITIVES:-1}"

FIELD_REPORT_INPUTS=()
if [ -n "${FIELD_REPORTS_PATHS:-}" ]; then
  IFS=":" read -r -a FIELD_REPORT_INPUTS <<< "$FIELD_REPORTS_PATHS"
elif [ -n "${FIELD_REPORT_PATH:-}" ]; then
  FIELD_REPORT_INPUTS=("$FIELD_REPORT_PATH")
elif [ -f "$FIELD_REPORTS_MANIFEST" ]; then
  while IFS= read -r FIELD_REPORT_INPUT || [ -n "$FIELD_REPORT_INPUT" ]; do
    FIELD_REPORT_INPUT="${FIELD_REPORT_INPUT#"${FIELD_REPORT_INPUT%%[![:space:]]*}"}"
    FIELD_REPORT_INPUT="${FIELD_REPORT_INPUT%"${FIELD_REPORT_INPUT##*[![:space:]]}"}"
    if [ -n "$FIELD_REPORT_INPUT" ] && [[ "$FIELD_REPORT_INPUT" != \#* ]]; then
      FIELD_REPORT_INPUTS+=("$FIELD_REPORT_INPUT")
    fi
  done < "$FIELD_REPORTS_MANIFEST"
else
  FIELD_REPORT_INPUTS=("$DEFAULT_FIELD_REPORTS_PATH")
fi
READABLE_FIELD_REPORT_INPUTS=()
UNREADABLE_FIELD_REPORT_INPUTS=()
for FIELD_REPORT_INPUT in "${FIELD_REPORT_INPUTS[@]}"; do
  if [ -e "$FIELD_REPORT_INPUT" ] && [ -r "$FIELD_REPORT_INPUT" ]; then
    READABLE_FIELD_REPORT_INPUTS+=("$FIELD_REPORT_INPUT")
  elif [ -e "$FIELD_REPORT_INPUT" ]; then
    UNREADABLE_FIELD_REPORT_INPUTS+=("$FIELD_REPORT_INPUT")
  fi
done

if [ "${#READABLE_FIELD_REPORT_INPUTS[@]}" -gt 0 ]; then
  if scripts/python.sh -m culvert_ai.cli import-field-reports \
    --input "${READABLE_FIELD_REPORT_INPUTS[@]}" \
    --output data/processed/field_report_culverts.gpkg \
    --csv-output data/processed/field_report_culverts.csv; then
    EXTRACTED_POINTS_PATH="data/processed/field_report_culverts.gpkg"
  elif [ -f data/processed/field_report_culverts.gpkg ]; then
    echo "Field report import failed; using existing extracted points in data/processed/field_report_culverts.gpkg."
    EXTRACTED_POINTS_PATH="data/processed/field_report_culverts.gpkg"
  else
    exit 1
  fi
elif [ -f data/processed/field_report_culverts.gpkg ]; then
  echo "Using existing extracted points in data/processed/field_report_culverts.gpkg."
  EXTRACTED_POINTS_PATH="data/processed/field_report_culverts.gpkg"
elif [ "${#UNREADABLE_FIELD_REPORT_INPUTS[@]}" -gt 0 ]; then
  echo "Field report path is not readable from this environment: ${UNREADABLE_FIELD_REPORT_INPUTS[*]}"
  exit 1
fi
if [ -f "$LLM_REVIEWED_CULVERTS_PATH" ]; then
  echo "Using LLM-reviewed field labels from $LLM_REVIEWED_CULVERTS_PATH."
  EXTRACTED_POINTS_PATH="$LLM_REVIEWED_CULVERTS_PATH"
fi

scripts/python.sh -m culvert_ai.cli build-candidates \
  --roads data/raw/roads.gpkg \
  --streams data/raw/streams.gpkg \
  --output data/interim/actual_ulster_candidates.gpkg \
  --snap-tolerance-m 20 \
  --min-spacing-m 20

CANDIDATES_PATH="data/interim/actual_ulster_candidates.gpkg"
ROUTE_COUNT=0
if [ -n "$EXTRACTED_POINTS_PATH" ] && [ -f "$EXTRACTED_POINTS_PATH" ]; then
  ROUTE_COUNT="$(POINTS_PATH="$EXTRACTED_POINTS_PATH" scripts/python.sh - <<'PY'
import os
import geopandas as gpd

path = os.environ["POINTS_PATH"]
points = gpd.read_file(path)
if "route" not in points:
    print(0)
else:
    print(int(points["route"].fillna("").astype(str).str.strip().ne("").sum()))
PY
)"
fi
BUILD_NUMBERED_ROAD_CANDIDATES="${BUILD_NUMBERED_ROAD_CANDIDATES:-1}"
if [ "$BUILD_NUMBERED_ROAD_CANDIDATES" = "1" ] || { [ -n "$EXTRACTED_POINTS_PATH" ] && [ -f "$EXTRACTED_POINTS_PATH" ] && [ "${ROUTE_COUNT:-0}" -gt 0 ]; }; then
  ROUTE_CANDIDATE_ARGS=(
    --roads data/raw/roads.gpkg
    --interval-m "${ROUTE_SAMPLE_INTERVAL_M:-10}"
    --lateral-offsets-m ${ROUTE_SAMPLE_OFFSETS_M:-0}
    --output data/interim/actual_ulster_route_candidates.gpkg
  )
  if [ "$BUILD_NUMBERED_ROAD_CANDIDATES" = "1" ]; then
    ROUTE_CANDIDATE_ARGS+=(--all-numbered-roads)
  fi
  if [ -n "$EXTRACTED_POINTS_PATH" ] && [ -f "$EXTRACTED_POINTS_PATH" ] && [ "${ROUTE_COUNT:-0}" -gt 0 ]; then
    ROUTE_CANDIDATE_ARGS+=(--routes-from "$EXTRACTED_POINTS_PATH")
  fi
  scripts/python.sh -m culvert_ai.cli build-road-candidates "${ROUTE_CANDIDATE_ARGS[@]}"

  scripts/python.sh -m culvert_ai.cli merge-candidates \
    --inputs data/interim/actual_ulster_candidates.gpkg data/interim/actual_ulster_route_candidates.gpkg \
    --output data/interim/actual_ulster_candidates_with_route_samples.gpkg
  CANDIDATES_PATH="data/interim/actual_ulster_candidates_with_route_samples.gpkg"
fi
if [ -n "$EXTRACTED_POINTS_PATH" ] && [ -f "$EXTRACTED_POINTS_PATH" ]; then
  ADD_FIELD_CANDIDATE_ARGS=(
    --candidates "$CANDIDATES_PATH"
    --field-reports "$EXTRACTED_POINTS_PATH"
    --output data/interim/actual_ulster_candidates_with_field_reports.gpkg
  )
  if [ -f "$BOUNDARY_PATH" ]; then
    ADD_FIELD_CANDIDATE_ARGS+=(--boundary "$BOUNDARY_PATH")
  fi
  scripts/python.sh -m culvert_ai.cli add-field-report-candidates "${ADD_FIELD_CANDIDATE_ARGS[@]}"
  CANDIDATES_PATH="data/interim/actual_ulster_candidates_with_field_reports.gpkg"

  ANALYZE_POINTS_ARGS=(
    --points "$EXTRACTED_POINTS_PATH"
    --roads data/raw/roads.gpkg
    --streams data/raw/streams.gpkg
    --candidates "$CANDIDATES_PATH"
    --output-geojson data/processed/extracted_points_analysis.geojson
    --output-csv data/processed/extracted_points_analysis.csv
    --output-json reports/extracted_points_analysis.json
    --output-markdown /private/tmp/culvert_extracted_points_analysis.md
    --match-radius-m 10
  )
  if [ -f "$BOUNDARY_PATH" ]; then
    ANALYZE_POINTS_ARGS+=(--boundary "$BOUNDARY_PATH")
  fi
  scripts/python.sh -m culvert_ai.cli analyze-extracted-points "${ANALYZE_POINTS_ARGS[@]}"

  scripts/python.sh -m culvert_ai.cli build-high-confidence-training-points \
    --analysis data/processed/extracted_points_analysis.geojson \
    --output data/processed/high_confidence_training_points.gpkg \
    --csv-output data/processed/high_confidence_training_points.csv
  KNOWN_CULVERTS_PATH="data/processed/high_confidence_training_points.gpkg"
fi

if [ -f data/processed/field_observations.geojson ]; then
  read -r CONFIRMED_OBSERVATIONS DENIED_OBSERVATIONS TOTAL_OBSERVATIONS < <(scripts/python.sh - <<'PY'
import json
from pathlib import Path

path = Path("data/processed/field_observations.geojson")
data = json.loads(path.read_text())
features = data.get("features", [])
confirmed = sum(1 for feature in features if feature.get("properties", {}).get("status") == "confirmed_culvert")
denied = sum(1 for feature in features if feature.get("properties", {}).get("status") == "no_culvert")
print(confirmed, denied, len(features))
PY
)
  if [ "$TOTAL_OBSERVATIONS" -gt 0 ] || { [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; }; then
    MERGE_OBSERVATIONS_ARGS=(
      --observations data/processed/field_observations.geojson
      --output data/processed/training_known_culverts.gpkg
      --csv-output data/processed/training_known_culverts.csv
      --confirmed-output data/processed/confirmed_field_observations.gpkg
      --denied-output data/processed/no_culvert_observations.gpkg
      --denied-csv-output data/processed/no_culvert_observations.csv
      --miss-threshold-m 10
    )
    if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
      MERGE_OBSERVATIONS_ARGS+=(--base-known "$KNOWN_CULVERTS_PATH")
    fi
    if [ "$INCLUDE_FIELD_OBSERVATIONS_AS_POSITIVES" != "1" ]; then
      MERGE_OBSERVATIONS_ARGS+=(--exclude-confirmed)
    fi
    scripts/python.sh -m culvert_ai.cli merge-field-observations "${MERGE_OBSERVATIONS_ARGS[@]}"
    KNOWN_CULVERTS_PATH="data/processed/training_known_culverts.gpkg"
    TRAINING_POINTS_SUMMARY_PATH="data/processed/training_known_culverts.csv"
    if [ "$INCLUDE_FIELD_OBSERVATIONS_AS_POSITIVES" = "1" ] && [ "$CONFIRMED_OBSERVATIONS" -gt 0 ] && [ -f data/processed/confirmed_field_observations.gpkg ]; then
      ADD_OBSERVATION_CANDIDATE_ARGS=(
        --candidates "$CANDIDATES_PATH"
        --field-reports data/processed/confirmed_field_observations.gpkg
        --output data/interim/actual_ulster_candidates_with_field_observations.gpkg
      )
      if [ -f "$BOUNDARY_PATH" ]; then
        ADD_OBSERVATION_CANDIDATE_ARGS+=(--boundary "$BOUNDARY_PATH")
      fi
      scripts/python.sh -m culvert_ai.cli add-field-report-candidates "${ADD_OBSERVATION_CANDIDATE_ARGS[@]}"
      CANDIDATES_PATH="data/interim/actual_ulster_candidates_with_field_observations.gpkg"
    fi
    if [ -f data/processed/no_culvert_observations.gpkg ]; then
      DENIED_CULVERTS_PATH="data/processed/no_culvert_observations.gpkg"
    fi
  fi
fi

FEATURE_ARGS=(
  --candidates "$CANDIDATES_PATH"
  --roads data/raw/roads.gpkg
  --streams data/raw/streams.gpkg
  --density-radii-m 50 100 250 500
  --output data/processed/actual_ulster_unlabeled_features.gpkg
)

if [ -n "$KNOWN_CULVERTS_PATH" ] && [ -f "$KNOWN_CULVERTS_PATH" ]; then
  FEATURE_ARGS+=(--known-culverts "$KNOWN_CULVERTS_PATH" --positive-radius-m 10)
fi
if [ -n "$DENIED_CULVERTS_PATH" ] && [ -f "$DENIED_CULVERTS_PATH" ]; then
  FEATURE_ARGS+=(--negative-culverts "$DENIED_CULVERTS_PATH" --negative-radius-m 10)
fi
if [ -f "$DEM_PATH" ]; then
  FEATURE_ARGS+=(--dem "$DEM_PATH")
fi
if [ -f data/raw/flow_accumulation.tif ]; then
  FEATURE_ARGS+=(--flow-accumulation data/raw/flow_accumulation.tif)
fi
if [ -f data/raw/drainage_area.tif ]; then
  FEATURE_ARGS+=(--drainage-area data/raw/drainage_area.tif)
fi

scripts/python.sh -m culvert_ai.cli build-features "${FEATURE_ARGS[@]}"

scripts/python.sh -m culvert_ai.cli score-unlabeled \
  --features data/processed/actual_ulster_unlabeled_features.gpkg \
  --output data/processed/actual_ulster_unlabeled_predictions.gpkg \
  --csv-output data/processed/actual_ulster_unlabeled_predictions.csv \
  --kml-output data/processed/actual_ulster_evidence_review.kml \
  --kml-max-points 1500

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
      --model-family "${CULVERT_MODEL_FAMILY:-hist_gradient_boosting}" \
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
  --kml-max-points 1500
)

if [ -n "$SUPERVISED_PREDICTIONS_PATH" ]; then
  DISCOVERY_ARGS+=(--supervised-predictions "$SUPERVISED_PREDICTIONS_PATH")
fi
DISCOVERY_ARGS+=(--known-radius-m 10)

scripts/python.sh -m culvert_ai.cli build-discovery-ranking "${DISCOVERY_ARGS[@]}"

if [ -f data/processed/confirmed_field_observations.gpkg ]; then
  scripts/python.sh -m culvert_ai.cli evaluate-success-rate \
    --predictions data/processed/actual_ulster_discovery_predictions.gpkg \
    --actual-culverts data/processed/confirmed_field_observations.gpkg \
    --output reports/field_success_rate_15m.json \
    --max-distance-m 15
fi

scripts/python.sh -m culvert_ai.cli export-web \
  --predictions data/processed/actual_ulster_discovery_predictions.gpkg \
  --output-dir web/data \
  --limit "${WEB_EXPORT_LIMIT:-5000}"

if [ -f data/processed/confirmed_field_observations.gpkg ]; then
  scripts/python.sh -m culvert_ai.cli evaluate-success-rate \
    --predictions web/data/findings.geojson \
    --actual-culverts data/processed/confirmed_field_observations.gpkg \
    --output reports/web_field_success_rate_15m.json \
    --max-distance-m 15
fi

scripts/python.sh scripts/write_model_summary.py \
  --metrics reports/actual_ulster_field_report_metrics.json \
  --point-analysis reports/extracted_points_analysis.json \
  --training-points "$TRAINING_POINTS_SUMMARY_PATH" \
  --output web/data/model_summary.json
