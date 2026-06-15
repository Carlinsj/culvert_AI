#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. The actual default workflow now uses Census TIGER/Line,
# because it is more reliable than live Overpass queries for repeatable research runs.
bash scripts/run_actual_ulster_census_pipeline.sh
