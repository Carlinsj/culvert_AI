#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
fi

MISSING="$(
  .venv/bin/python - <<'PY'
import importlib.util

modules = [
    "geopandas",
    "joblib",
    "numpy",
    "pandas",
    "pytest",
    "rasterio",
    "sklearn",
    "shapely",
    "yaml",
]

missing = [module for module in modules if importlib.util.find_spec(module) is None]
print(" ".join(missing))
PY
)"

if [ -n "$MISSING" ]; then
  echo "Installing missing Python dependencies: $MISSING"
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CACHE_DIR="$ROOT_DIR/.cache/pip" \
    .venv/bin/python -m pip install -r requirements.txt "pytest>=7.4" "ruff>=0.4"
else
  echo "Python dependencies already available."
fi
