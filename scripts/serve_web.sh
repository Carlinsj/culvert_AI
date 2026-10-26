#!/usr/bin/env bash
set -euo pipefail

RESEARCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "$RESEARCH_SCRIPT_DIR/python.sh" "$RESEARCH_SCRIPT_DIR/serve_research.py" "$@"
