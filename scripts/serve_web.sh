#!/usr/bin/env bash
set -euo pipefail

cd web
python3 -m http.server 8080 --bind 127.0.0.1
