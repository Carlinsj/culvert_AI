#!/usr/bin/env python3
"""Create a validated, server-only FLOW-C runtime package."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "flow_c_frozen_v1.joblib"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "flow_c_frozen_v1-runtime.zip")
    args = parser.parse_args()
    model = args.model.resolve()
    load_frozen_flow_c_bundle(model)
    required = [
        model,
        model.with_suffix(model.suffix + ".manifest.json"),
        model.with_suffix(model.suffix + ".schema.json"),
        model.with_suffix(model.suffix + ".training_records.csv"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in required:
            archive.write(path, arcname=f"models/{path.name}")
    print(f"Created validated server runtime package: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
