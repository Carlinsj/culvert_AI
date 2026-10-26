"""Load the isolated v3 artifact and score a compatible candidate parquet."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from staging.model_sprint import VERSION, numeric_matrix, rank_sites


DEFAULT_MODEL = ROOT / "models/experiments/culvert_retrieval_v3_20261017/model.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policy", choices=["raw", "soft_local_diversity"], default="raw")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if bundle.get("version") != VERSION or not bundle.get("experimental"):
        raise ValueError("Only the isolated experimental v3 bundle is accepted")
    frame = gpd.read_parquet(args.input)
    if frame.crs is None or frame.crs.to_epsg() != 32618:
        raise ValueError("Candidate geometry must use EPSG:32618 metre coordinates")
    missing = set(bundle["feature_columns"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate feature contract is incomplete: {sorted(missing)}")
    frame["score_v3"] = bundle["pipeline"].predict_proba(
        numeric_matrix(frame, bundle["feature_columns"])
    )[:, 1]
    if not np.isfinite(frame.score_v3).all():
        raise ValueError("Model emitted non-finite scores")
    ranked = rank_sites(frame, "score_v3", args.policy).frame
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
