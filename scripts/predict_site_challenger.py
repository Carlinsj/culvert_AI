"""Research-only inference for the frozen site challenger; never fits a model.

Input is a GeoPackage/GeoParquet with the exact 48 existing numeric features,
candidate_id, geometry in EPSG:32618, and a label-free route/road identifier.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import joblib

from culvert_ai.flow_c_bundle import sha256_file


def load_runtime():
    source = Path(__file__).with_name("train_site_challenger.py")
    spec = importlib.util.spec_from_file_location("site_challenger_runtime", source)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def predict(features, artifact, route_column="route_id"):
    artifact = Path(artifact)
    manifest = json.loads((artifact.parent / "manifest.json").read_text())
    if sha256_file(artifact) != manifest["files"][artifact.name]:
        raise ValueError("Experimental model artifact checksum mismatch")
    bundle = joblib.load(artifact)
    required = set(bundle["feature_columns"]) | {"candidate_id", route_column}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing frozen input fields: {sorted(missing)}")
    if not features.candidate_id.is_unique:
        raise ValueError("Candidate IDs must be unique")
    if features.crs is None or features.crs.to_string() != bundle["feature_crs"]:
        raise ValueError(f"Feature geometry must use {bundle['feature_crs']}")
    runtime = load_runtime()
    frame = features.copy().reset_index(drop=True)
    frame["route"] = frame[route_column].fillna("<unassigned>").astype(str)
    frame["raw_score"] = runtime.score_model(bundle, frame, bundle["feature_columns"])
    metric = frame.to_crs(bundle["evaluation_crs"])
    frame["x_m"], frame["y_m"] = metric.geometry.x, metric.geometry.y
    frame["priority"] = runtime.normalize(frame, bundle["normalization"], bundle["normalization_fallback"])
    sites = runtime.saved_priority_sites(frame, "priority")
    sites["status"] = "experimental_review"
    sites["field_message"] = bundle["message"]
    return gpd.GeoDataFrame(sites, geometry="geometry", crs=features.crs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-column", default="route_id")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    features = gpd.read_parquet(args.features) if args.features.suffix == ".parquet" else gpd.read_file(args.features)
    sites = predict(features, args.model, args.route_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sites.to_file(args.output, driver="GPKG")
    print(f"Wrote {len(sites)} experimental physical-site recommendations")


if __name__ == "__main__":
    main()
