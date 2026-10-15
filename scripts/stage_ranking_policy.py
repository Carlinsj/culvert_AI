"""Offline staging inference; leaves the production service and model untouched."""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file  # noqa: E402
from staging.ranking_policy import (  # noqa: E402
    RankingPolicy, export_sites, load_policy, predict_from_features,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True,
                        help="Existing frozen-pipeline features in native EPSG:32618, including route_id")
    parser.add_argument("--config", type=Path, default=ROOT / "staging/ranking-policy.json")
    parser.add_argument("--policy", choices=[p.value for p in RankingPolicy], help="Explicit staging override")
    parser.add_argument("--output", type=Path, required=True, help="New staging export directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory; existing artifacts are never overwritten.")
    policy = RankingPolicy(args.policy) if args.policy else load_policy(args.config)
    model = ROOT / "models/flow_c_frozen_v1.joblib"
    bundle = load_frozen_flow_c_bundle(model)
    if bundle["model"].get_params()["n_estimators"] != 800 or len(bundle["feature_columns"]) != 48:
        raise ValueError("Expected the exact frozen 800-tree, 48-feature RF.")
    features = (gpd.read_parquet(args.features) if args.features.suffix == ".parquet"
                else gpd.read_file(args.features))
    sites = predict_from_features(features, model, policy=policy)
    export_sites(sites, args.output)
    manifest = dict(environment="staging", ranking_policy=policy.value,
                    model_sha256=sha256_file(model), features_sha256=sha256_file(args.features),
                    candidates=len(features), representatives=len(sites), fit_calls=0,
                    feature_columns=bundle["feature_columns"], production_modified=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
