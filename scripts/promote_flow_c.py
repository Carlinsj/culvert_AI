"""Fit exactly one final estimator for the already-selected FLOW-C specification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from culvert_ai.flow_c_bundle import (
    BUNDLE_FORMAT_VERSION,
    CANDIDATE_GENERATION_VERSION,
    FLOW_C_SPECIFICATION_ID,
    HYDROLOGY_FEATURE_VERSION,
    PHYSICAL_SITE_RADIUS_M,
    PREPROCESSING_SPEC,
    REQUIRED_FEATURE_CRS,
    feature_schema,
    sha256_file,
    sha256_json,
    validate_frozen_flow_c_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
V8_DIR = ROOT / "reports/experiments/nysdot_ulster_dem_hydrology_v8_20261005"
V7_RUNNER = ROOT / "reports/experiments/nysdot_ulster_dem_hydrology_v7_completed_20260924/run_experiment.py"
FEATURE_LIST_PATH = V8_DIR / "exact_feature_lists.json"
MODEL_SPEC_PATH = V8_DIR / "control_model_specification.json"
RUN_CONFIG_PATH = V8_DIR / "run_config.json"
TRAINING_ROWS_PATH = V8_DIR / "training_rows.csv"
OOF_PATH = V8_DIR / "hydro_c_plus_flow_c__whole_route__oof_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "models/flow_c_frozen_v1.joblib"


def _load_runner():
    spec = importlib.util.spec_from_file_location("culvert_flow_c_promotion_v7", V7_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen experiment runner: {V7_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _model_parameters(model) -> dict:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in model.get_params(deep=False).items()
    }


def promote(output: Path) -> dict:
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing promoted bundle: {output}")

    feature_groups = _json(FEATURE_LIST_PATH)
    run_config = _json(RUN_CONFIG_PATH)
    model_spec = _json(MODEL_SPEC_PATH)
    columns = list(feature_groups["hydro_c_plus_flow_c"])
    if run_config["diagnostic_strongest_configuration"] != "hydro_c_plus_flow_c":
        raise RuntimeError("The preserved experiment does not identify FLOW-C as selected.")
    if columns != run_config["feature_groups"]["hydro_c_plus_flow_c"]:
        raise RuntimeError("The two preserved ordered FLOW-C schemas differ.")
    if model_spec["model_family"] != "RandomForestClassifier":
        raise RuntimeError("The preserved FLOW-C estimator class is not recoverable exactly.")
    if model_spec["seed"] != 42 or run_config["random_state"] != 42:
        raise RuntimeError("The preserved FLOW-C random seed is inconsistent.")

    runner = _load_runner()
    data = runner.V6._prepare_experiment()
    features = runner._add_dem_features(data["features"])
    training_manifest = pd.read_csv(TRAINING_ROWS_PATH, dtype={"candidate_id": str})
    ids = training_manifest["candidate_id"].tolist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("The preserved FLOW-C training record manifest has duplicate IDs.")
    index = features.assign(candidate_id=features["candidate_id"].astype(str)).set_index(
        "candidate_id", drop=False
    )
    if not set(ids).issubset(index.index):
        raise RuntimeError("A preserved FLOW-C training record cannot be reconstructed.")
    training = index.loc[ids].copy()
    reconstructed_ids = data["selected"]["candidate_id"].astype(str).tolist()
    if ids != reconstructed_ids:
        raise RuntimeError("Preserved and reconstructed FLOW-C training row ordering differs.")
    if training["experiment_target"].astype(int).tolist() != training_manifest[
        "experiment_target"
    ].astype(int).tolist():
        raise RuntimeError("Preserved and reconstructed FLOW-C labels differ.")

    fill_values = runner.V4._feature_fill_values(training, columns)
    x = runner.V4._prepare_features(training, columns, fill_values)
    y = training["experiment_target"].astype(int)
    estimator = clone(runner.V4._candidate_models(42)["random_forest"])
    recovered_parameters = _model_parameters(estimator)
    if recovered_parameters != model_spec["parameters"]:
        raise RuntimeError("Recovered estimator parameters differ from the audited specification.")

    # This is the one and only fit performed by model promotion.
    estimator.fit(x, y)

    oof = pd.read_parquet(OOF_PATH, columns=["probability"])
    probabilities = pd.to_numeric(oof["probability"], errors="coerce").dropna()
    global_center = float(probabilities.mean())
    global_scale = float(probabilities.std(ddof=0))
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise RuntimeError("The preserved whole-route OOF scores cannot support ROUTE-Z fallback.")

    schema = feature_schema(columns)
    schema_digest = sha256_json(schema)
    labels = training_manifest["training_label_class"].astype(str).value_counts().sort_index()
    sources = training_manifest["experiment_label_source"].astype(str).value_counts().sort_index()
    bundle = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "flow_c_specification": FLOW_C_SPECIFICATION_ID,
        "model": estimator,
        "model_class": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
        "model_parameters": recovered_parameters,
        "feature_columns": columns,
        "feature_schema": schema,
        "schema_sha256": schema_digest,
        "fill_values": fill_values,
        "preprocessing": dict(PREPROCESSING_SPEC),
        "missing_value_policy": PREPROCESSING_SPEC["missing_value_policy"],
        "candidate_generation": {
            "version": CANDIDATE_GENERATION_VERSION,
            "interval_m": 75.0,
            "interval_point_rule": "37.5 m then every 75 m along each road line part",
            "lateral_offsets_m": [0.0],
            "coordinates_changed_from_selected_v8_pool": False,
        },
        "hydrology_feature_version": HYDROLOGY_FEATURE_VERSION,
        "physical_site_definition": {
            "radius_m": PHYSICAL_SITE_RADIUS_M,
            "application": "global greedy consolidation after FLOW-C and ROUTE-Z sorting",
        },
        "crs_requirements": {
            "feature_crs": REQUIRED_FEATURE_CRS,
            "linear_units": "metre",
            "hydrology_grid_resolution_m": 10.0,
        },
        "route_z_global": {
            "center": global_center,
            "scale": global_scale,
            "method": "mean_and_population_std",
            "source": str(OOF_PATH.relative_to(ROOT)),
            "source_sha256": sha256_file(OOF_PATH),
            "score_rows": int(len(probabilities)),
            "labels_or_match_distances_used": False,
        },
        "training_data_provenance": {
            "records": int(len(training)),
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
            "ordered_candidate_ids": ids,
            "ordered_candidate_ids_sha256": sha256_json(ids),
            "training_label_counts": {key: int(value) for key, value in labels.items()},
            "training_label_source_counts": {key: int(value) for key, value in sources.items()},
            "negative_strategy": run_config["negative_strategy"],
            "positive_weighting": run_config["positive_weighting"],
            "sample_weighting": "none",
            "training_label_match_rule": "selected V8 labels, including official matches <=25 m",
            "positive_site_grouping": "preserved 25 m V8 training-site clusters",
            "deployment_site_consolidation_m": 20.0,
            "training_rows_source": str(TRAINING_ROWS_PATH.relative_to(ROOT)),
            "training_rows_source_sha256": sha256_file(TRAINING_ROWS_PATH),
            "feature_source": "data/processed/actual_ulster_unlabeled_features.gpkg",
            "feature_source_sha256": sha256_file(
                ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg"
            ),
            "dem_hydrology_feature_source": (
                "data/processed/hydrology/usgs_3dep_10m/candidate_hydrology_features.parquet"
            ),
            "dem_hydrology_feature_source_sha256": sha256_file(
                ROOT
                / "data/processed/hydrology/usgs_3dep_10m/candidate_hydrology_features.parquet"
            ),
        },
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": {
            "git_commit": _git_commit(),
            "experiment_id": run_config["experiment_id"],
            "experiment_environment": _json(V8_DIR / "environment_versions.json"),
            "promotion_script_sha256": sha256_file(Path(__file__)),
            "frozen_v7_runner_sha256": sha256_file(V7_RUNNER),
            "inference_module_sha256": sha256_file(ROOT / "src/culvert_ai/inference.py"),
        },
    }
    validate_frozen_flow_c_bundle(bundle)

    output.parent.mkdir(parents=True, exist_ok=True)
    schema_path = output.with_suffix(output.suffix + ".schema.json")
    training_path = output.with_suffix(output.suffix + ".training_records.csv")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(TRAINING_ROWS_PATH, training_path)
    joblib.dump(bundle, output, compress=3)
    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "flow_c_specification": FLOW_C_SPECIFICATION_ID,
        "bundle_file": output.name,
        "bundle_sha256": sha256_file(output),
        "schema_file": schema_path.name,
        "schema_sha256": sha256_file(schema_path),
        "training_records_file": training_path.name,
        "training_records_sha256": sha256_file(training_path),
        "feature_schema_content_sha256": schema_digest,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest | {
        "training_rows": len(training),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "feature_count": len(columns),
        "route_z_global": bundle["route_z_global"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(promote(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
