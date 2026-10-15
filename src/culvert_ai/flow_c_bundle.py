"""Validation and integrity checks for promoted FLOW-C inference bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np


BUNDLE_FORMAT_VERSION = "culvert-ai-flow-c-bundle-v1"
FLOW_C_SPECIFICATION_ID = (
    "hydro_c_plus_flow_c__nysdot_ulster_dem_hydrology_v8_20261005"
)
CANDIDATE_GENERATION_VERSION = "v13-interval-baseline"
HYDROLOGY_FEATURE_VERSION = "usgs-3dep-10m-flow-c-v8"
REQUIRED_FEATURE_CRS = "EPSG:32618"
PHYSICAL_SITE_RADIUS_M = 20.0
PREPROCESSING_SPEC = {
    "categorical_handling": "none",
    "infinite_values": "replace_with_nan",
    "missing_value_policy": "full_training_feature_medians",
    "numeric_cast": "float64",
    "scaling": "none",
}

FORBIDDEN_FEATURE_EXACT = {
    "route",
    "route_id",
    "matched_route",
    "road_id",
    "road_name",
    "experiment_route",
    "route_fold",
    "spatial_fold",
    "density_per_km",
    "n_small_culverts",
}
FORBIDDEN_FEATURE_TOKENS = (
    "dist_to_known",
    "known_culvert",
    "official",
    "positive_site",
    "target",
    "label",
    "field_report",
    "feedback",
    "match_distance",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize integrity metadata in one stable representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def feature_schema(feature_columns: list[str]) -> dict:
    return {
        "feature_columns": list(feature_columns),
        "expected_dtypes": {column: "float64" for column in feature_columns},
        "preprocessing": dict(PREPROCESSING_SPEC),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def assert_route_general_feature_columns(columns: list[str]) -> None:
    forbidden = [
        column
        for column in columns
        if column.lower() in FORBIDDEN_FEATURE_EXACT
        or any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            "Frozen FLOW-C bundle contains route identity or inference-forbidden fields: "
            + ", ".join(forbidden)
        )


def validate_frozen_flow_c_bundle(bundle: Mapping) -> Mapping:
    """Fail closed unless a bundle exactly satisfies the promoted v1 contract."""
    if not isinstance(bundle, Mapping):
        raise TypeError("FLOW-C artifact must be a mapping-style model bundle.")
    core_required = {"model", "feature_columns"}
    missing_core = sorted(core_required - set(bundle))
    if missing_core:
        raise ValueError(f"FLOW-C bundle is missing: {missing_core}")

    columns = bundle["feature_columns"]
    if not isinstance(columns, list) or not columns or not all(
        isinstance(column, str) and column for column in columns
    ):
        raise ValueError("FLOW-C feature_columns must be a non-empty ordered string list.")
    if len(columns) != len(set(columns)):
        raise ValueError("FLOW-C feature_columns contains duplicates.")
    assert_route_general_feature_columns(columns)

    required = {
        "bundle_format_version",
        "candidate_generation",
        "code_version",
        "crs_requirements",
        "feature_schema",
        "flow_c_specification",
        "hydrology_feature_version",
        "missing_value_policy",
        "model_class",
        "model_parameters",
        "physical_site_definition",
        "preprocessing",
        "route_z_global",
        "schema_sha256",
        "training_data_provenance",
        "training_date_utc",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Frozen FLOW-C bundle metadata is missing: {missing}")
    if bundle["bundle_format_version"] != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported FLOW-C bundle format version.")
    if bundle["flow_c_specification"] != FLOW_C_SPECIFICATION_ID:
        raise ValueError("The bundle is not the promoted FLOW-C specification.")
    if bundle["hydrology_feature_version"] != HYDROLOGY_FEATURE_VERSION:
        raise ValueError("Incompatible FLOW-C hydrology feature version.")

    candidate = bundle["candidate_generation"]
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("version") != CANDIDATE_GENERATION_VERSION
        or float(candidate.get("interval_m", np.nan)) != 75.0
        or list(candidate.get("lateral_offsets_m", [])) != [0.0]
    ):
        raise ValueError("Incompatible frozen candidate-generation specification.")
    site = bundle["physical_site_definition"]
    if not isinstance(site, Mapping) or float(site.get("radius_m", np.nan)) != 20.0:
        raise ValueError("Incompatible physical-site definition.")
    crs = bundle["crs_requirements"]
    if not isinstance(crs, Mapping) or crs.get("feature_crs") != REQUIRED_FEATURE_CRS:
        raise ValueError("Incompatible FLOW-C feature CRS requirement.")

    if bundle["preprocessing"] != PREPROCESSING_SPEC:
        raise ValueError("Incompatible FLOW-C preprocessing specification.")
    if bundle["missing_value_policy"] != PREPROCESSING_SPEC["missing_value_policy"]:
        raise ValueError("Incompatible FLOW-C missing-value policy.")
    fill = bundle.get("fill_values")
    if not isinstance(fill, Mapping) or list(fill) != columns:
        raise ValueError("FLOW-C median fill schema does not match exact feature ordering.")
    if not all(np.isfinite(float(fill[column])) for column in columns):
        raise ValueError("FLOW-C median fill values must all be finite.")

    expected_schema = feature_schema(columns)
    if bundle["feature_schema"] != expected_schema:
        raise ValueError("FLOW-C feature schema differs from the frozen ordered schema.")
    if bundle["schema_sha256"] != sha256_json(expected_schema):
        raise ValueError("FLOW-C feature schema checksum mismatch.")

    model = bundle["model"]
    model_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
    if bundle["model_class"] != model_class:
        raise ValueError("FLOW-C estimator class does not match bundle metadata.")
    if not hasattr(model, "get_params") or _jsonable(model.get_params(deep=False)) != bundle[
        "model_parameters"
    ]:
        raise ValueError("FLOW-C estimator parameters do not match bundle metadata.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Frozen FLOW-C estimator must implement predict_proba().")
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None or list(model_features) != columns:
        raise ValueError("Fitted estimator feature ordering differs from the frozen schema.")
    classes = getattr(model, "classes_", None)
    if classes is None or list(classes) != [0, 1]:
        raise ValueError("FLOW-C estimator must have binary classes ordered as [0, 1].")

    if not isinstance(bundle["training_data_provenance"], Mapping) or not bundle[
        "training_data_provenance"
    ].get("records"):
        raise ValueError("FLOW-C training-data provenance is incomplete.")
    if not isinstance(bundle["code_version"], Mapping) or not bundle["code_version"]:
        raise ValueError("FLOW-C code/version metadata is incomplete.")

    route_z = bundle["route_z_global"]
    if not isinstance(route_z, Mapping):
        raise ValueError("FLOW-C route_z_global metadata is invalid.")
    center = route_z.get("center")
    scale = route_z.get("scale")
    if not np.isfinite(center) or not np.isfinite(scale) or float(scale) <= 0:
        raise ValueError("FLOW-C global ROUTE-Z fallback statistics are invalid.")
    return bundle


def load_frozen_flow_c_bundle(path_or_bundle: str | Path | Mapping) -> Mapping:
    """Verify an on-disk bundle before deserialization, then validate its contents."""
    if isinstance(path_or_bundle, Mapping):
        return validate_frozen_flow_c_bundle(path_or_bundle)

    path = Path(path_or_bundle)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen FLOW-C bundle not found: {path}")
    if not manifest_path.is_file():
        raise ValueError(f"Frozen FLOW-C integrity manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("FLOW-C integrity manifest has an unsupported format.")
    if manifest.get("bundle_sha256") != sha256_file(path):
        raise ValueError("FLOW-C bundle checksum mismatch.")
    if manifest.get("flow_c_specification") != FLOW_C_SPECIFICATION_ID:
        raise ValueError("FLOW-C integrity manifest identifies a different specification.")

    schema_path = path.parent / str(manifest.get("schema_file", ""))
    if not schema_path.is_file() or manifest.get("schema_sha256") != sha256_file(schema_path):
        raise ValueError("FLOW-C external feature schema is missing or has changed.")
    training_path = path.parent / str(manifest.get("training_records_file", ""))
    if not training_path.is_file() or manifest.get("training_records_sha256") != sha256_file(
        training_path
    ):
        raise ValueError("FLOW-C training-record manifest is missing or has changed.")
    bundle = joblib.load(path)
    validate_frozen_flow_c_bundle(bundle)
    external_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if external_schema != bundle["feature_schema"]:
        raise ValueError("External and bundled FLOW-C feature schemas differ.")
    return bundle
