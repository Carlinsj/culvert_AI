import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from culvert_ai.inference import (
    RouteZConfig,
    dynamic_route_z,
    filter_predictions_by_route,
    predict_culverts_from_features,
    rank_flow_c_candidates,
    score_frozen_flow_c,
)
from culvert_ai.flow_c_bundle import (
    BUNDLE_FORMAT_VERSION,
    CANDIDATE_GENERATION_VERSION,
    FLOW_C_SPECIFICATION_ID,
    HYDROLOGY_FEATURE_VERSION,
    PREPROCESSING_SPEC,
    REQUIRED_FEATURE_CRS,
    feature_schema,
    sha256_file,
    sha256_json,
)


class DummyFlowC:
    feature_names_in_ = np.asarray(["signal"])
    classes_ = np.asarray([0, 1])

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Inference must never call fit().")

    def get_params(self, deep=False):
        return {}

    def predict_proba(self, frame):
        probability = frame["signal"].to_numpy(float)
        return np.column_stack([1.0 - probability, probability])


def _points(records):
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32618")


def _bundle():
    columns = ["signal"]
    schema = feature_schema(columns)
    model = DummyFlowC()
    return {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "flow_c_specification": FLOW_C_SPECIFICATION_ID,
        "model": model,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_parameters": {},
        "feature_columns": columns,
        "feature_schema": schema,
        "schema_sha256": sha256_json(schema),
        "fill_values": {"signal": 0.0},
        "preprocessing": dict(PREPROCESSING_SPEC),
        "missing_value_policy": PREPROCESSING_SPEC["missing_value_policy"],
        "candidate_generation": {
            "version": CANDIDATE_GENERATION_VERSION,
            "interval_m": 75.0,
            "lateral_offsets_m": [0.0],
        },
        "hydrology_feature_version": HYDROLOGY_FEATURE_VERSION,
        "physical_site_definition": {"radius_m": 20.0},
        "crs_requirements": {"feature_crs": REQUIRED_FEATURE_CRS},
        "route_z_global": {"center": 0.5, "scale": 0.25},
        "training_data_provenance": {"records": 2},
        "training_date_utc": "2026-10-13T00:00:00+00:00",
        "code_version": {"test": True},
    }


def test_dynamic_route_z_uses_current_route_distribution():
    scores = np.arange(8, dtype=float)
    frame = pd.DataFrame({"route_id": ["unseen"] * 8, "flow_c_score": scores})

    result = dynamic_route_z(frame)

    assert set(result["route_z_method"]) == {"route_mean_std"}
    assert result["route_z_score"].mean() == pytest.approx(0.0)
    assert result["route_z_score"].std(ddof=0) == pytest.approx(1.0)
    assert set(result["route_z_candidate_count"]) == {8}


def test_dynamic_route_z_uses_robust_rule_for_small_nonconstant_route():
    frame = pd.DataFrame(
        {"route_id": ["short"] * 4, "flow_c_score": [0.1, 0.2, 0.4, 0.9]}
    )

    result = dynamic_route_z(frame)

    assert set(result["route_z_method"]) == {"route_median_iqr"}
    assert result["route_z_center"].unique().tolist() == pytest.approx([0.3])


def test_dynamic_route_z_uses_frozen_global_fallback_for_tiny_route():
    frame = pd.DataFrame(
        {"route_id": ["tiny", "tiny"], "flow_c_score": [0.4, 0.8]}
    )

    result = dynamic_route_z(
        frame,
        config=RouteZConfig(global_center=0.5, global_scale=0.25),
    )

    assert set(result["route_z_method"]) == {"frozen_global_fallback"}
    assert result["route_z_score"].tolist() == pytest.approx([-0.4, 1.2])


def test_dynamic_route_z_zero_variance_is_neutral_without_global_statistics():
    frame = pd.DataFrame(
        {"route_id": ["flat"] * 10, "flow_c_score": [0.7] * 10}
    )

    result = dynamic_route_z(frame)

    assert set(result["route_z_method"]) == {"neutral_fallback"}
    assert result["route_z_score"].tolist() == [0.0] * 10


def test_global_ranking_suppresses_one_physical_site_across_routes_without_quotas():
    scored = _points(
        [
            {
                "candidate_id": "a-high",
                "route_id": "A",
                "flow_c_score": 0.95,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "b-duplicate",
                "route_id": "B",
                "flow_c_score": 0.80,
                "geometry": Point(10, 0),
            },
            {
                "candidate_id": "a-second",
                "route_id": "A",
                "flow_c_score": 0.90,
                "geometry": Point(100, 0),
            },
        ]
    )

    ranked = rank_flow_c_candidates(
        scored,
        route_z_config=RouteZConfig(global_center=0.5, global_scale=0.25),
    )

    assert ranked["candidate_id"].tolist() == ["a-high", "a-second"]
    assert ranked["global_rank"].tolist() == [1, 2]
    assert ranked["physical_site_id"].is_unique
    assert set(ranked["route_id"]) == {"A"}


def test_frozen_flow_c_rejects_route_identity_and_label_leakage():
    features = _points(
        [{"candidate_id": "x", "route_id": "A", "signal": 0.6, "geometry": Point(0, 0)}]
    )

    with pytest.raises(ValueError, match="route identity"):
        score_frozen_flow_c(
            features,
            {"model": DummyFlowC(), "feature_columns": ["signal", "route_id"]},
        )
    with pytest.raises(ValueError, match="inference-forbidden"):
        score_frozen_flow_c(
            features,
            {"model": DummyFlowC(), "feature_columns": ["signal", "known_culvert_distance"]},
        )


def test_feature_boundary_scores_without_refitting_and_supports_post_filtering():
    features = _points(
        [
            {"candidate_id": "a", "route_id": "new-1", "signal": 0.2, "geometry": Point(0, 0)},
            {"candidate_id": "b", "route_id": "new-2", "signal": 0.9, "geometry": Point(100, 0)},
        ]
    )
    bundle = _bundle()

    predictions = predict_culverts_from_features(features, flow_c_model=bundle)
    filtered = filter_predictions_by_route(predictions, "new-1")

    required = {
        "global_rank", "route_id", "latitude", "longitude", "candidate_id",
        "physical_site_id", "flow_c_score", "route_z_score", "final_priority",
    }
    assert required.issubset(predictions.columns)
    assert predictions["candidate_id"].tolist() == ["b", "a"]
    assert filtered["candidate_id"].tolist() == ["a"]
    assert filtered.iloc[0]["global_rank"] == 2


def test_bundle_schema_and_crs_fail_closed():
    features = _points(
        [{"candidate_id": "x", "route_id": "new", "signal": 0.4, "geometry": Point(0, 0)}]
    )
    wrong_schema = _bundle()
    wrong_schema["feature_schema"] = {
        **wrong_schema["feature_schema"],
        "feature_columns": ["different"],
    }
    with pytest.raises(ValueError, match="feature schema differs"):
        score_frozen_flow_c(features, wrong_schema)

    wrong_crs = features.to_crs("EPSG:3857")
    with pytest.raises(ValueError, match="must use the frozen EPSG:32618"):
        score_frozen_flow_c(wrong_crs, _bundle())


def test_bundle_path_checks_integrity_before_scoring(tmp_path):
    bundle_path = tmp_path / "flow_c.joblib"
    schema_path = tmp_path / "flow_c.joblib.schema.json"
    training_path = tmp_path / "flow_c.joblib.training_records.csv"
    manifest_path = tmp_path / "flow_c.joblib.manifest.json"
    bundle = _bundle()
    schema_path.write_text(json.dumps(bundle["feature_schema"]), encoding="utf-8")
    training_path.write_text("candidate_id\nx\n", encoding="utf-8")
    joblib.dump(bundle, bundle_path)
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "flow_c_specification": FLOW_C_SPECIFICATION_ID,
                "bundle_sha256": sha256_file(bundle_path),
                "schema_file": schema_path.name,
                "schema_sha256": sha256_file(schema_path),
                "training_records_file": training_path.name,
                "training_records_sha256": sha256_file(training_path),
            }
        ),
        encoding="utf-8",
    )
    features = _points(
        [{"candidate_id": "x", "route_id": "unseen", "signal": 0.4, "geometry": Point(0, 0)}]
    )

    scored, _ = score_frozen_flow_c(features, bundle_path)
    assert scored["flow_c_score"].tolist() == pytest.approx([0.4])

    with bundle_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        score_frozen_flow_c(features, bundle_path)


def test_required_flow_c_feature_is_never_synthesized_at_inference():
    features = _points(
        [{"candidate_id": "x", "route_id": "unseen", "geometry": Point(0, 0)}]
    )
    with pytest.raises(ValueError, match="did not produce every frozen FLOW-C field"):
        score_frozen_flow_c(features, _bundle())
