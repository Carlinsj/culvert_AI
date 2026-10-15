"""Fixed-policy regression tests: no fitting and no production configuration edits."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.inference import RouteZConfig, dynamic_route_z, rank_flow_c_candidates  # noqa: E402
from staging.ranking_policy import (  # noqa: E402
    WARNING, RankingPolicy, export_sites, load_policy, predict_from_features, rank_candidates,
)


def candidates():
    return gpd.GeoDataFrame(dict(
        candidate_id=["b", "a", "c", "d", "e", "f", "g", "h"],
        route_id=["unseen-X", "unseen-Y"] * 4,
        flow_c_score=[.9, .9, .8, .7, .6, .5, .4, .3],
        geometry=[Point(500000+x, 4600000) for x in (0, 20, 40.001, 80, 120, 160, 200, 240)],
    ), crs=32618)


def test_raw_ties_boundary_cross_road_suppression_and_no_mutation():
    frame = candidates()
    original = frame.copy()
    sites = rank_candidates(frame, policy="raw_rf")
    assert sites.candidate_id.tolist() == ["a", "c", "d", "e", "f", "g", "h"]
    assert sites.final_priority.equals(sites.flow_c_score)
    assert sites.global_rank.tolist() == list(range(1, len(sites)+1))
    pd.testing.assert_frame_equal(frame, original)
    for seed in range(5):
        pd.testing.assert_frame_equal(sites, rank_candidates(frame.sample(frac=1, random_state=seed), policy="raw_rf"))


def test_current_is_default_and_exact_rollback():
    frame = candidates()
    config = RouteZConfig(global_center=.5, global_scale=.25)
    before = rank_candidates(frame, route_z_config=config)
    expected = rank_flow_c_candidates(frame, route_z_config=config)
    pd.testing.assert_frame_equal(before[expected.columns], expected)
    raw = rank_candidates(frame, policy="raw_rf", route_z_config=config)
    pd.testing.assert_frame_equal(before, rank_candidates(frame, route_z_config=config))
    assert not set(before.staging_site_key) & set(raw.staging_site_key)
    diagnostics = dynamic_route_z(frame, config=config).set_index("candidate_id")
    assert raw.route_z_score.tolist() == raw.candidate_id.map(diagnostics.route_z_score).tolist()


def test_raw_order_independent_of_route_ids_and_normalization():
    frame = candidates()
    expected = rank_candidates(frame, policy="raw_rf").candidate_id.tolist()
    frame["route_id"] = [f"arbitrary-{i}" for i in range(len(frame))]
    sites = rank_candidates(frame, policy="raw_rf", route_z_config=RouteZConfig(global_center=-20, global_scale=5))
    assert sites.candidate_id.tolist() == expected


@pytest.mark.parametrize("policy", list(RankingPolicy))
def test_export_round_trip_and_no_truth_leakage(tmp_path, policy):
    frame = candidates()
    frame["ground_truth"] = 1
    sites = rank_candidates(frame, policy=policy)
    sites["ground_truth"] = 1
    folder = tmp_path / policy.value
    export_sites(sites, folder)
    csv = pd.read_csv(folder / "sites.csv")
    geo = gpd.read_file(folder / "sites.geojson")
    assert "ground_truth" not in csv and "ground_truth" not in geo
    assert csv.candidate_id.tolist() == geo.candidate_id.tolist() == sites.candidate_id.tolist()
    assert csv.ranking_policy.eq(policy.value).all()
    assert csv.field_verification_notice.eq(WARNING).all()
    np.testing.assert_allclose(geo.geometry.x, csv.longitude, rtol=0, atol=1e-10)
    np.testing.assert_allclose(geo.geometry.y, csv.latitude, rtol=0, atol=1e-10)
    with pytest.raises(FileExistsError):
        export_sites(sites, folder)


def test_config_is_reversible_and_fail_closed(tmp_path):
    assert load_policy(ROOT / "staging/ranking-policy.json") is RankingPolicy.CURRENT
    path = tmp_path / "policy.json"
    for policy in RankingPolicy:
        path.write_text(json.dumps(dict(environment="staging", ranking_policy=policy.value)))
        assert load_policy(path) is policy
    path.write_text(json.dumps(dict(environment="production", ranking_policy="raw_rf")))
    with pytest.raises(ValueError):
        load_policy(path)
    with pytest.raises(ValueError):
        rank_candidates(candidates(), policy="typo")


@pytest.mark.parametrize("mutation", ["duplicates", "nan_score", "infinite_score", "crs", "null_id", "geometry"])
def test_invalid_inputs_fail_closed(mutation):
    frame = candidates()
    if mutation == "duplicates":
        frame.loc[0, "candidate_id"] = "a"
    elif mutation == "nan_score":
        frame.loc[0, "flow_c_score"] = np.nan
    elif mutation == "infinite_score":
        frame.loc[0, "flow_c_score"] = np.inf
    elif mutation == "crs":
        frame = frame.to_crs(4326)
    elif mutation == "null_id":
        frame.loc[0, "candidate_id"] = None
    else:
        frame.loc[0, "geometry"] = None
    with pytest.raises(ValueError):
        rank_candidates(frame, policy="raw_rf")


def test_empty_population():
    for policy in RankingPolicy:
        with pytest.raises(ValueError, match="At least one"):
            rank_candidates(candidates().iloc[:0], policy=policy)


def test_replacement_model_rejected_before_loading(tmp_path):
    replacement = tmp_path / "replacement.joblib"
    replacement.write_bytes(b"not the frozen artifact")
    with pytest.raises(ValueError, match="exact frozen RF"):
        predict_from_features(candidates(), replacement, policy="raw_rf")


def test_actual_frozen_model_arbitrary_ids_no_fit(monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("No training is permitted in staging inference.")

    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    fixture = pd.read_parquet(ROOT / "reports/experiments/site_challenger_20261015/development_features.parquet").head(12)
    native = gpd.read_file(ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg", columns=["candidate_id", "geometry"])
    geometry = native.set_index("candidate_id").loc[fixture.candidate_id].geometry.reset_index(drop=True)
    features = gpd.GeoDataFrame(fixture.reset_index(drop=True), geometry=geometry, crs=32618)
    features["route_id"] = [f"UNSEEN-{i % 3}" for i in range(len(features))]
    features["candidate_id"] = [f"arbitrary-{i}" for i in range(len(features))]
    for policy in RankingPolicy:
        result = predict_from_features(features, ROOT / "models/flow_c_frozen_v1.joblib", policy=policy)
        assert set(result.route_id) == set(features.route_id)
        assert set(result.candidate_id).issubset(set(features.candidate_id))
