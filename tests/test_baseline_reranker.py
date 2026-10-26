"""Inference compatibility and fail-closed checks for the v5 reranker."""

import geopandas as gpd
import numpy as np
import pytest

from staging.baseline_reranker import load_bundle, rank


def test_v5_bundle_replays_saved_queue_exactly():
    candidates = gpd.read_parquet("reports/experiments/physical_discovery_20261016/variant_BASELINE.parquet")
    expected = gpd.read_parquet("reports/experiments/baseline_preserving_v5_20261019/ranked_sites.parquet")
    actual = rank(candidates, load_bundle())
    assert actual.candidate_id.tolist() == expected.candidate_id.tolist()
    np.testing.assert_allclose(actual.score_v5, expected.score_v5, rtol=0, atol=1e-15)
    assert actual.global_rank.tolist() == list(range(1, len(actual) + 1))


def test_v5_inference_fails_closed_without_original_rf_score():
    candidates = gpd.read_parquet("reports/experiments/physical_discovery_20261016/variant_BASELINE.parquet").head(10).drop(columns="score_RF")
    with pytest.raises(ValueError, match="score_RF"):
        rank(candidates, load_bundle())


def test_v5_inference_never_changes_candidate_geometry_or_membership():
    candidates = gpd.read_parquet("reports/experiments/physical_discovery_20261016/variant_BASELINE.parquet").head(200).copy()
    before = candidates.set_index("candidate_id").geometry.to_wkb()
    actual = rank(candidates, load_bundle()).set_index("candidate_id")
    assert set(actual.index) == set(before.index)
    assert actual.geometry.to_wkb().sort_index().equals(before.sort_index())
