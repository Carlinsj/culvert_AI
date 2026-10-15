"""Regression checks for evaluation mechanics, not model selection."""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_inventory_heldout_roads import (  # noqa: E402
    coverage, evaluate, fold_for, match_one_to_one, nearest_lines, safe_training_frame,
)


def points(x):
    return gpd.GeoDataFrame(geometry=[Point(v, 0) for v in x], crs=32618)


def test_matching_maximizes_cardinality_instead_of_nearest_greedy():
    sites = points([10, -15])
    # First site can match either record, second only record 0.
    truth = np.array([[0., 0.], [25., 0.]])
    result = match_one_to_one(sites, truth)
    assert len(result) == 2
    assert result.real_row.is_unique and result.site_row.is_unique
    pd.testing.assert_frame_equal(result, match_one_to_one(sites, truth))


def test_twenty_meter_boundary_and_reused_ids_do_not_merge_records():
    sites = points([0, 45])
    truth = np.array([[20., 0.], [65., 0.]])
    assert len(match_one_to_one(sites, truth)) == 2
    assert len(match_one_to_one(sites.iloc[:1], truth)) == 1
    assert len(match_one_to_one(sites, np.array([[20.001, 0.]]))) == 0


def test_empty_candidates_are_reported_as_no_coverage():
    assert np.isinf(coverage(points([]), points([0, 1]))).all()
    assert match_one_to_one(points([]), np.array([[0., 0.]])).empty


def test_matching_competing_records_counts_capacity_loss():
    # Distinct inventory records remain distinct even at identical coordinates.
    result = match_one_to_one(points([0]), np.array([[0., 0.], [0., 0.]]))
    assert len(result) == 1


def test_split_rule_is_stable_and_corridor_complete():
    routes = ["212", "32", "unseen-road", "375"]
    assert [fold_for(r) for r in routes] == [fold_for(r) for r in routes]
    assert all(0 <= fold_for(r) < 3 for r in routes)


def test_full_line_buffer_excludes_between_candidate_positions():
    roads = gpd.GeoDataFrame(dict(corridor=["held", "train"], geometry=[
        LineString([(0, 0), (1000, 0)]), LineString([(0, 300), (1000, 300)]),
    ]), crs=32618)
    frame = gpd.GeoDataFrame(dict(corridor=["train"]*4 + ["held"], geometry=[
        Point(37, 99), Point(37, 100), Point(37, 101), Point(200, 300), Point(0, 500),
    ]), crs=32618)
    result = safe_training_frame(frame, roads, {"held"})
    assert len(result) == 2
    assert result.distance_to_heldout_road_m.gt(100).all()
    assert not result.corridor.eq("held").any()


def test_geometry_ties_break_by_segment_identity():
    roads = gpd.GeoDataFrame(dict(segment_id=["b", "a"], geometry=[
        LineString([(0, 0), (10, 0)]), LineString([(0, 0), (10, 0)]),
    ]), crs=32618)
    assert nearest_lines(points([5]), roads).road_id.tolist() == ["a"]


def test_budget_uses_available_site_count_and_unmatched_is_not_false_positive():
    sites = points([0, 100])
    sites["corridor"] = ["A", "B"]
    truth = points([0])
    result = evaluate(sites, sites, truth, dict(model="RF"))
    row = next(r for r in result if r["k"] == 250)
    assert row["actual_denominator"] == 2
    assert row["matched"] == 1 and row["inventory_matched_fraction"] == .5
    assert row["inventory_unmatched_recommendations"] == 1
    assert "false_positive" not in row
