"""Density audit invariants, independent of the official inventory outcomes."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from scripts.validate_density_controls import (
    adjacency,
    cardinality,
    conditional_order,
    prefix_matches,
    stream_seed,
    summary,
    uniform,
)


@pytest.fixture
def roads():
    return gpd.GeoDataFrame(
        dict(
            id=["opaque-A", "new-road-Z"],
            corridor=["A", "Z"],
            geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 100), (13, 100)])],
        ),
        crs=32618,
    )


def test_uniform_half_offset_and_short_midpoint(roads):
    points = uniform(roads, spacing=20)
    assert points.loc[points.road_id.eq("opaque-A"), "road_chainage_m"].tolist() == [
        10,
        30,
        50,
        70,
        90,
    ]
    assert points.loc[points.road_id.eq("new-road-Z"), "road_chainage_m"].tolist() == [10]
    short = uniform(roads, spacing=75)
    assert short.loc[short.road_id.eq("new-road-Z"), "road_chainage_m"].tolist() == [6.5]


@pytest.mark.parametrize("budget", [1, 10, 100])
def test_exact_budget_is_label_free_deterministic_and_input_order_invariant(roads, budget):
    a = uniform(roads, budget=budget)
    b = uniform(roads.iloc[::-1], budget=budget)
    assert len(a) == budget and a.candidate_id.is_unique
    assert a.candidate_id.tolist() == b.candidate_id.tolist()
    np.testing.assert_array_equal(
        np.c_[a.geometry.x, a.geometry.y], np.c_[b.geometry.x, b.geometry.y]
    )


def test_incremental_matching_resolves_capacity_conflicts():
    edges = [[0, 1], [0], [1, 2], [], [2]]
    assert prefix_matches(edges, [0, 1], (1, 2)).tolist() == [1, 2]
    for seed in range(20):
        order = np.random.default_rng(seed).permutation(5)
        for k, value in zip(range(1, 6), prefix_matches(edges, order, range(1, 6))):
            assert value == cardinality(edges, order[:k], 3)


def test_sparse_random_graph_matches_scipy():
    rng = np.random.default_rng(123)
    for _ in range(25):
        edges = [np.flatnonzero(rng.random(20) < 0.06).tolist() for _ in range(70)]
        order = rng.permutation(70)
        for k, value in zip([0, 1, 10, 50, 100], prefix_matches(edges, order, [0, 1, 10, 50, 100])):
            assert value == cardinality(edges, order[:k], 20)


def test_conditional_randomization_preserves_every_prefix_stratum():
    labels = np.array(["A", "B", "A", "C", "B", "A"])
    order = np.array([5, 1, 2, 3, 4, 0])
    for control_seed in range(30):
        shuffled = conditional_order(order, labels, np.random.default_rng(control_seed))
        assert set(shuffled) == set(order)
        np.testing.assert_array_equal(labels[shuffled], labels[order])


def test_seed_streams_reproduce_but_are_distinct():
    a = stream_seed("fold", 1).permutation(100)
    np.testing.assert_array_equal(a, stream_seed("fold", 1).permutation(100))
    assert not np.array_equal(a, stream_seed("fold", 2).permutation(100))


def test_empirical_p_value_and_zero_expectation():
    s = summary(np.zeros(1000), 1)
    assert s["p_one_sided"] == 1 / 1001 and s["lift"] is None
    assert summary(np.arange(1000), 0)["p_one_sided"] == 1


def test_duplicate_inventory_capacity_not_double_counted():
    edges = [[0], [0], [0], [1, 2], [1, 2]]
    assert prefix_matches(edges, np.arange(5), (3, 4, 5)).tolist() == [1, 2, 3]


def test_empty_population_and_zero_budget_prefix():
    assert prefix_matches([], [], [0, 10, 500]).tolist() == [0, 0, 0]


def test_spatial_matching_uses_inclusive_native_metre_tolerance():
    truth = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=32618)
    points = gpd.GeoDataFrame(geometry=[Point(20, 0), Point(20.001, 0)], crs=32618)
    assert [list(v) for v in adjacency(points, truth, 20)] == [[0], []]
