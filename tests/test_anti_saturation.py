"""Anti-saturation control invariants; no model fitting."""

import geopandas as gpd
from shapely.geometry import Point

from scripts.validate_anti_saturation import deterministic_uniform, prefix_state


def test_exact_budget_uniform_sampling_is_label_free_and_deterministic():
    pool = gpd.GeoDataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(20)],
            "road_id": ["r"] * 20,
            "road_part_index": [0] * 20,
            "road_chainage_m": list(range(0, 200, 10)),
        },
        geometry=[Point(i, 0) for i in range(20)],
        crs=32618,
    )
    first = deterministic_uniform(pool, 5)
    second = deterministic_uniform(pool.sample(frac=1, random_state=3), 5)
    assert len(first) == 5 and first.candidate_id.tolist() == second.candidate_id.tolist()


def test_prefix_matching_reassigns_to_reach_maximum_cardinality():
    # Site 0 can use either actual; site 1 can use only actual 0. Greedy counting
    # would stop at one, while augmenting paths correctly reassign site 0.
    states = prefix_state([[0, 1], [0]], [0, 1], [1, 2])
    assert states[1]["matches"] == 1
    assert states[2]["matches"] == 2
