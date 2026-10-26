"""Invariants for the label-free compact physical-site policy."""

import json

import geopandas as gpd
from shapely.geometry import Point

from staging.final_physical_sites import (
    compact_cross_generator_sites,
    eligible_physical_union,
)


def candidates():
    return gpd.GeoDataFrame(
        {
            "candidate_id": ["base", "terrain", "terrain-near", "channel-a", "channel-b", "weak"],
            "road_id": ["r1"] * 6,
            "road_part_index": [0] * 6,
            "road_chainage_m": [0.0, 5.0, 8.0, 30.0, 34.0, 60.0],
            "candidate_source": [
                "existing_hybrid", "multiscale_terrain_extremum",
                "multiscale_terrain_extremum", "existing_hybrid",
                "multiscale_terrain_extremum", "existing_hybrid",
            ],
            "proposal_kind": [
                "internal_road_probe", "internal_road_probe", "internal_road_probe",
                "dem_crossing", "dem_crossing", "internal_road_probe",
            ],
            "channel_key": ["", "", "", "d8:a", "d8:b", ""],
            "mapped_water_id": ["", "", "", "", "", ""],
            "existing_score": [.9, .3, .4, .8, .7, .99],
            "global_rank": [1, 20, 19, 5, 6, 2],
            "physical_evidence_count": [0, 2, 3, 1, 1, 0],
        },
        geometry=[Point(0, 0), Point(5, 0), Point(8, 0), Point(30, 0), Point(34, 0), Point(60, 0)],
        crs=32618,
    )


def test_physical_union_excludes_uncorroborated_added_internal_probe():
    result = eligible_physical_union(candidates(), {"base"})
    assert set(result.candidate_id) == {"base", "terrain", "terrain-near", "channel-a", "channel-b"}


def test_cross_generator_grouping_preserves_same_generator_and_channel_distinctions():
    admitted = eligible_physical_union(candidates(), {"base"})
    result = compact_cross_generator_sites(admitted, {"base": 1})
    groups = [set(json.loads(value)) for value in result.member_candidate_ids]
    assert {"base", "terrain"} in groups
    assert {"terrain-near"} in groups
    assert {"channel-a"} in groups
    assert {"channel-b"} in groups
    assert len(result) == 4


def test_representatives_use_existing_coordinates_and_keep_baseline_first():
    admitted = eligible_physical_union(candidates(), {"base"})
    result = compact_cross_generator_sites(admitted, {"base": 1})
    original_coordinates = {(point.x, point.y) for point in admitted.geometry}
    assert {(point.x, point.y) for point in result.geometry} <= original_coordinates
    assert result.iloc[0].candidate_id == "base"
    assert result.iloc[0].global_rank == 1
