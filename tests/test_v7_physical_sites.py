import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from staging.v7_physical_sites import conservative_consolidation, terrain_crossing_hypotheses


def roads():
    return gpd.GeoDataFrame(
        [{"id": "r", "route_id": "r", "geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs=32618,
    )


def measured():
    rows = []
    for index, elevation in enumerate([4, 3, 1, 3, 4, 4, 4]):
        rows.append(
            {
                "candidate_id": f"p{index}",
                "road_id": "r",
                "road_part_index": 0,
                "road_chainage_m": index * 10.0,
                "route_sample_distance_m": index * 10.0,
                "proposal_kind": "internal_road_probe",
                "elevation_m": elevation,
                "negative_tpi_9x9_m": 1.0,
                "negative_tpi_15x15_m": 2.0,
                "route_elevation_dip_60m": 1.0,
                "upstream_area_m2": 5000.0,
                "geometry": Point(index * 10.0, 0),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=32618)


def test_terrain_sites_are_sparse_physical_minima_and_inventory_independent():
    frame = measured()
    before = terrain_crossing_hypotheses(frame, roads())
    frame["distance_to_inventory"] = 0
    frame["target"] = 1
    after = terrain_crossing_hypotheses(frame.sample(frac=1, random_state=4), roads())
    assert len(before) == 1
    assert before.candidate_id.tolist() == after.candidate_id.tolist()
    assert before.iloc[0].geometry.equals(after.iloc[0].geometry)
    assert before.iloc[0].localization_method.startswith("quadratic")


def candidate(identifier, x, source, *, area=np.nan):
    return {
        "candidate_id": identifier,
        "road_id": "r",
        "road_part_index": 0,
        "road_chainage_m": x,
        "candidate_source": source,
        "physical_evidence": identifier,
        "upstream_area_at_link_m2": area,
        "geometry": Point(x, 0),
    }


def test_consolidation_merges_terrain_support_but_preserves_close_explicit_crossings():
    frame = gpd.GeoDataFrame(
        [
            candidate("d8a", 10, "multiscale_conditioned_d8_crossing", area=20_000),
            candidate("terrain", 16, "road_profile_terrain_crossing"),
            candidate("d8b", 14, "multiscale_conditioned_d8_crossing", area=30_000),
        ],
        geometry="geometry",
        crs=32618,
    )
    result = conservative_consolidation(frame)
    assert len(result) == 2
    assert sorted(result.member_count.tolist()) == [1, 2]
    assert set(result.candidate_id) == {"d8a", "d8b"}


def test_nested_existing_network_representation_can_merge_with_large_d8_crossing():
    frame = gpd.GeoDataFrame(
        [
            candidate("d8", 10, "multiscale_conditioned_d8_crossing", area=120_000),
            candidate("old", 20, "explicit_water_road_intersection"),
        ],
        geometry="geometry",
        crs=32618,
    )
    result = conservative_consolidation(frame)
    assert len(result) == 1
    assert result.iloc[0].candidate_id == "d8"
