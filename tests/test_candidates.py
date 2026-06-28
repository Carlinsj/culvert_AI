import geopandas as gpd
import pytest
from shapely.geometry import LineString

from culvert_ai.candidates import generate_road_route_candidates, _route_tokens_from_text


def test_route_tokens_parse_numbered_highway_names():
    assert _route_tokens_from_text("US Hwy 9w") == {"9W"}
    assert _route_tokens_from_text("State Rte 32A") == {"32A"}
    assert _route_tokens_from_text("R-8 NY-9G") == {"9G"}


def test_build_road_route_candidates_can_sample_all_numbered_roads():
    roads = gpd.GeoDataFrame(
        [
            {
                "FULLNAME": "US Hwy 9w",
                "geometry": LineString([(0, 0), (120, 0)]),
            },
            {
                "FULLNAME": "Local Rd",
                "geometry": LineString([(0, 100), (120, 100)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    candidates = generate_road_route_candidates(
        roads,
        routes=[],
        interval_m=40,
        include_numbered_roads=True,
    )

    assert len(candidates) == 3
    assert set(candidates["road_name"]) == {"US Hwy 9w"}
    assert set(candidates["matched_route"]) == {"9W"}


def test_build_road_route_candidates_can_add_lateral_offsets():
    roads = gpd.GeoDataFrame(
        [
            {
                "FULLNAME": "US Hwy 9w",
                "geometry": LineString([(0, 0), (40, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    candidates = generate_road_route_candidates(
        roads,
        routes=[],
        interval_m=20,
        include_numbered_roads=True,
        lateral_offsets_m=(0, -8, 8),
    )

    assert len(candidates) == 6
    assert set(candidates["route_lateral_offset_m"]) == {0.0, -8.0, 8.0}
    assert set(round(point.y, 3) for point in candidates.geometry) == {-8.0, 0.0, 8.0}


def test_all_numbered_roads_mode_skips_county_routes_by_default():
    roads = gpd.GeoDataFrame(
        [
            {
                "FULLNAME": "Co Rd 6",
                "RTTYP": "C",
                "geometry": LineString([(0, 0), (120, 0)]),
            },
            {
                "FULLNAME": "State Rte 32",
                "RTTYP": "S",
                "geometry": LineString([(0, 100), (120, 100)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    candidates = generate_road_route_candidates(
        roads,
        routes=[],
        interval_m=40,
        include_numbered_roads=True,
    )

    assert set(candidates["road_name"]) == {"State Rte 32"}


def test_build_road_route_candidates_still_requires_a_route_without_numbered_mode():
    roads = gpd.GeoDataFrame(
        [{"FULLNAME": "US Hwy 9w", "geometry": LineString([(0, 0), (120, 0)])}],
        geometry="geometry",
        crs="EPSG:32618",
    )

    with pytest.raises(ValueError, match="At least one usable route"):
        generate_road_route_candidates(roads, routes=[], interval_m=40)
