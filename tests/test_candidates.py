from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from culvert_ai.candidates import (
    generate_road_route_candidates,
    merge_candidate_layers,
    _route_tokens_from_text,
)
from culvert_ai.cli import _build_road_candidates


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


def test_build_road_candidates_reads_routes_from_road_name_column(tmp_path):
    roads_path = tmp_path / "roads.geojson"
    routes_path = tmp_path / "observations.geojson"
    output_path = tmp_path / "route_candidates.geojson"
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
    observations = gpd.GeoDataFrame(
        [{"road_name": "US Hwy 9w", "geometry": Point(0, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )
    roads.to_file(roads_path, driver="GeoJSON")
    observations.to_file(routes_path, driver="GeoJSON")

    result = _build_road_candidates(
        SimpleNamespace(
            roads=str(roads_path),
            output=str(output_path),
            routes=[],
            routes_from=[str(routes_path)],
            interval_m=40,
            all_numbered_roads=False,
            lateral_offsets_m=[0.0],
        )
    )
    candidates = gpd.read_file(output_path)

    assert result["rows"] == 3
    assert set(candidates["road_name"]) == {"US Hwy 9w"}
    assert set(candidates["matched_route"]) == {"9W"}


def test_merge_candidate_layers_deduplicates_overlapping_route_samples():
    all_numbered = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "route_000001",
                "road_name": "State Rte 32",
                "source": "route_interval_sample",
                "road_stream_distance_m": float("nan"),
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "route_000002",
                "road_name": "State Rte 32",
                "source": "route_interval_sample",
                "road_stream_distance_m": float("nan"),
                "geometry": Point(20, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    observed_route = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "route_000003",
                "road_name": "State Rte 32",
                "source": "route_interval_sample",
                "road_stream_distance_m": float("nan"),
                "geometry": Point(1, 1),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    merged = merge_candidate_layers([all_numbered, observed_route], min_spacing_m=5)

    assert len(merged) == 2
    assert list(merged["candidate_id"]) == ["cand_000001", "cand_000002"]
    assert sorted(point.x for point in merged.geometry) == [0.0, 20.0]


def test_merge_candidate_layers_prefers_road_stream_crossing_over_route_sample():
    route_sample = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "route_000001",
                "road_name": "State Rte 32",
                "source": "route_interval_sample",
                "road_stream_distance_m": float("nan"),
                "geometry": Point(0, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    crossing = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "cand_000001",
                "road_name": "State Rte 32",
                "source": "exact_road_stream_intersection",
                "road_stream_distance_m": 0.0,
                "geometry": Point(1, 1),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    merged = merge_candidate_layers([route_sample, crossing], min_spacing_m=5)

    assert len(merged) == 1
    assert merged.iloc[0]["candidate_id"] == "cand_000001"
    assert merged.iloc[0]["source"] == "exact_road_stream_intersection"


def test_merge_candidate_layers_preserves_distinct_close_crossings():
    crossings = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "crossing-1",
                "road_id": "road-1",
                "stream_id": "stream-1",
                "source": "exact_road_stream_intersection",
                "road_stream_distance_m": 0.0,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "crossing-2",
                "road_id": "road-1",
                "stream_id": "stream-2",
                "source": "exact_road_stream_intersection",
                "road_stream_distance_m": 0.0,
                "geometry": Point(4, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    merged = merge_candidate_layers([crossings], min_spacing_m=20)

    assert len(merged) == 2


def test_merge_candidate_layers_collapses_same_coordinate_artifacts():
    crossings = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "crossing-1",
                "road_id": "road-1",
                "stream_id": "stream-1",
                "source": "exact_road_stream_intersection",
                "road_stream_distance_m": 0.0,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "crossing-2",
                "road_id": "road-1",
                "stream_id": "stream-2",
                "source": "exact_road_stream_intersection",
                "road_stream_distance_m": 0.0,
                "geometry": Point(0.5, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    merged = merge_candidate_layers([crossings], min_spacing_m=20)

    assert len(merged) == 1
