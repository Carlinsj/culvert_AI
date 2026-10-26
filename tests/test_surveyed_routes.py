import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from culvert_ai.surveyed_routes import (
    associate_inventory_to_segments,
    canonical_culvert_id,
    clip_measured_paths,
    deduplicate_inventory,
    label_physical_sites,
    marker_in_interval,
)


def test_marker_interval_requires_matching_prefix_and_inclusive_sequence():
    assert marker_in_interval(" 28 86012300", " 28 86012296", " 28 86012306")
    assert not marker_in_interval(" 28 86012307", " 28 86012296", " 28 86012306")
    assert not marker_in_interval("214 86011300", " 28 86012296", " 28 86012306")


def test_measured_geometry_is_clipped_without_inventory_coordinates():
    features = [
        {
            "attributes": {"Route_ID": "route-a"},
            "geometry": {"paths": [[[0, 0, 0.0], [100, 0, 1.0], [200, 0, 2.0]]]},
        }
    ]
    clipped = clip_measured_paths(features, "route-a", 0.5, 1.5)
    assert round(clipped.length, 6) == 100
    assert tuple(clipped.coords[0]) == (50.0, 0.0)
    assert tuple(clipped.coords[-1]) == (150.0, 0.0)


def test_dedupe_uses_stable_id_but_preserves_nearby_distinct_culverts():
    records = gpd.GeoDataFrame(
        [
            {"record_id": "o1", "source": "official", "culvert_id": 150001, "route": "NY 28", "reference_marker": "", "geometry": Point(0, 0)},
            {"record_id": "s1", "source": "summer", "culvert_id": "SC150001", "route": "NY28", "reference_marker": "", "geometry": Point(2, 0)},
            {"record_id": "s2", "source": "summer", "culvert_id": "", "route": "NY28", "reference_marker": "", "geometry": Point(3, 0)},
            {"record_id": "o2", "source": "official", "culvert_id": 150001, "route": "NY28", "reference_marker": "", "geometry": Point(500, 0)},
            {"record_id": "o3", "source": "official", "culvert_id": "", "route": "NY28", "reference_marker": "", "geometry": Point(100, 0)},
            {"record_id": "s3", "source": "summer", "culvert_id": "", "route": "NY28", "reference_marker": "", "geometry": Point(100.9, 0)},
        ],
        geometry="geometry",
        crs=26918,
    )
    result = deduplicate_inventory(records)
    assert canonical_culvert_id("SC150001") == "150001"
    assert len(result) == 4
    assert result.source_record_count.tolist() == [2, 1, 2, 1]


def test_three_way_labels_keep_extra_nearby_candidate_uncertain():
    sites = gpd.GeoDataFrame(
        [
            {"candidate_id": "a", "segment_id": "s", "geometry": Point(0, 0)},
            {"candidate_id": "b", "segment_id": "s", "geometry": Point(4, 0)},
            {"candidate_id": "c", "segment_id": "s", "geometry": Point(100, 0)},
        ],
        geometry="geometry",
        crs=26918,
    )
    truth = gpd.GeoDataFrame(
        [{"actual_id": "actual", "segment_id": "s", "geometry": Point(1, 0)}],
        geometry="geometry",
        crs=26918,
    )
    labeled, matches = label_physical_sites(sites, truth)
    assert labeled.supervision_label.tolist() == ["positive", "uncertain", "negative"]
    assert matches.actual_id.tolist() == ["actual"]


def test_valid_marker_outside_interval_cannot_fall_back_to_coordinate():
    segments = gpd.GeoDataFrame(
        [
            {
                "segment_id": "surveyed",
                "route": "NY28",
                "start_marker": " 28 86012276",
                "end_marker": " 28 86012286",
                "geometry": Point(0, 0).buffer(1).boundary,
            }
        ],
        geometry="geometry",
        crs=26918,
    )
    inventory = gpd.GeoDataFrame(
        [
            {
                "actual_id": "outside",
                "route": "NY28",
                "reference_marker": " 28 86012270",
                "geometry": Point(1, 0),
            }
        ],
        geometry="geometry",
        crs=26918,
    )
    associated, excluded = associate_inventory_to_segments(inventory, segments)
    assert associated.empty
    assert excluded.reason.tolist() == ["reference_marker_outside_documented_extents"]


def test_ambiguous_inventory_association_only_creates_uncertain_candidates():
    sites = gpd.GeoDataFrame(
        [{"candidate_id": "a", "segment_id": "s", "geometry": Point(0, 0)}],
        geometry="geometry",
        crs=26918,
    )
    truth = gpd.GeoDataFrame(
        [
            {
                "actual_id": "actual",
                "segment_id": "s",
                "road_association_ambiguous": True,
                "geometry": Point(1, 0),
            }
        ],
        geometry="geometry",
        crs=26918,
    )
    labeled, matches = label_physical_sites(sites, truth)
    assert labeled.supervision_label.tolist() == ["uncertain"]
    assert matches.empty
