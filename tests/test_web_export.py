import gzip
import json

import geopandas as gpd
from shapely.geometry import Point

from culvert_ai.web_export import (
    _decluster_for_web,
    _limit_for_web,
    _prediction_export_pool,
    _select_geospatial_route_peaks,
    export_web_data,
)


def test_limit_for_web_exports_discovery_candidates_only():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "top",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 1,
                "dist_to_known_culvert_m": 500.0,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "near-field",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 100,
                "dist_to_known_culvert_m": 12.0,
                "source": "route_interval_sample",
                "geometry": Point(1, 0),
            },
            {
                "candidate_id": "known",
                "discovery_status": "known_field_match",
                "discovery_rank": 200,
                "dist_to_known_culvert_m": 0.0,
                "geometry": Point(2, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    limited = _limit_for_web(predictions, limit=3)

    assert set(limited["candidate_id"]) == {"top"}


def test_limit_for_web_does_not_reserve_weak_field_recall_candidates():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "top-1",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 1,
                "discovery_score": 90.0,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "top-2",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 2,
                "discovery_score": 89.0,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "geometry": Point(0.01, 0),
            },
            {
                "candidate_id": "top-3",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 3,
                "discovery_score": 88.0,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "geometry": Point(0.02, 0),
            },
            {
                "candidate_id": "field-corridor",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 99,
                "discovery_score": 55.0,
                "field_recall_score": 60.0,
                "dist_to_known_culvert_m": 500.0,
                "source": "route_interval_sample",
                "matched_route": "9W",
                "geometry": Point(0.03, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    limited = _limit_for_web(predictions, limit=3)

    assert "field-corridor" not in set(limited["candidate_id"])
    assert len(limited) == 3


def test_limit_for_web_only_removes_true_near_duplicates():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "a1",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 1,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "road_id": "road-a",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "a2",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 2,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "road_id": "road-a",
                "geometry": Point(20, 0),
            },
            {
                "candidate_id": "a3",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 3,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "road_id": "road-a",
                "geometry": Point(40, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    limited = _limit_for_web(predictions, limit=3)

    assert limited["candidate_id"].tolist() == ["a1", "a2", "a3"]


def test_route_samples_are_selected_as_geospatial_local_peaks():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "sample-1",
                "source": "route_interval_sample",
                "matched_route": "28",
                "discovery_rank": 3,
                "geospatial_evidence_score": 48.0,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "local-peak",
                "source": "route_interval_sample",
                "matched_route": "28",
                "discovery_rank": 1,
                "geospatial_evidence_score": 67.0,
                "geometry": Point(25, 0),
            },
            {
                "candidate_id": "sample-3",
                "source": "route_interval_sample",
                "matched_route": "28",
                "discovery_rank": 2,
                "geospatial_evidence_score": 55.0,
                "geometry": Point(50, 0),
            },
            {
                "candidate_id": "separate-peak",
                "source": "route_interval_sample",
                "matched_route": "28",
                "discovery_rank": 4,
                "geospatial_evidence_score": 52.0,
                "geometry": Point(200, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    selected = _select_geospatial_route_peaks(predictions)

    assert selected["candidate_id"].tolist() == ["local-peak", "separate-peak"]


def test_limit_for_web_filters_weak_scores_when_scores_exist():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "strong",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 1,
                "discovery_score": 50.0,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "road_id": "road-a",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "weak",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 2,
                "discovery_score": 25.0,
                "field_recall_score": 0.0,
                "dist_to_known_culvert_m": 500.0,
                "road_id": "road-b",
                "geometry": Point(100, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    limited = _limit_for_web(predictions, limit=10)

    assert limited["candidate_id"].tolist() == ["strong"]


def test_prediction_export_pool_removes_known_and_denied_rows():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "discovery",
                "discovery_status": "undiscovered_candidate",
                "is_known_field_match": 0,
                "dist_to_known_culvert_m": 100.0,
                "source": "route_interval_sample",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "near-known",
                "discovery_status": "undiscovered_candidate",
                "is_known_field_match": 0,
                "dist_to_known_culvert_m": 12.0,
                "source": "route_interval_sample",
                "geometry": Point(0.5, 0),
            },
            {
                "candidate_id": "known",
                "discovery_status": "known_field_match",
                "is_known_field_match": 1,
                "dist_to_known_culvert_m": 0.0,
                "source": "route_interval_sample",
                "geometry": Point(1, 0),
            },
            {
                "candidate_id": "denied",
                "discovery_status": "field_denied",
                "is_known_field_match": 0,
                "dist_to_known_culvert_m": 100.0,
                "source": "route_interval_sample",
                "geometry": Point(2, 0),
            },
            {
                "candidate_id": "field-label",
                "discovery_status": "undiscovered_candidate",
                "is_known_field_match": 0,
                "dist_to_known_culvert_m": 100.0,
                "source": "field_report_observed_culvert",
                "geometry": Point(3, 0),
            },
            {
                "candidate_id": "field-negative",
                "discovery_status": "undiscovered_candidate",
                "is_known_field_match": 0,
                "dist_to_known_culvert_m": 100.0,
                "source": "field_observed_non_culvert",
                "geometry": Point(4, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    filtered = _prediction_export_pool(predictions)

    assert filtered["candidate_id"].tolist() == ["discovery"]


def test_export_web_data_writes_full_route_count_source(tmp_path):
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "route-a",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 1,
                "discovery_score": 80.0,
                "culvert_probability": 0.8,
                "dist_to_known_culvert_m": 100.0,
                "matched_route": "212",
                "road_name": "State Rte 212",
                "geometry": Point(-74.0, 42.0),
            },
            {
                "candidate_id": "route-b",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 2,
                "discovery_score": 70.0,
                "culvert_probability": 0.7,
                "dist_to_known_culvert_m": 100.0,
                "matched_route": "212",
                "road_name": "State Rte 212",
                "geometry": Point(-74.01, 42.01),
            },
            {
                "candidate_id": "route-c",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 3,
                "discovery_score": 60.0,
                "culvert_probability": 0.6,
                "dist_to_known_culvert_m": 100.0,
                "matched_route": "32",
                "road_name": "State Rte 32",
                "geometry": Point(-74.02, 42.02),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    predictions_path = tmp_path / "predictions.geojson"
    output_dir = tmp_path / "web"
    predictions.to_file(predictions_path, driver="GeoJSON")

    result = export_web_data(predictions_path, output_dir, limit=1)

    findings = json.loads((output_dir / "findings.geojson").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with gzip.open(result["route_count_source"], "rt", encoding="utf-8") as handle:
        route_count_source = json.load(handle)

    assert len(findings["features"]) == 1
    assert len(route_count_source["features"]) == 3
    assert summary["route_count_rows"] == 3
    assert summary["route_count_source"] == "route_count_source.geojson.gz"
    assert route_count_source["features"][0]["properties"]["matched_route"] == "212"


def test_decluster_for_web_applies_spacing_and_road_cap():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "a1",
                "road_id": "road-a",
                "discovery_rank": 1,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "a2-too-close",
                "road_id": "road-a",
                "discovery_rank": 2,
                "geometry": Point(0.0001, 0),
            },
            {
                "candidate_id": "a3-road-cap",
                "road_id": "road-a",
                "discovery_rank": 3,
                "geometry": Point(0.01, 0),
            },
            {
                "candidate_id": "b1",
                "road_id": "road-b",
                "discovery_rank": 4,
                "geometry": Point(0.02, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    declustered = _decluster_for_web(
        predictions,
        limit=10,
        min_spacing_m=50,
        max_per_road=1,
    )

    assert declustered["candidate_id"].tolist() == ["a1", "b1"]


def test_decluster_for_web_caps_visible_route_before_segment_id():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "route-a1",
                "matched_route": "9W",
                "road_name": "US Hwy 9w",
                "road_id": "segment-1",
                "discovery_rank": 1,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "route-a2",
                "matched_route": "9W",
                "road_name": "US Hwy 9w",
                "road_id": "segment-2",
                "discovery_rank": 2,
                "geometry": Point(0.01, 0),
            },
            {
                "candidate_id": "route-b1",
                "matched_route": "32",
                "road_name": "State Rte 32",
                "road_id": "segment-3",
                "discovery_rank": 3,
                "geometry": Point(0.02, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    declustered = _decluster_for_web(
        predictions,
        limit=10,
        min_spacing_m=0,
        max_per_road=1,
    )

    assert declustered["candidate_id"].tolist() == ["route-a1", "route-b1"]
