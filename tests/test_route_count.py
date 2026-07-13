import json

import geopandas as gpd
from shapely.geometry import LineString, Point

from culvert_ai.route_count import build_route_count_report, write_route_count_report


def test_route_count_report_clusters_route_predictions_into_sites():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "near-a",
                "road_name": "State Rte 212",
                "matched_route": "NY212",
                "source": "route_interval_sample",
                "discovery_rank": 1,
                "discovery_score": 80.0,
                "culvert_probability": 0.8,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "near-b",
                "road_name": "State Rte 212",
                "matched_route": "212",
                "source": "route_interval_sample",
                "discovery_rank": 2,
                "discovery_score": 78.0,
                "culvert_probability": 0.4,
                "geometry": Point(10, 0),
            },
            {
                "candidate_id": "far",
                "road_name": "State Route 212",
                "matched_route": "NY212",
                "source": "route_interval_sample",
                "discovery_rank": 3,
                "discovery_score": 72.0,
                "culvert_probability": 0.6,
                "geometry": Point(100, 0),
            },
            {
                "candidate_id": "other-route",
                "road_name": "State Rte 32",
                "matched_route": "NY32",
                "source": "route_interval_sample",
                "discovery_rank": 4,
                "discovery_score": 99.0,
                "culvert_probability": 0.9,
                "geometry": Point(200, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    report, clusters = build_route_count_report(
        predictions,
        route="Route 212",
        cluster_radius_m=30,
        thresholds=(70,),
    )

    assert report["filtered_prediction_rows"] == 3
    assert report["candidate_clusters"] == 2
    assert report["recommended"]["expected_count"] == 1.4
    assert report["recommended"]["predicted_count"] == 1
    assert report["threshold_counts"][0]["cluster_count"] == 2
    assert clusters["candidate_id"].tolist() == ["near-a", "far"]


def test_route_count_report_can_filter_to_walked_segment_buffer():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "inside",
                "road_name": "US Hwy 9w",
                "discovery_score": 75.0,
                "culvert_probability": 0.7,
                "geometry": Point(10, 10),
            },
            {
                "candidate_id": "outside",
                "road_name": "US Hwy 9w",
                "discovery_score": 90.0,
                "culvert_probability": 0.9,
                "geometry": Point(10, 50),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    segment = gpd.GeoDataFrame(
        [{"geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs="EPSG:32618",
    )

    report, clusters = build_route_count_report(
        predictions,
        segment=segment,
        buffer_m=20,
        cluster_radius_m=15,
    )

    assert report["filtered_prediction_rows"] == 1
    assert report["candidate_clusters"] == 1
    assert report["segment_length_m"] == 100.0
    assert clusters.iloc[0]["candidate_id"] == "inside"


def test_write_route_count_report_writes_json_csv_and_geojson(tmp_path):
    predictions_path = tmp_path / "predictions.geojson"
    output_path = tmp_path / "route-count.json"
    csv_output = tmp_path / "route-count.csv"
    geojson_output = tmp_path / "route-count.geojson"
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "candidate",
                "road_name": "NY 28",
                "discovery_score": 70.0,
                "culvert_probability": 0.55,
                "geometry": Point(0, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    predictions.to_file(predictions_path, driver="GeoJSON")

    result = write_route_count_report(
        predictions_path=predictions_path,
        output_path=output_path,
        route="NY28",
        csv_output=csv_output,
        geojson_output=geojson_output,
    )

    report = json.loads(output_path.read_text())
    assert result["predicted_count"] == 1
    assert report["candidate_clusters"] == 1
    assert csv_output.exists()
    assert geojson_output.exists()
