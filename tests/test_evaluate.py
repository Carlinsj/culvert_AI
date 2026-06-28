import geopandas as gpd
from shapely.geometry import Point

from culvert_ai.evaluate import evaluate_success_rate_at_actuals


def test_evaluate_success_rate_counts_predictions_within_15m():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "hit",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 1,
                "geometry": Point(10, 0),
            },
            {
                "candidate_id": "miss",
                "discovery_status": "undiscovered_candidate",
                "discovery_rank": 2,
                "geometry": Point(100, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    actual = gpd.GeoDataFrame(
        [
            {"culvert_id": "A", "geometry": Point(0, 0)},
            {"culvert_id": "B", "geometry": Point(70, 0)},
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    metrics = evaluate_success_rate_at_actuals(predictions, actual, max_distance_m=15)

    assert metrics["hits_within_distance"] == 1
    assert metrics["actual_culverts"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["misses"][0]["actual_id"] == "B"


def test_evaluate_success_rate_excludes_known_matches_by_default():
    predictions = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "known",
                "discovery_status": "known_field_match",
                "source": "field_report_observed_culvert",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "real-prediction",
                "discovery_status": "undiscovered_candidate",
                "source": "route_interval_sample",
                "geometry": Point(12, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    actual = gpd.GeoDataFrame(
        [{"culvert_id": "A", "geometry": Point(0, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )

    metrics = evaluate_success_rate_at_actuals(predictions, actual, max_distance_m=15)

    assert metrics["prediction_candidates"] == 1
    assert metrics["hits_within_distance"] == 1
    assert metrics["matches"][0]["nearest_candidate_id"] == "real-prediction"
