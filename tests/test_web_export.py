import geopandas as gpd
from shapely.geometry import Point

from culvert_ai.web_export import _limit_for_web


def test_limit_for_web_keeps_unknown_predictions_near_known_field_points():
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

    limited = _limit_for_web(predictions, limit=1)

    assert set(limited["candidate_id"]) == {"top", "near-field", "known"}
