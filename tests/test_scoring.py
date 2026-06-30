import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from culvert_ai.scoring import build_discovery_ranking, score_unlabeled_candidates


def test_score_unlabeled_candidates_ranks_by_evidence():
    features = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "high",
                "road_stream_distance_m": 0.0,
                "is_exact_road_stream_intersection": 1,
                "stream_order": 3,
                "stream_density_250m_m_per_sqkm": 100,
                "valley_depth_9x9_m": 2.0,
                "crossing_angle_degrees": 85,
                "slope_degrees": 4,
                "road_density_250m_m_per_sqkm": 50,
                "latitude": 41.72,
                "longitude": -73.96,
                "geometry": Point(-73.96, 41.72),
            },
            {
                "candidate_id": "low",
                "road_stream_distance_m": 80.0,
                "is_exact_road_stream_intersection": 0,
                "stream_order": 1,
                "stream_density_250m_m_per_sqkm": 10,
                "valley_depth_9x9_m": 0.0,
                "crossing_angle_degrees": 20,
                "slope_degrees": 0.2,
                "road_density_250m_m_per_sqkm": 5,
                "latitude": 41.73,
                "longitude": -73.97,
                "geometry": Point(-73.97, 41.73),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    scored = score_unlabeled_candidates(features)

    assert scored.iloc[0]["candidate_id"] == "high"
    assert scored.iloc[0]["culvert_likelihood_score"] > scored.iloc[1]["culvert_likelihood_score"]
    assert "earth.google.com" in scored.iloc[0]["google_earth_url"]
    assert scored.iloc[0]["evidence_summary"]


def test_score_unlabeled_candidates_promotes_dem_route_drainage_signal():
    features = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "road-dip",
                "source": "route_interval_sample",
                "road_stream_distance_m": np.nan,
                "valley_depth_9x9_m": 4.0,
                "topographic_position_9x9_m": -2.0,
                "topographic_wetness_proxy_9x9": 1.2,
                "topographic_wetness_proxy_31x31": 1.8,
                "low_slope_valley_score_31x31": 0.9,
                "negative_tpi_31x31_m": 3.5,
                "terrain_break_score_proxy_9x9": 5.0,
                "stream_density_250m_m_per_sqkm": 90,
                "road_density_250m_m_per_sqkm": 70,
                "latitude": 42.09,
                "longitude": -73.94,
                "geometry": Point(-73.94, 42.09),
            },
            {
                "candidate_id": "flat-road",
                "source": "route_interval_sample",
                "road_stream_distance_m": np.nan,
                "valley_depth_9x9_m": 0.0,
                "topographic_position_9x9_m": 1.0,
                "topographic_wetness_proxy_9x9": 0.0,
                "topographic_wetness_proxy_31x31": 0.0,
                "low_slope_valley_score_31x31": 0.0,
                "negative_tpi_31x31_m": 0.0,
                "terrain_break_score_proxy_9x9": 0.2,
                "stream_density_250m_m_per_sqkm": 5,
                "road_density_250m_m_per_sqkm": 5,
                "latitude": 42.10,
                "longitude": -73.95,
                "geometry": Point(-73.95, 42.10),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    scored = score_unlabeled_candidates(features)

    assert scored.iloc[0]["candidate_id"] == "road-dip"
    assert scored.iloc[0]["dem_route_drainage_score"] > scored.iloc[1]["dem_route_drainage_score"]
    assert "DEM road low point" in scored.iloc[0]["evidence_summary"]


def test_score_unlabeled_candidates_promotes_field_confirmed_corridor():
    features = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "near-abu-corridor",
                "source": "route_interval_sample",
                "road_stream_distance_m": np.nan,
                "dist_to_known_culvert_m": 450.0,
                "nearest_field_report_source_file": "field_observations.geojson",
                "valley_depth_9x9_m": 1.0,
                "stream_density_250m_m_per_sqkm": 20,
                "road_density_250m_m_per_sqkm": 20,
                "latitude": 42.12,
                "longitude": -73.94,
                "geometry": Point(-73.94, 42.12),
            },
            {
                "candidate_id": "far-route-sample",
                "source": "route_interval_sample",
                "road_stream_distance_m": np.nan,
                "dist_to_known_culvert_m": 2500.0,
                "nearest_field_report_source_file": "field_observations.geojson",
                "valley_depth_9x9_m": 1.0,
                "stream_density_250m_m_per_sqkm": 20,
                "road_density_250m_m_per_sqkm": 20,
                "latitude": 42.13,
                "longitude": -73.95,
                "geometry": Point(-73.95, 42.13),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    scored = score_unlabeled_candidates(features)

    by_id = scored.set_index("candidate_id")

    assert by_id.loc["near-abu-corridor", "field_corridor_support_score"] == 0
    assert by_id.loc["far-route-sample", "field_corridor_support_score"] == 0
    assert "field-confirmed culvert corridor" not in by_id.loc["near-abu-corridor", "evidence_summary"]


def test_discovery_ranking_prioritizes_undiscovered_candidates():
    evidence = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "known",
                "culvert_likelihood_score": 95.0,
                "is_culvert": 1,
                "dist_to_known_culvert_m": 8.0,
                "latitude": 41.72,
                "longitude": -73.96,
                "geometry": Point(-73.96, 41.72),
            },
            {
                "candidate_id": "new",
                "culvert_likelihood_score": 72.0,
                "is_culvert": 0,
                "dist_to_known_culvert_m": 800.0,
                "latitude": 41.73,
                "longitude": -73.97,
                "geometry": Point(-73.97, 41.73),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    supervised = gpd.GeoDataFrame(
        [
            {"candidate_id": "known", "culvert_probability": 0.98, "geometry": Point(-73.96, 41.72)},
            {"candidate_id": "new", "culvert_probability": 0.74, "geometry": Point(-73.97, 41.73)},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    ranked = build_discovery_ranking(evidence, supervised_predictions=supervised)

    assert ranked.iloc[0]["candidate_id"] == "new"
    assert ranked.iloc[0]["discovery_status"] == "undiscovered_candidate"
    assert ranked.iloc[1]["discovery_status"] == "known_field_match"


def test_discovery_ranking_does_not_apply_known_corridor_floor_to_route_samples():
    evidence = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "corridor",
                "source": "route_interval_sample",
                "culvert_likelihood_score": 35.0,
                "field_corridor_support_score": 0.50,
                "is_culvert": 0,
                "dist_to_known_culvert_m": 450.0,
                "latitude": 42.12,
                "longitude": -73.94,
                "geometry": Point(-73.94, 42.12),
            },
            {
                "candidate_id": "other",
                "source": "route_interval_sample",
                "culvert_likelihood_score": 48.0,
                "field_corridor_support_score": 0.0,
                "is_culvert": 0,
                "dist_to_known_culvert_m": 3000.0,
                "latitude": 42.13,
                "longitude": -73.95,
                "geometry": Point(-73.95, 42.13),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    ranked = build_discovery_ranking(evidence, known_radius_m=10).set_index("candidate_id")

    assert ranked.loc["corridor", "discovery_score"] == 35.0
    assert ranked.loc["corridor", "discovery_score"] < ranked.loc["other", "discovery_score"]


def test_discovery_ranking_does_not_count_50m_as_known_match():
    evidence = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "too-far",
                "culvert_likelihood_score": 95.0,
                "is_culvert": 0,
                "dist_to_known_culvert_m": 50.0,
                "latitude": 41.72,
                "longitude": -73.96,
                "geometry": Point(-73.96, 41.72),
            },
            {
                "candidate_id": "denied",
                "culvert_likelihood_score": 90.0,
                "field_denied": 1,
                "dist_to_denied_culvert_m": 8.0,
                "latitude": 41.73,
                "longitude": -73.97,
                "geometry": Point(-73.97, 41.73),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    ranked = build_discovery_ranking(evidence, known_radius_m=10)
    by_id = ranked.set_index("candidate_id")

    assert by_id.loc["too-far", "discovery_status"] == "undiscovered_candidate"
    assert by_id.loc["denied", "discovery_status"] == "field_denied"
    assert by_id.loc["denied", "discovery_score"] == 0


def test_discovery_ranking_trusts_field_denied_flag_over_distance():
    evidence = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "true-culvert",
                "culvert_likelihood_score": 95.0,
                "is_culvert": 1,
                "field_denied": 0,
                "dist_to_denied_culvert_m": 0.0,
                "dist_to_known_culvert_m": 0.0,
                "latitude": 41.73,
                "longitude": -73.97,
                "geometry": Point(-73.97, 41.73),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    ranked = build_discovery_ranking(evidence, known_radius_m=10).set_index("candidate_id")

    assert ranked.loc["true-culvert", "discovery_status"] == "known_field_match"
