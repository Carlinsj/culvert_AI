import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from culvert_ai.features import (
    add_route_profile_dem_features,
    add_training_sample_weights,
    build_feature_table,
)


def test_build_feature_table_adds_dem_hydrology_proxies(tmp_path):
    dem_path = tmp_path / "dem.tif"
    rows = cols = 45
    y, x = np.indices((rows, cols))
    center_channel = np.abs(x - cols // 2)
    data = (100 + y * 0.2 + center_channel * 0.8).astype("float32")
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:32618",
        transform=from_origin(0, rows, 1, 1),
    ) as dst:
        dst.write(data, 1)

    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "cand-1",
                "road_stream_distance_m": 0.0,
                "crossing_angle_degrees": 88.0,
                "source": "exact_road_stream_intersection",
                "geometry": Point(cols // 2 + 0.5, rows // 2 + 0.5),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(candidates, dem_path=dem_path)

    assert "elevation_m" in features.columns
    assert "topographic_wetness_proxy_9x9" in features.columns
    assert "terrain_break_score_proxy_31x31" in features.columns
    assert "dem_culvert_terrain_score" in features.columns
    assert "crossing_geometry_signal" in features.columns
    assert features.iloc[0]["source_exact_intersection"] == 1


def test_geographic_dem_slope_uses_metre_pixel_resolution(tmp_path):
    dem_path = tmp_path / "geographic-dem.tif"
    rows = cols = 45
    _y, x = np.indices((rows, cols))
    data = (100 + x).astype("float32")
    pixel_degrees = 1 / 3600
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4269",
        transform=from_origin(-74.01, 42.01, pixel_degrees, pixel_degrees),
    ) as dst:
        dst.write(data, 1)

    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "geographic-slope",
                "geometry": Point(
                    -74.01 + (cols // 2 + 0.5) * pixel_degrees,
                    42.01 - (rows // 2 + 0.5) * pixel_degrees,
                ),
            }
        ],
        geometry="geometry",
        crs="EPSG:4269",
    )

    features = build_feature_table(candidates, dem_path=dem_path)

    assert 0 < features.iloc[0]["slope_degrees"] < 10


def test_route_profile_dem_features_find_real_low_point_without_regular_spacing():
    elevations = [103.0, 102.5, 102.0, 100.0, 100.0, 100.0, 101.5, 102.0, 102.5]
    points = gpd.GeoDataFrame(
        [
            {
                "candidate_id": f"route-{index}",
                "source": "route_interval_sample",
                "road_id": "road-55",
                "matched_route": "55",
                "route_part_index": 0,
                "route_lateral_offset_m": 0,
                "route_sample_distance_m": float(index * 10),
                "elevation_m": elevation,
                "geometry": Point(index * 10, 0),
            }
            for index, elevation in enumerate(elevations)
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = add_route_profile_dem_features(points).set_index("candidate_id")

    assert features.loc["route-4", "route_elevation_dip_60m"] >= 1.5
    assert features.loc["route-4", "route_low_point_score"] >= 0.7
    assert features.loc["route-0", "route_low_point_score"] == 0


def test_manual_and_moved_feedback_is_inferred_onto_nearest_route():
    route_55 = [
        {
            "candidate_id": f"route-55-{position}",
            "source": "route_interval_sample",
            "road_id": "road-55",
            "road_name": "State Rte 55",
            "matched_route": "55",
            "route_part_index": 0,
            "route_sample_distance_m": float(position),
            "geometry": Point(position, 0),
        }
        for position in range(0, 101, 20)
    ]
    candidates = gpd.GeoDataFrame(
        [
            *route_55,
            {
                "candidate_id": "route-32",
                "source": "route_interval_sample",
                "road_id": "road-32",
                "road_name": "State Rte 32",
                "matched_route": "32",
                "route_part_index": 0,
                "route_sample_distance_m": 0.0,
                "geometry": Point(0, 500),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [
            {
                "observation_id": f"positive-{index}",
                "feedback_type": feedback_type,
                "source_file": "field_observations.geojson",
                "geometry": Point(position, 3),
            }
            for index, (position, feedback_type) in enumerate(
                [(0, "cbu"), (20, "abu"), (40, "mvd"), (60, "cbu")]
            )
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negative = gpd.GeoDataFrame(
        [
            {
                "observation_id": "negative-1",
                "feedback_type": "inc",
                "geometry": Point(80, 3),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        negative_culverts=negative,
        positive_radius_m=5,
        negative_radius_m=5,
    ).set_index("candidate_id")

    assert features.loc["route-55-100", "field_route_feedback_reviews"] == 5.25
    assert features.loc["route-55-100", "field_route_feedback_score"] > 0.60
    assert features.loc["route-32", "field_route_feedback_reviews"] == 0


def test_build_feature_table_applies_negative_observations_at_10m():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "denied",
                "road_stream_distance_m": 0.0,
                "source": "exact_road_stream_intersection",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "positive",
                "road_stream_distance_m": 0.0,
                "source": "exact_road_stream_intersection",
                "geometry": Point(100, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [{"culvert_id": "known-1", "geometry": Point(0, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negative = gpd.GeoDataFrame(
        [{"observation_id": "obs-denied", "notes": "not there", "geometry": Point(10, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        negative_culverts=negative,
        positive_radius_m=10,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["denied", "field_denied"] == 1
    assert features.loc["denied", "is_culvert"] == 0
    assert features.loc["denied", "nearest_denied_observation_id"] == "obs-denied"
    assert features.loc["positive", "field_denied"] == 0


def test_known_culvert_route_match_accepts_bare_and_prefixed_tokens():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "route-sample",
                "source": "route_interval_sample",
                "matched_route": "209",
                "road_name": "US Hwy 209",
                "geometry": Point(0, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [{"route": "NY209", "geometry": Point(100, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        positive_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["route-sample", "nearest_known_route_match"] == 1


def test_build_feature_table_applies_missed_prediction_by_candidate_id():
    roads = gpd.GeoDataFrame(
        [{"geometry": LineString([(0, -1), (0, 1)])}],
        geometry="geometry",
        crs="EPSG:32618",
    )
    streams = gpd.GeoDataFrame(
        [{"stream_order": 1, "geometry": LineString([(-1, 0), (1, 0)])}],
        geometry="geometry",
        crs="EPSG:32618",
    )
    candidates = gpd.GeoDataFrame(
        [
            {"candidate_id": "bad-prediction", "geometry": Point(0, 0)},
            {"candidate_id": "other", "geometry": Point(200, 0)},
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [{"geometry": Point(100, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negatives = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-miss",
                "candidate_id": "bad-prediction",
                "label": "missed_prediction",
                "miss_distance_m": 100.0,
                "notes": "confirmed culvert was 100.0 m from this prediction",
                "geometry": Point(100, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        positive_radius_m=10,
        negative_culverts=negatives,
        negative_radius_m=10,
        roads=roads,
        streams=streams,
    ).set_index("candidate_id")

    assert features.loc["bad-prediction", "field_denied"] == 1
    assert features.loc["bad-prediction", "is_culvert"] == 0
    assert features.loc["bad-prediction", "dist_to_denied_culvert_m"] == 100.0
    assert features.loc["bad-prediction", "nearest_denied_observation_id"] == "obs-miss"
    assert features.loc["other", "field_denied"] == 0


def test_build_feature_table_handles_missing_miss_distance_for_exact_negative():
    candidates = gpd.GeoDataFrame(
        [
            {"candidate_id": "bad-prediction", "geometry": Point(0, 0)},
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negatives = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-denied",
                "candidate_id": "bad-prediction",
                "label": "no_culvert",
                "miss_distance_m": "<NA>",
                "notes": "no structure found",
                "geometry": Point(0, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        negative_culverts=negatives,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["bad-prediction", "field_denied"] == 1
    assert features.loc["bad-prediction", "is_culvert"] == 0
    assert features.loc["bad-prediction", "dist_to_denied_culvert_m"] == 0.0


def test_missed_prediction_negative_does_not_deny_true_culvert_geometry():
    candidates = gpd.GeoDataFrame(
        [
            {"candidate_id": "bad-prediction", "geometry": Point(0, 0)},
            {"candidate_id": "true-culvert", "geometry": Point(100, 0)},
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [{"culvert_id": "FC-1", "geometry": Point(100, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negatives = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-miss",
                "candidate_id": "bad-prediction",
                "label": "missed_prediction",
                "miss_distance_m": 100.0,
                "notes": "confirmed culvert was 100.0 m from this prediction",
                "geometry": Point(100, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        positive_radius_m=10,
        negative_culverts=negatives,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["bad-prediction", "field_denied"] == 1
    assert features.loc["bad-prediction", "is_culvert"] == 0
    assert features.loc["true-culvert", "field_denied"] == 0
    assert features.loc["true-culvert", "is_culvert"] == 1


def test_direct_mvd_origin_candidate_is_used_when_old_candidate_id_is_stale():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "field_000001",
                "source": "field_observed_non_culvert",
                "field_observation_label": "missed_prediction",
                "geometry": Point(0, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negatives = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-mvd",
                "candidate_id": "stale-candidate-id",
                "label": "missed_prediction",
                "feedback_type": "mvd_origin",
                "geometry": Point(0, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        negative_culverts=negatives,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["field_000001", "field_denied"] == 1
    assert features.loc["field_000001", "nearest_denied_feedback_type"] == "mvd_origin"


def test_direct_feedback_row_prevents_duplicate_negative_weighting():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "generated-nearby",
                "source": "route_interval_sample",
                "geometry": Point(2, 0),
            },
            {
                "candidate_id": "field_000001",
                "source": "field_observed_non_culvert",
                "field_feedback_observation_id": "obs-inc",
                "geometry": Point(0, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negatives = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-inc",
                "candidate_id": "old-candidate-id",
                "label": "no_culvert",
                "feedback_type": "inc",
                "geometry": Point(0, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        negative_culverts=negatives,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["field_000001", "field_denied"] == 1
    assert features.loc["generated-nearby", "field_denied"] == 0


def test_direct_feedback_row_still_denies_original_candidate_id():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "reviewed-candidate",
                "source": "exact_road_stream_intersection",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "field_000001",
                "source": "field_observed_non_culvert",
                "field_feedback_observation_id": "obs-inc",
                "geometry": Point(0, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    negatives = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-inc",
                "candidate_id": "reviewed-candidate",
                "label": "no_culvert",
                "feedback_type": "inc",
                "geometry": Point(0, 0),
            }
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        negative_culverts=negatives,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["reviewed-candidate", "field_denied"] == 1
    assert features.loc["field_000001", "field_denied"] == 1


def test_known_culvert_labels_one_training_point_per_known_culvert():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "near-route",
                "source": "route_interval_sample",
                "geometry": Point(4, 0),
            },
            {
                "candidate_id": "near-crossing",
                "source": "exact_road_stream_intersection",
                "geometry": Point(2, 0),
            },
            {
                "candidate_id": "field-point",
                "source": "field_report_observed_culvert",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "far",
                "source": "route_interval_sample",
                "geometry": Point(50, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [{"culvert_id": "known-1", "geometry": Point(0, 0)}],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        positive_radius_m=10,
    ).set_index("candidate_id")

    assert int(features["is_culvert"].sum()) == 1
    assert features.loc["field-point", "is_culvert"] == 0
    assert features.loc["near-crossing", "is_culvert"] == 1
    assert features.loc["near-route", "is_culvert"] == 0


def test_known_culvert_pattern_features_use_approved_abu_and_doc_points():
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "known-abu",
                "source": "field_report_observed_culvert",
                "matched_route": "NY28",
                "road_name": "State Rte 28",
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "near-abu-route",
                "source": "route_interval_sample",
                "matched_route": "NY28",
                "road_name": "State Rte 28",
                "geometry": Point(450, 0),
            },
            {
                "candidate_id": "near-doc-route",
                "source": "route_interval_sample",
                "matched_route": "NY212",
                "road_name": "State Rte 212",
                "geometry": Point(2400, 0),
            },
            {
                "candidate_id": "far-route",
                "source": "route_interval_sample",
                "matched_route": "NY28",
                "road_name": "State Rte 28",
                "geometry": Point(5000, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )
    known = gpd.GeoDataFrame(
        [
            {
                "culvert_id": "ABU-1",
                "route": "NY28",
                "source_file": "field_observations.geojson",
                "label_confidence": 1.0,
                "geometry": Point(0, 0),
            },
            {
                "culvert_id": "DOC-1",
                "route": "NY212",
                "source_file": "team_report.pdf",
                "label_confidence": 0.85,
                "geometry": Point(2000, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        positive_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["known-abu", "is_culvert"] == 1
    assert features.loc["known-abu", "approved_known_culvert_pattern_score"] == 0
    assert features.loc["near-abu-route", "nearest_known_route_match"] == 1
    assert features.loc["near-abu-route", "nearest_known_source_abu"] == 1
    assert features.loc["near-doc-route", "nearest_known_source_doc"] == 1
    assert features.loc["near-abu-route", "approved_known_culvert_pattern_score"] > features.loc[
        "far-route", "approved_known_culvert_pattern_score"
    ]


def test_training_sample_weights_prioritize_abu_inputs():
    features = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "weak-unlabeled",
                "is_culvert": 0,
                "field_denied": 0,
                "geometry": Point(0, 0),
            },
            {
                "candidate_id": "report-positive",
                "is_culvert": 1,
                "nearest_field_report_source_file": "team_report.pdf",
                "geometry": Point(1, 0),
            },
            {
                "candidate_id": "abu-positive",
                "is_culvert": 1,
                "nearest_field_report_source_file": "field_observations.geojson",
                "nearest_field_feedback_type": "abu",
                "geometry": Point(2, 0),
            },
            {
                "candidate_id": "mvd-positive",
                "is_culvert": 1,
                "nearest_field_report_source_file": "field_observations.geojson",
                "nearest_field_feedback_type": "mvd",
                "geometry": Point(2.5, 0),
            },
            {
                "candidate_id": "abu-denied",
                "is_culvert": 0,
                "field_denied": 1,
                "nearest_denied_feedback_type": "inc",
                "nearest_denied_notes": "no crossing found",
                "geometry": Point(3, 0),
            },
            {
                "candidate_id": "abu-missed",
                "is_culvert": 0,
                "field_denied": 1,
                "nearest_denied_feedback_type": "mvd_origin",
                "nearest_denied_notes": "confirmed culvert was 100.0 m from this prediction",
                "geometry": Point(4, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    weighted = add_training_sample_weights(features).set_index("candidate_id")

    assert weighted.loc["weak-unlabeled", "training_sample_weight"] == 0.05
    assert weighted.loc["report-positive", "training_sample_weight"] == 6.0
    assert weighted.loc["abu-positive", "training_sample_weight"] == 24.0
    assert weighted.loc["mvd-positive", "training_sample_weight"] == 32.0
    assert weighted.loc["abu-denied", "training_sample_weight"] == 24.0
    assert weighted.loc["abu-missed", "training_sample_weight"] == 32.0
