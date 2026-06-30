import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from culvert_ai.features import add_training_sample_weights, build_feature_table


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
    assert "crossing_geometry_signal" in features.columns
    assert features.iloc[0]["source_exact_intersection"] == 1


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
    assert features.loc["field-point", "is_culvert"] == 1
    assert features.loc["near-crossing", "is_culvert"] == 0
    assert features.loc["near-route", "is_culvert"] == 0


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
                "geometry": Point(2, 0),
            },
            {
                "candidate_id": "abu-denied",
                "is_culvert": 0,
                "field_denied": 1,
                "nearest_denied_notes": "no crossing found",
                "geometry": Point(3, 0),
            },
            {
                "candidate_id": "abu-missed",
                "is_culvert": 0,
                "field_denied": 1,
                "nearest_denied_notes": "confirmed culvert was 100.0 m from this prediction",
                "geometry": Point(4, 0),
            },
        ],
        geometry="geometry",
        crs="EPSG:32618",
    )

    weighted = add_training_sample_weights(features).set_index("candidate_id")

    assert weighted.loc["weak-unlabeled", "training_sample_weight"] == 0.25
    assert weighted.loc["report-positive", "training_sample_weight"] == 6.0
    assert weighted.loc["abu-positive", "training_sample_weight"] == 24.0
    assert weighted.loc["abu-denied", "training_sample_weight"] == 12.0
    assert weighted.loc["abu-missed", "training_sample_weight"] == 16.0
