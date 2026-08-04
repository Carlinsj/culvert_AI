import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from culvert_ai.model import (
    _precision_floor_operating_point,
    predict_culvert_probability,
    select_feature_columns,
    select_training_rows,
    train_model,
)


def test_select_feature_columns_excludes_labels_and_coordinates():
    table = pd.DataFrame(
        {
            "is_culvert": [1, 0],
            "dist_to_known_culvert_m": [2.0, 80.0],
            "field_denied": [0, 1],
            "dist_to_denied_culvert_m": [100.0, 3.0],
            "longitude": [-73.1, -73.2],
            "latitude": [41.1, 41.2],
            "road_stream_distance_m": [0.0, 12.0],
            "stream_density_m_per_sqkm": [100.0, 20.0],
            "dem_culvert_terrain_score": [0.8, 0.2],
            "source_route_interval_sample": [1, 0],
            "source_exact_intersection": [0, 1],
            "has_matched_route": [1, 0],
            "route_sample_distance_m": [37.5, 0.0],
            "route_lateral_offset_m": [8.0, 0.0],
            "priority_seed": [0.0, 1.0],
            "training_sample_weight": [18.0, 0.25],
            "field_recall_score": [65.0, 0.0],
            "approved_known_culvert_pattern_score": [0.7, 0.0],
            "approved_known_culvert_count_500m": [1, 0],
            "nearest_known_culvert_distance_decay": [0.6, 0.0],
            "nearest_known_route_match": [1, 0],
            "road_id": [10, 11],
        }
    )

    assert select_feature_columns(table) == [
        "stream_density_m_per_sqkm",
        "dem_culvert_terrain_score",
    ]


def test_select_training_rows_excludes_unreviewed_candidates():
    table = gpd.GeoDataFrame(
        [
            {"candidate_id": "positive-1", "is_culvert": 1, "field_denied": 0},
            {"candidate_id": "positive-2", "is_culvert": 1, "field_denied": 0},
            {"candidate_id": "denied-1", "is_culvert": 0, "field_denied": 1},
            {"candidate_id": "denied-2", "is_culvert": 0, "field_denied": 1},
            {"candidate_id": "unreviewed", "is_culvert": 0, "field_denied": 0},
        ],
        geometry=[Point(index, 0) for index in range(5)],
        crs="EPSG:32618",
    )

    selected, summary = select_training_rows(table)

    assert set(selected["candidate_id"]) == {
        "positive-1",
        "positive-2",
        "denied-1",
        "denied-2",
    }
    assert selected["is_culvert"].value_counts().to_dict() == {1: 2, 0: 2}
    assert summary == {
        "strategy": "explicit_confirmed_and_denied_field_labels",
        "source_rows": 5,
        "training_rows": 4,
        "unreviewed_rows_excluded": 1,
        "confirmed_rows": 2,
        "denied_rows": 2,
    }


def test_precision_floor_operating_point_finds_60_percent_cutoff():
    operating_point = _precision_floor_operating_point(
        y_true=[1, 0, 1, 0, 1, 0],
        y_probability=[0.95, 0.9, 0.8, 0.7, 0.65, 0.1],
        target_precision=0.60,
    )

    assert operating_point["meets_precision_floor"] is True
    assert operating_point["precision"] >= 0.60
    assert operating_point["threshold"] == 0.65
    assert operating_point["predicted_positive_count"] == 5


def test_train_model_supports_soft_voting_ensemble(tmp_path):
    rows = []
    for index in range(24):
        is_culvert = int(index % 3 == 0)
        rows.append(
            {
                "candidate_id": f"cand-{index}",
                "is_culvert": is_culvert,
                "road_stream_distance_m": 2.0 if is_culvert else 60.0 + index,
                "stream_density_m_per_sqkm": 90.0 if is_culvert else 10.0 + index,
                "dem_culvert_terrain_score": 0.85 if is_culvert else 0.15,
                "training_sample_weight": 4.0 if is_culvert else 0.5,
                "geometry": Point(float(index), float(index % 4)),
            }
        )
    features = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32618")
    model_path = tmp_path / "ensemble.joblib"
    metrics_path = tmp_path / "metrics.json"
    importance_path = tmp_path / "importance.csv"

    metrics = train_model(
        features,
        model_output=model_path,
        metrics_output=metrics_path,
        importance_output=importance_path,
        model_family="soft_voting_ensemble",
        spatial_cv=False,
        test_size=0.25,
    )
    predictions = predict_culvert_probability(features, model_path)

    assert metrics["selected_model"] == "soft_voting_ensemble"
    assert model_path.exists()
    assert metrics_path.exists()
    assert importance_path.exists()
    assert "culvert_probability" in predictions.columns
    assert predictions["culvert_probability"].between(0, 1).all()
    assert metrics["feature_importance"][0]["method"] == "ensemble_component_feature_importance"
