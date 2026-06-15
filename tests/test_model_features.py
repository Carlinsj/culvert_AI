import pandas as pd

from culvert_ai.model import select_feature_columns


def test_select_feature_columns_excludes_labels_and_coordinates():
    table = pd.DataFrame(
        {
            "is_culvert": [1, 0],
            "dist_to_known_culvert_m": [2.0, 80.0],
            "longitude": [-73.1, -73.2],
            "latitude": [41.1, 41.2],
            "road_stream_distance_m": [0.0, 12.0],
            "stream_density_m_per_sqkm": [100.0, 20.0],
            "road_id": [10, 11],
        }
    )

    assert select_feature_columns(table) == [
        "road_stream_distance_m",
        "stream_density_m_per_sqkm",
    ]
