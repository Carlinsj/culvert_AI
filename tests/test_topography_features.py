import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from culvert_ai.features import build_feature_table


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


def test_build_feature_table_applies_negative_observations_at_20m():
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
        positive_radius_m=20,
        negative_radius_m=20,
    ).set_index("candidate_id")

    assert features.loc["denied", "field_denied"] == 1
    assert features.loc["denied", "is_culvert"] == 0
    assert features.loc["denied", "nearest_denied_observation_id"] == "obs-denied"
    assert features.loc["positive", "field_denied"] == 0
