import geopandas as gpd
from shapely.geometry import Point

from culvert_ai.region import filter_to_region


def test_filter_to_ulster_poughkeepsie_region():
    points = gpd.GeoDataFrame(
        [
            {"name": "Highland area", "geometry": Point(-73.96, 41.72)},
            {"name": "Outside pilot", "geometry": Point(-75.2, 42.8)},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    filtered = filter_to_region(points, clip=False)

    assert filtered["name"].tolist() == ["Highland area"]
