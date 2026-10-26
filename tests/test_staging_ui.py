"""Transport/navigation regressions; no training or model inference."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import Point, box


@pytest.mark.parametrize("failure", [BrokenPipeError, ConnectionResetError])
def test_browser_disconnect_does_not_become_second_server_error(monkeypatch, failure):
    from scripts.serve_feature_staging import Handler, BaseHandler

    called = []

    def disconnected(*args):
        called.append(args)
        raise failure("Browser left the page")

    monkeypatch.setattr(BaseHandler, "respond", disconnected)
    handler = object.__new__(Handler)
    handler.respond(200, {"read_only": True})
    assert handler.close_connection is True
    assert len(called) == 1 and called[0][1] == 200


def test_non_disconnect_errors_are_not_silently_swallowed(monkeypatch):
    from scripts.serve_feature_staging import Handler, BaseHandler

    def failed(*args):
        raise ValueError("Invalid JSON payload")

    monkeypatch.setattr(BaseHandler, "respond", failed)
    with pytest.raises(ValueError):
        object.__new__(Handler).respond(200, {})


def test_restored_road_area_coordinates_match_existing_navigation_and_coverage():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "web/data/supported-areas.json").read_text())["areas"]
    raster_path = root / "data/processed/hydrology/usgs_3dep_10m/dem_projected.tif"
    if not raster_path.exists():
        pytest.skip("Local GIS assets are not packaged in this checkout")
    with rasterio.open(raster_path) as dem:
        bounds = box(*dem.bounds)
    for record in original:
        point = (
            gpd.GeoSeries([Point(record["longitude"], record["latitude"])], crs=4326)
            .to_crs(32618)
            .iloc[0]
        )
        assert bounds.covers(point.buffer(2500)), record["name"]
