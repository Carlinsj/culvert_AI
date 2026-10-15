import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from culvert_ai import product_service


def _write_dem(path, *, west=499000, north=4602000, size=4000):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=1,
        dtype="float32",
        crs="EPSG:32618",
        transform=from_origin(west, north, size / 40, size / 40),
    ) as dataset:
        dataset.write(np.ones((1, 40, 40), dtype="float32"))


def _roads(count=1, *, crs="EPSG:32618", start=500000):
    frame = gpd.GeoDataFrame(
        {
            "route_id": [f"Never Trained Route {index + 1}" for index in range(count)],
            "name": [f"Road {index + 1}" for index in range(count)],
            "geometry": [
                LineString([(start, 4600000 + index * 100), (start + 500, 4600000 + index * 100)])
                for index in range(count)
            ],
        },
        crs="EPSG:32618",
    )
    return frame.to_crs(crs)


def _payload(frame, filename="roads.geojson"):
    document = frame.__geo_interface__
    document["crs"] = {"type": "name", "properties": {"name": frame.crs.to_string()}}
    return {"filename": filename, "text": json.dumps(document)}


def test_inspection_converts_known_crs_and_reports_coverage(tmp_path):
    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    roads = _roads(crs="EPSG:4326")

    summary = product_service.inspect_roads(roads, "roads.geojson", dem_path=dem)

    assert summary["detected_crs"] == "EPSG:4326"
    assert summary["analysis_crs"] == "EPSG:32618"
    assert summary["crs_transformed"] is True
    assert summary["coverage"] == "available"


def test_geojson_missing_crs_fails_without_guessing(tmp_path):
    document = _roads().__geo_interface__
    with pytest.raises(product_service.ProductInputError, match="could not determine"):
        product_service.decode_roads(
            {"filename": "roads.geojson", "text": json.dumps(document)}, workspace=tmp_path
        )


@pytest.mark.parametrize(
    ("start", "expected"),
    [(510000, "outside_coverage"), (502800, "partial_coverage")],
)
def test_unsupported_and_partial_coverage_fail_closed(tmp_path, start, expected):
    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    with pytest.raises(product_service.ProductInputError) as caught:
        product_service.inspect_roads(_roads(start=start), "roads.gpkg", dem_path=dem)
    assert caught.value.code == expected


@pytest.mark.parametrize("road_count", [1, 2])
def test_successful_prediction_request_supports_one_multi_and_unseen_routes(
    tmp_path, monkeypatch, road_count
):
    dem = tmp_path / "dem.tif"
    streams = tmp_path / "streams.geojson"
    _write_dem(dem)
    gpd.GeoDataFrame(
        {"geometry": [LineString([(500000, 4599900), (500500, 4600100)])]},
        crs="EPSG:32618",
    ).to_file(streams, driver="GeoJSON")
    seen = {}

    def fake_predict(roads, **kwargs):
        seen["count"] = len(roads)
        seen["route_column"] = kwargs["route_id_column"]
        records = []
        for index, row in roads.reset_index(drop=True).iterrows():
            point = row.geometry.interpolate(0.5, normalized=True)
            records.append(
                {
                    "global_rank": index + 1,
                    "route_local_rank": 1,
                    "route_id": row["route_id"],
                    "road_name": row["name"],
                    "latitude": 41.5,
                    "longitude": -74.9,
                    "candidate_id": f"candidate-{index}",
                    "physical_site_id": f"site-{index}",
                    "flow_c_score": 0.8,
                    "route_z_score": 1.1,
                    "priority_percentile": 1.0,
                    "known_culvert_distance_m": 1.0,
                    "ground_truth_label": 1,
                    "geometry": point,
                }
            )
        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32618")

    monkeypatch.setattr(product_service, "model_status", lambda _: {
        "ready": True,
        "display_name": "FLOW-C Frozen v1",
        "bundle_hash": "abc",
        "bundle_hash_short": "abc",
        "analysis_crs": "EPSG:32618",
    })
    monkeypatch.setattr(product_service, "predict_culverts", fake_predict)
    monkeypatch.setattr(
        product_service.gpd,
        "read_parquet",
        lambda _: gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32618"),
    )
    settings = {
        "model": tmp_path / "model.joblib",
        "streams": streams,
        "dem": dem,
        "flow": dem,
        "drainage": dem,
        "hydrology": tmp_path,
    }

    result = product_service.analyze_payload(_payload(_roads(road_count)), settings=settings)

    assert result["site_count"] == road_count
    assert seen == {"count": road_count, "route_column": "route_id"}
    serialized = json.dumps(result)
    assert "Never Trained Route" in serialized
    assert "ground_truth" not in serialized
    assert "known_culvert" not in serialized


def test_missing_and_corrupted_model_disable_analysis(tmp_path):
    missing = product_service.model_status(tmp_path / "missing.joblib")
    assert missing["ready"] is False
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a model")
    assert product_service.model_status(corrupt)["ready"] is False


def test_location_first_request_selects_nearby_roads_and_runs_frozen_boundary(
    tmp_path, monkeypatch
):
    dem = tmp_path / "dem.tif"
    roads_path = tmp_path / "roads.geojson"
    streams_path = tmp_path / "streams.geojson"
    _write_dem(dem)
    _roads(2).to_file(roads_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"geometry": [LineString([(500000, 4599900), (500500, 4600100)])]},
        crs="EPSG:32618",
    ).to_file(streams_path, driver="GeoJSON")
    captured = {}

    def fake_predict(roads, **kwargs):
        captured["road_count"] = len(roads)
        return gpd.GeoDataFrame(
            [{
                "global_rank": 1,
                "route_local_rank": 1,
                "route_id": "Never Trained Route 1",
                "road_name": "Road 1",
                "latitude": 41.55,
                "longitude": -75.0,
                "candidate_id": "candidate-1",
                "physical_site_id": "site-1",
                "flow_c_score": 0.8,
                "route_z_score": 1.2,
                "priority_percentile": 1.0,
                "geometry": Point(500100, 4600000),
            }],
            geometry="geometry",
            crs="EPSG:32618",
        )

    monkeypatch.setattr(product_service, "model_status", lambda _: {
        "ready": True,
        "display_name": "FLOW-C Frozen v1",
        "bundle_hash": "abc",
        "bundle_hash_short": "abc",
        "analysis_crs": "EPSG:32618",
    })
    monkeypatch.setattr(product_service, "predict_culverts", fake_predict)
    monkeypatch.setattr(
        product_service.gpd,
        "read_parquet",
        lambda _: gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32618"),
    )
    center = gpd.GeoSeries([Point(500200, 4600000)], crs="EPSG:32618").to_crs("EPSG:4326").iloc[0]
    result = product_service.analyze_nearby_payload(
        {"latitude": center.y, "longitude": center.x, "radius_m": 1000},
        settings={
            "model": tmp_path / "model.joblib",
            "roads": roads_path,
            "streams": streams_path,
            "dem": dem,
            "flow": dem,
            "drainage": dem,
            "hydrology": tmp_path,
        },
    )

    assert result["analysis"]["source"] == "current_location"
    assert result["analysis"]["radius_m"] == 1000
    assert result["site_count"] == 1
    assert captured["road_count"] == 2
