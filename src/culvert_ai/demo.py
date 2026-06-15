from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import LineString, Point

from culvert_ai.candidates import CandidateSettings, generate_candidates
from culvert_ai.features import build_feature_table
from culvert_ai.io import write_vector
from culvert_ai.model import predict_culvert_probability, train_model
from culvert_ai.scoring import score_unlabeled_candidates, write_google_earth_kml


DEMO_CRS = "EPSG:32618"
DEMO_X_OFFSET_M = 78_000
DEMO_Y_OFFSET_M = -88_000


def create_demo_dataset(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    roads = _demo_roads()
    streams = _demo_streams()
    known_culverts = _demo_known_culverts(roads, streams)
    dem_path = raw_dir / "demo_dem.tif"

    write_vector(roads, raw_dir / "roads.gpkg")
    write_vector(streams, raw_dir / "streams.gpkg")
    write_vector(known_culverts, raw_dir / "known_culverts.gpkg")
    _write_demo_dem(dem_path)

    return {
        "roads": raw_dir / "roads.gpkg",
        "streams": raw_dir / "streams.gpkg",
        "known_culverts": raw_dir / "known_culverts.gpkg",
        "dem": dem_path,
    }


def run_demo_pipeline(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = create_demo_dataset(output_dir)

    interim_dir = output_dir / "interim"
    processed_dir = output_dir / "processed"
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    for directory in (interim_dir, processed_dir, model_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    roads = gpd.read_file(paths["roads"])
    streams = gpd.read_file(paths["streams"])
    known = gpd.read_file(paths["known_culverts"])

    candidates = generate_candidates(
        roads,
        streams,
        CandidateSettings(snap_tolerance_m=30, min_spacing_m=20),
    )
    candidate_path = interim_dir / "candidates.gpkg"
    write_vector(candidates, candidate_path)

    features = build_feature_table(
        candidates,
        known_culverts=known,
        roads=roads,
        streams=streams,
        dem_path=paths["dem"],
        positive_radius_m=35,
        density_radius_m=100,
        density_radii_m=(50, 100, 250, 500),
    )
    feature_path = processed_dir / "training_features.gpkg"
    write_vector(features, feature_path)

    unlabeled_predictions = score_unlabeled_candidates(features.drop(columns=["is_culvert"], errors="ignore"))
    unlabeled_prediction_path = processed_dir / "unlabeled_predictions.gpkg"
    unlabeled_prediction_csv = processed_dir / "unlabeled_predictions.csv"
    unlabeled_kml = processed_dir / "google_earth_review.kml"
    write_vector(unlabeled_predictions, unlabeled_prediction_path)
    write_vector(unlabeled_predictions, unlabeled_prediction_csv)
    write_google_earth_kml(unlabeled_predictions, unlabeled_kml)

    model_path = model_dir / "culvert_model.joblib"
    metrics_path = report_dir / "metrics.json"
    importance_path = report_dir / "feature_importance.csv"
    train_model(features, model_path, metrics_path, importance_output=importance_path)

    predictions = predict_culvert_probability(features, model_path)
    prediction_path = processed_dir / "predictions.gpkg"
    prediction_csv = processed_dir / "predictions.csv"
    write_vector(predictions, prediction_path)
    write_vector(predictions, prediction_csv)

    paths.update(
        {
            "candidates": candidate_path,
            "features": feature_path,
            "unlabeled_predictions": unlabeled_prediction_path,
            "unlabeled_predictions_csv": unlabeled_prediction_csv,
            "google_earth_kml": unlabeled_kml,
            "model": model_path,
            "metrics": metrics_path,
            "feature_importance": importance_path,
            "predictions": prediction_path,
            "predictions_csv": prediction_csv,
        }
    )
    return paths


def _demo_roads() -> gpd.GeoDataFrame:
    records = [
        {
            "road_id": "R-001",
            "name": "US Route 9W",
            "municipality": "Lloyd / Highland",
            "geometry": LineString([(501000, 4705200), (501000, 4710900)]),
        },
        {
            "road_id": "R-002",
            "name": "NY 299",
            "municipality": "New Paltz",
            "geometry": LineString([(502350, 4705200), (502350, 4710900)]),
        },
        {
            "road_id": "R-003",
            "name": "Marlboro Road",
            "municipality": "Marlborough",
            "geometry": LineString([(504350, 4705200), (504350, 4710900)]),
        },
        {
            "road_id": "R-004",
            "name": "New Paltz Road",
            "municipality": "Lloyd",
            "geometry": LineString([(500300, 4706600), (505500, 4706600)]),
        },
        {
            "road_id": "R-005",
            "name": "South Street",
            "municipality": "Plattekill",
            "geometry": LineString([(500300, 4708500), (505500, 4708500)]),
        },
        {
            "road_id": "R-006",
            "name": "Esopus Creek Road",
            "municipality": "Esopus",
            "geometry": LineString([(500300, 4710300), (505500, 4710300)]),
        },
    ]
    return _move_to_ulster_pilot(gpd.GeoDataFrame(records, geometry="geometry", crs=DEMO_CRS))


def _demo_streams() -> gpd.GeoDataFrame:
    records = [
        {
            "stream_id": "S-001",
            "name": "Black Creek Tributary",
            "stream_order": 2,
            "geometry": LineString(
                [(500200, 4705600), (501300, 4706900), (502900, 4708600), (505600, 4710600)]
            ),
        },
        {
            "stream_id": "S-002",
            "name": "Twaalfskill Brook Tributary",
            "stream_order": 1,
            "geometry": LineString([(500200, 4709300), (501900, 4708200), (505600, 4706100)]),
        },
        {
            "stream_id": "S-003",
            "name": "Swarte Kill Tributary",
            "stream_order": 1,
            "geometry": LineString([(500400, 4710800), (502250, 4709800), (505400, 4707600)]),
        },
        {
            "stream_id": "S-004",
            "name": "Lloyd Drainage",
            "stream_order": 2,
            "geometry": LineString([(500500, 4706100), (503200, 4707200), (505500, 4709100)]),
        },
    ]
    return _move_to_ulster_pilot(gpd.GeoDataFrame(records, geometry="geometry", crs=DEMO_CRS))


def _demo_known_culverts(roads: gpd.GeoDataFrame, streams: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    records = []
    for road_index, road in roads.iterrows():
        for stream_index, stream in streams.iterrows():
            intersection = road.geometry.intersection(stream.geometry)
            if intersection.is_empty or intersection.geom_type != "Point":
                continue
            keep = stream["stream_order"] >= 2 or (
                road["road_id"] in {"R-002", "R-005"} and stream["stream_id"] == "S-002"
            )
            if keep:
                records.append(
                    {
                        "culvert_id": f"C-{len(records) + 1:03d}",
                        "source": "synthetic_demo",
                        "geometry": Point(intersection.x + 4, intersection.y - 3),
                    }
                )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=DEMO_CRS)


def _write_demo_dem(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 220
    height = 220
    cell_size = 30
    west = 499800 + DEMO_X_OFFSET_M
    north = 4711600 + DEMO_Y_OFFSET_M
    transform = from_origin(west, north, cell_size, cell_size)

    y_indices, x_indices = np.indices((height, width))
    east_gradient = x_indices * 0.12
    south_gradient = y_indices * 0.08
    valley = -8 * np.exp(-((x_indices - 95) ** 2) / 2500) - 5 * np.exp(
        -((y_indices - 120) ** 2) / 1800
    )
    data = (145 + east_gradient + south_gradient + valley).astype("float32")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=DEMO_CRS,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def _move_to_ulster_pilot(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    moved = gdf.copy()
    moved.geometry = moved.geometry.apply(
        lambda geom: translate(geom, xoff=DEMO_X_OFFSET_M, yoff=DEMO_Y_OFFSET_M)
    )
    return moved
