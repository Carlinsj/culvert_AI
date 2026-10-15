"""Application boundary for the frozen Culvert AI inference pipeline.

This module validates uploaded road data and serializes a deliberately small,
ground-truth-free response for the web application. It never fits a model.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, box

from culvert_ai.flow_c_bundle import REQUIRED_FEATURE_CRS, load_frozen_flow_c_bundle
from culvert_ai.inference import predict_culverts


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "flow_c_frozen_v1.joblib"
DEFAULT_STREAMS = ROOT / "data" / "raw" / "streams.gpkg"
DEFAULT_ROADS = ROOT / "data" / "raw" / "roads.gpkg"
DEFAULT_DEM = ROOT / "data" / "processed" / "hydrology" / "usgs_3dep_10m" / "dem_projected.tif"
DEFAULT_FLOW = ROOT / "data" / "processed" / "hydrology" / "usgs_3dep_10m" / "flow_accumulation_cells.tif"
DEFAULT_DRAINAGE = ROOT / "data" / "processed" / "hydrology" / "usgs_3dep_10m" / "upstream_area_m2.tif"
DEFAULT_HYDROLOGY = ROOT / "data" / "processed" / "hydrology" / "usgs_3dep_10m"

MISSING_CRS_MESSAGE = "We could not determine the coordinate system for this road data."
OUTSIDE_COVERAGE_MESSAGE = (
    "This area is outside the currently supported Culvert AI analysis coverage."
)
PARTIAL_COVERAGE_MESSAGE = "Required terrain data are not available for part of this road network."
MODEL_ERROR_MESSAGE = (
    "Culvert AI could not verify the prediction model. Analysis has been disabled."
)
SCHEMA_ERROR_MESSAGE = "The analysis data are not compatible with the current Culvert AI model."

PUBLIC_SITE_FIELDS = (
    "global_rank",
    "route_local_rank",
    "route_id",
    "road_id",
    "road_name",
    "latitude",
    "longitude",
    "candidate_id",
    "physical_site_id",
    "flow_c_score",
    "route_z_score",
    "priority_percentile",
    "route_z_method",
    "route_z_candidate_count",
    "distance_to_nearest_stream_m",
    "route_low_point_score",
    "local_road_stream_separation_m",
    "distance_to_nearest_mapped_crossing_m",
    "dem_culvert_terrain_score",
    "hydrology_edge_risk",
    "log1p_distance_to_dem_road_drainage_crossing_0p1km2_m",
    "log1p_distance_to_dem_road_drainage_crossing_2km2_m",
)
FORBIDDEN_PUBLIC_TOKENS = ("known", "ground_truth", "nysdot", "label", "match_distance")


class ProductInputError(ValueError):
    """Expected user-correctable ingestion failure."""

    def __init__(self, message: str, *, code: str, technical: str = ""):
        super().__init__(message)
        self.code = code
        self.technical = technical


def model_status(model_path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Fail closed after validating the complete on-disk artifact set."""
    path = Path(model_path)
    try:
        bundle = load_frozen_flow_c_bundle(path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "ready": True,
            "display_name": "FLOW-C Frozen v1",
            "bundle_hash": manifest["bundle_sha256"],
            "bundle_hash_short": manifest["bundle_sha256"][:12],
            "specification": bundle["flow_c_specification"],
            "analysis_crs": REQUIRED_FEATURE_CRS,
        }
    except Exception as error:  # startup must fail closed with a friendly public message
        LOGGER.exception("Frozen model validation failed")
        return {
            "ready": False,
            "display_name": "FLOW-C Frozen v1",
            "message": MODEL_ERROR_MESSAGE,
            "technical": str(error),
        }


def decode_roads(payload: Mapping[str, Any], *, workspace: Path) -> tuple[gpd.GeoDataFrame, str]:
    filename = Path(str(payload.get("filename") or "roads.geojson")).name
    suffix = Path(filename).suffix.lower()
    raw = _payload_bytes(payload)
    if suffix in {".geojson", ".json"}:
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProductInputError(
                "This file is not valid GeoJSON.", code="invalid_vector", technical=str(error)
            ) from error
        if not document.get("crs"):
            raise ProductInputError(MISSING_CRS_MESSAGE, code="missing_crs")
    elif suffix not in {".gpkg", ".zip"}:
        raise ProductInputError(
            "Choose a GeoJSON, GeoPackage, or zipped shapefile.",
            code="unsupported_format",
            technical=f"Unsupported extension: {suffix or '<none>'}",
        )

    input_path = workspace / filename
    input_path.write_bytes(raw)
    read_path = f"zip://{input_path}" if suffix == ".zip" else input_path
    try:
        roads = gpd.read_file(read_path)
    except Exception as error:
        raise ProductInputError(
            "We could not read this road data file.", code="invalid_vector", technical=str(error)
        ) from error
    if roads.crs is None:
        raise ProductInputError(MISSING_CRS_MESSAGE, code="missing_crs")
    if roads.empty:
        raise ProductInputError("This file does not contain any road features.", code="empty_input")
    geometry_types = set(roads.geometry.geom_type.dropna())
    if not geometry_types or not geometry_types.issubset({"LineString", "MultiLineString"}):
        raise ProductInputError(
            "Road data must contain line geometry.",
            code="invalid_geometry",
            technical=f"Geometry types: {sorted(geometry_types)}",
        )
    return roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy(), filename


def inspect_roads(roads: gpd.GeoDataFrame, filename: str, *, dem_path: str | Path = DEFAULT_DEM) -> dict:
    original_crs = roads.crs.to_string()
    try:
        analysis_roads = roads.to_crs(REQUIRED_FEATURE_CRS)
    except Exception as error:
        raise ProductInputError(
            MISSING_CRS_MESSAGE, code="ambiguous_crs", technical=str(error)
        ) from error
    coverage = _coverage_status(analysis_roads, dem_path)
    if coverage == "outside":
        raise ProductInputError(OUTSIDE_COVERAGE_MESSAGE, code="outside_coverage")
    if coverage == "partial":
        raise ProductInputError(PARTIAL_COVERAGE_MESSAGE, code="partial_coverage")
    display = roads.to_crs("EPSG:4326")
    bounds = [round(float(value), 6) for value in display.total_bounds]
    return {
        "filename": filename,
        "feature_count": int(len(roads)),
        "detected_crs": original_crs,
        "analysis_crs": REQUIRED_FEATURE_CRS,
        "crs_transformed": original_crs.upper() != REQUIRED_FEATURE_CRS,
        "extent_wgs84": bounds,
        "coverage": "available",
        "roads": json.loads(display.to_json(drop_id=True)),
    }


def analyze_payload(payload: Mapping[str, Any], *, settings: Mapping[str, Path] | None = None) -> dict:
    paths = _paths(settings)
    status = model_status(paths["model"])
    if not status["ready"]:
        raise ProductInputError(MODEL_ERROR_MESSAGE, code="model_unavailable", technical=status["technical"])
    with tempfile.TemporaryDirectory(prefix="culvert-ai-upload-") as directory:
        roads, filename = decode_roads(payload, workspace=Path(directory))
        summary = inspect_roads(roads, filename, dem_path=paths["dem"])
        sites = _predict_roads(roads.to_crs(REQUIRED_FEATURE_CRS), payload, paths)

    return {
        "analysis": summary,
        "model": status,
        "site_count": int(len(sites)),
        "sites": _public_geojson(sites),
    }


def analyze_nearby_payload(
    payload: Mapping[str, Any], *, settings: Mapping[str, Path] | None = None
) -> dict:
    """Run frozen inference on the existing road network around a map location."""
    try:
        latitude = float(payload.get("latitude"))
        longitude = float(payload.get("longitude"))
        radius_m = float(payload.get("radius_m", 5000))
    except (TypeError, ValueError) as error:
        raise ProductInputError(
            "Choose a valid map location to continue.", code="invalid_location"
        ) from error
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ProductInputError("Choose a valid map location to continue.", code="invalid_location")
    if not 500 <= radius_m <= 16000:
        raise ProductInputError(
            "Choose a search radius between 0.5 and 16 kilometers.", code="invalid_radius"
        )

    paths = _paths(settings)
    status = model_status(paths["model"])
    if not status["ready"]:
        raise ProductInputError(
            MODEL_ERROR_MESSAGE, code="model_unavailable", technical=status["technical"]
        )
    center = gpd.GeoDataFrame(
        {"geometry": [Point(longitude, latitude)]}, crs="EPSG:4326"
    ).to_crs(REQUIRED_FEATURE_CRS)
    search_area = center.geometry.iloc[0].buffer(radius_m)
    coverage_frame = gpd.GeoDataFrame(geometry=[search_area], crs=REQUIRED_FEATURE_CRS)
    coverage = _coverage_status(coverage_frame, paths["dem"])
    if coverage == "outside":
        raise ProductInputError(OUTSIDE_COVERAGE_MESSAGE, code="outside_coverage")
    if coverage == "partial":
        raise ProductInputError(PARTIAL_COVERAGE_MESSAGE, code="partial_coverage")

    roads = gpd.read_file(paths["roads"]).to_crs(REQUIRED_FEATURE_CRS)
    roads = roads.loc[roads.intersects(search_area)].copy()
    if roads.empty:
        raise ProductInputError(
            "No compatible roads were found near this location.", code="no_nearby_roads"
        )
    roads.geometry = roads.geometry.intersection(search_area)
    roads = roads.loc[~roads.geometry.is_empty].reset_index(drop=True)
    sites = _predict_roads(roads, payload, paths)
    display_roads = roads.to_crs("EPSG:4326")
    return {
        "analysis": {
            "source": "current_location",
            "center": {"latitude": latitude, "longitude": longitude},
            "radius_m": radius_m,
            "feature_count": int(len(roads)),
            "analysis_crs": REQUIRED_FEATURE_CRS,
            "coverage": "available",
            "roads": json.loads(display_roads.to_json(drop_id=True)),
        },
        "model": status,
        "site_count": int(len(sites)),
        "sites": _public_geojson(sites),
    }


def _predict_roads(
    analysis_roads: gpd.GeoDataFrame,
    payload: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> gpd.GeoDataFrame:
    route_column = _route_column(analysis_roads, payload.get("route_id_column"))
    streams = gpd.read_file(paths["streams"]).to_crs(REQUIRED_FEATURE_CRS)
    crossings = {
        threshold: gpd.read_parquet(
            paths["hydrology"] / f"road_drainage_crossings_{threshold}.parquet"
        )
        for threshold in ("0p1km2", "0p5km2", "2km2", "10km2")
    }
    try:
        return predict_culverts(
            analysis_roads,
            streams=streams,
            dem_path=paths["dem"],
            flow_c_model=paths["model"],
            route_id_column=route_column,
            flow_accumulation_path=paths["flow"],
            drainage_area_path=paths["drainage"],
            dem_crossings=crossings,
        )
    except ProductInputError:
        raise
    except Exception as error:
        LOGGER.exception("Culvert analysis failed")
        raise ProductInputError(
            SCHEMA_ERROR_MESSAGE, code="analysis_failed", technical=str(error)
        ) from error


def _payload_bytes(payload: Mapping[str, Any]) -> bytes:
    if isinstance(payload.get("text"), str):
        return payload["text"].encode("utf-8")
    if isinstance(payload.get("content_base64"), str):
        try:
            return base64.b64decode(payload["content_base64"], validate=True)
        except ValueError as error:
            raise ProductInputError("The uploaded file could not be decoded.", code="invalid_upload") from error
    raise ProductInputError("Choose a road data file to continue.", code="missing_upload")


def _coverage_status(roads: gpd.GeoDataFrame, dem_path: str | Path) -> str:
    path = Path(dem_path)
    if not path.is_file():
        raise ProductInputError(
            PARTIAL_COVERAGE_MESSAGE, code="missing_terrain", technical=f"Missing DEM: {path}"
        )
    with rasterio.open(path) as source:
        coverage = box(*source.bounds)
        layer = roads.to_crs(source.crs)
    road_bounds = box(*layer.total_bounds)
    if not coverage.intersects(road_bounds):
        return "outside"
    if not coverage.covers(road_bounds):
        return "partial"
    return "available"


def _route_column(roads: gpd.GeoDataFrame, requested: Any) -> str | None:
    if requested and str(requested) in roads.columns:
        return str(requested)
    for column in ("route_id", "route", "road_id", "name", "FULLNAME", "FULL_NAME", "LINEARID"):
        if column in roads.columns:
            return column
    return None


def _paths(overrides: Mapping[str, Path] | None) -> dict[str, Path]:
    paths = {
        "model": DEFAULT_MODEL,
        "roads": DEFAULT_ROADS,
        "streams": DEFAULT_STREAMS,
        "dem": DEFAULT_DEM,
        "flow": DEFAULT_FLOW,
        "drainage": DEFAULT_DRAINAGE,
        "hydrology": DEFAULT_HYDROLOGY,
    }
    if overrides:
        paths.update({key: Path(value) for key, value in overrides.items()})
    return paths


def _public_geojson(sites: gpd.GeoDataFrame) -> dict:
    forbidden = [
        column for column in sites.columns
        if any(token in column.lower() for token in FORBIDDEN_PUBLIC_TOKENS)
    ]
    if forbidden:
        LOGGER.warning("Dropping forbidden prediction fields: %s", forbidden)
    columns = [column for column in PUBLIC_SITE_FIELDS if column in sites.columns]
    public = sites.loc[:, columns + ["geometry"]].to_crs("EPSG:4326").copy()
    for column in columns:
        if column in public:
            public[column] = public[column].map(_json_scalar)
    return json.loads(public.to_json(drop_id=True))


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _main() -> int:
    parser = argparse.ArgumentParser(description="Culvert AI product API bridge")
    parser.add_argument("operation", choices=("status", "inspect", "analyze", "nearby"))
    args = parser.parse_args()
    try:
        if args.operation == "status":
            result = model_status()
        else:
            payload = json.load(__import__("sys").stdin)
            if args.operation == "nearby":
                result = analyze_nearby_payload(payload)
            else:
                with tempfile.TemporaryDirectory(prefix="culvert-ai-upload-") as directory:
                    roads, filename = decode_roads(payload, workspace=Path(directory))
                    result = inspect_roads(roads, filename) if args.operation == "inspect" else analyze_payload(payload)
        print(json.dumps({"ok": True, "data": result}, allow_nan=False))
        return 0
    except ProductInputError as error:
        print(json.dumps({"ok": False, "error": str(error), "code": error.code, "technical": error.technical}))
        return 2
    except Exception as error:  # pragma: no cover - last-resort bridge protection
        LOGGER.exception("Unexpected product service failure")
        print(json.dumps({"ok": False, "error": "Culvert AI could not complete this request.", "code": "internal", "technical": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
