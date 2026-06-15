from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import geopandas as gpd

from culvert_ai.io import clean_geometry, write_vector


DEFAULT_TIGER_YEAR = "2024"
DEFAULT_COUNTY_BOUNDARY_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip"


def download_ulster_census_inputs(
    output_dir: str | Path = "data/raw",
    tiger_year: str = DEFAULT_TIGER_YEAR,
    statefp: str = "36",
    countyfp: str = "111",
    county_boundary_url: str = DEFAULT_COUNTY_BOUNDARY_URL,
) -> dict:
    """Download actual county-level TIGER/Line roads and linear-water data for Ulster County."""

    output_dir = Path(output_dir)
    source_dir = output_dir / "sources"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    geoid = f"{statefp}{countyfp}"
    roads_url = (
        f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/ROADS/"
        f"tl_{tiger_year}_{geoid}_roads.zip"
    )
    water_url = (
        f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/LINEARWATER/"
        f"tl_{tiger_year}_{geoid}_linearwater.zip"
    )

    roads_zip = _download_if_missing(roads_url, source_dir / f"tl_{tiger_year}_{geoid}_roads.zip")
    water_zip = _download_if_missing(
        water_url,
        source_dir / f"tl_{tiger_year}_{geoid}_linearwater.zip",
    )
    boundary_zip = _download_if_missing(
        county_boundary_url,
        source_dir / f"cb_{tiger_year}_us_county_500k.zip",
    )

    roads = _normalize_roads(gpd.read_file(f"zip://{roads_zip}"))
    streams = _normalize_linear_water(gpd.read_file(f"zip://{water_zip}"))
    boundary = _county_boundary(boundary_zip, statefp, countyfp)

    roads_path = output_dir / "roads.gpkg"
    streams_path = output_dir / "streams.gpkg"
    boundary_path = output_dir / "ulster_county_boundary.gpkg"
    metadata_path = output_dir / "census_download_metadata.json"

    write_vector(roads, roads_path)
    write_vector(streams, streams_path)
    write_vector(boundary, boundary_path)

    metadata = {
        "source": "U.S. Census Bureau TIGER/Line",
        "tiger_year": tiger_year,
        "statefp": statefp,
        "countyfp": countyfp,
        "downloaded_at_unix": time.time(),
        "roads_url": roads_url,
        "linear_water_url": water_url,
        "county_boundary_url": county_boundary_url,
        "roads": str(roads_path),
        "streams": str(streams_path),
        "boundary": str(boundary_path),
        "road_rows": int(len(roads)),
        "stream_rows": int(len(streams)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "roads": roads_path,
        "streams": streams_path,
        "boundary": boundary_path,
        "metadata": metadata_path,
        "road_rows": int(len(roads)),
        "stream_rows": int(len(streams)),
    }


def _download_if_missing(url: str, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "culvert-ai-ulster-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())
    return output_path


def _normalize_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    roads = clean_geometry(roads).to_crs("EPSG:4326")
    normalized = roads.copy()
    normalized["id"] = normalized.get("LINEARID", normalized.index.astype(str))
    normalized["name"] = normalized.get("FULLNAME", normalized["id"]).fillna(normalized["id"])
    normalized["road_highway"] = normalized.get("MTFCC", "")
    normalized["highway"] = normalized.get("MTFCC", "")
    normalized["road_bridge"] = False
    normalized["road_tunnel"] = False
    normalized["source_dataset"] = "census_tiger_roads"
    return normalized


def _normalize_linear_water(water: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    water = clean_geometry(water).to_crs("EPSG:4326")
    normalized = water.copy()
    normalized["id"] = normalized.get("LINEARID", normalized.index.astype(str))
    normalized["name"] = normalized.get("FULLNAME", normalized["id"]).fillna(normalized["id"])
    normalized["stream_waterway"] = normalized.get("MTFCC", "")
    normalized["waterway"] = normalized.get("MTFCC", "")
    normalized["stream_tunnel"] = ""
    normalized["stream_culvert"] = False
    normalized["source_dataset"] = "census_tiger_linearwater"
    return normalized


def _county_boundary(zip_path: Path, statefp: str, countyfp: str) -> gpd.GeoDataFrame:
    counties = gpd.read_file(f"zip://{zip_path}").to_crs("EPSG:4326")
    match = counties[
        (counties["STATEFP"].astype(str) == str(statefp))
        & (counties["COUNTYFP"].astype(str) == str(countyfp))
    ].copy()
    if match.empty:
        raise ValueError(f"County boundary not found for STATEFP={statefp}, COUNTYFP={countyfp}.")
    return clean_geometry(match[["STATEFP", "COUNTYFP", "NAME", "GEOID", "geometry"]])
