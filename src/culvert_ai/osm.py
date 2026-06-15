from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from culvert_ai.io import clean_geometry, write_vector


DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_CENSUS_COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip"

ROAD_FILTER = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|road|track|"
    "living_street"
)
WATERWAY_FILTER = "river|stream|ditch|drain|canal"


def download_ulster_osm_inputs(
    output_dir: str | Path = "data/raw",
    overpass_url: str = DEFAULT_OVERPASS_URL,
    census_county_url: str = DEFAULT_CENSUS_COUNTY_URL,
    county_name: str = "Ulster County",
    state_name: str = "New York",
    statefp: str = "36",
    countyfp: str = "111",
    timeout_seconds: int = 240,
    tile_size_degrees: float = 0.18,
) -> dict:
    """Download actual OSM roads and waterways for Ulster County.

    These are real public map features, not synthetic demo data. They are suitable for generating
    first-pass candidate crossing predictions before NYSDOT or field-confirmed inventories exist.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = _load_or_download_county_boundary(
        output_dir=output_dir,
        census_county_url=census_county_url,
        statefp=statefp,
        countyfp=countyfp,
    )

    roads = _fetch_tiled_osm_lines(
        overpass_url=overpass_url,
        boundary=boundary,
        tag_key="highway",
        tag_filter=ROAD_FILTER,
        layer_kind="roads",
        timeout_seconds=timeout_seconds,
        tile_size_degrees=tile_size_degrees,
    )
    streams = _fetch_tiled_osm_lines(
        overpass_url=overpass_url,
        boundary=boundary,
        tag_key="waterway",
        tag_filter=WATERWAY_FILTER,
        layer_kind="streams",
        timeout_seconds=timeout_seconds,
        tile_size_degrees=tile_size_degrees,
    )

    if roads.empty:
        raise ValueError("No OSM road lines were downloaded for the requested county.")
    if streams.empty:
        raise ValueError("No OSM waterway lines were downloaded for the requested county.")

    roads_path = output_dir / "roads.gpkg"
    streams_path = output_dir / "streams.gpkg"
    boundary_path = output_dir / "ulster_county_boundary.gpkg"
    metadata_path = output_dir / "osm_download_metadata.json"

    write_vector(roads, roads_path)
    write_vector(streams, streams_path)
    write_vector(boundary, boundary_path)
    metadata = {
        "source": "OpenStreetMap via Overpass API",
        "boundary_source": "U.S. Census Bureau cartographic county boundary",
        "county": county_name,
        "state": state_name,
        "statefp": statefp,
        "countyfp": countyfp,
        "downloaded_at_unix": time.time(),
        "roads": str(roads_path),
        "streams": str(streams_path),
        "boundary": str(boundary_path),
        "road_rows": int(len(roads)),
        "stream_rows": int(len(streams)),
        "overpass_url": overpass_url,
        "census_county_url": census_county_url,
        "tile_size_degrees": tile_size_degrees,
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


def _load_or_download_county_boundary(
    output_dir: Path,
    census_county_url: str,
    statefp: str,
    countyfp: str,
) -> gpd.GeoDataFrame:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    zip_path = source_dir / "cb_2024_us_county_500k.zip"

    if not zip_path.exists():
        urllib.request.urlretrieve(census_county_url, zip_path)

    counties = gpd.read_file(f"zip://{zip_path}").to_crs("EPSG:4326")
    match = counties[
        (counties["STATEFP"].astype(str) == str(statefp))
        & (counties["COUNTYFP"].astype(str) == str(countyfp))
    ].copy()
    if match.empty:
        raise ValueError(f"County boundary not found for STATEFP={statefp}, COUNTYFP={countyfp}.")

    return clean_geometry(match[["STATEFP", "COUNTYFP", "NAME", "GEOID", "geometry"]])


def _fetch_tiled_osm_lines(
    overpass_url: str,
    boundary: gpd.GeoDataFrame,
    tag_key: str,
    tag_filter: str,
    layer_kind: str,
    timeout_seconds: int,
    tile_size_degrees: float,
) -> gpd.GeoDataFrame:
    chunks = []
    for tile in _boundary_tiles(boundary, tile_size_degrees):
        query = _bbox_query(tag_key, tag_filter, tile.bounds, timeout_seconds)
        tile_lines = _fetch_osm_lines(
            overpass_url=overpass_url,
            query=query,
            layer_kind=layer_kind,
            timeout_seconds=timeout_seconds,
        )
        if not tile_lines.empty:
            chunks.append(tile_lines)

    if not chunks:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    merged = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), crs="EPSG:4326")
    merged = merged.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)
    clipped = gpd.clip(merged, boundary)
    clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    return clean_geometry(clipped)


def _boundary_tiles(boundary: gpd.GeoDataFrame, tile_size_degrees: float) -> list:
    minx, miny, maxx, maxy = boundary.total_bounds
    tiles = []
    y = miny
    boundary_union = boundary.geometry.union_all()
    while y < maxy:
        x = minx
        north = min(y + tile_size_degrees, maxy)
        while x < maxx:
            east = min(x + tile_size_degrees, maxx)
            tile = box(x, y, east, north)
            if tile.intersects(boundary_union):
                tiles.append(tile)
            x = east
        y = north
    return tiles


def _bbox_query(
    tag_key: str,
    tag_filter: str,
    bounds: tuple[float, float, float, float],
    timeout_seconds: int,
) -> str:
    west, south, east, north = bounds
    return f"""
[out:json][timeout:{timeout_seconds}];
(
  way["{tag_key}"~"^({tag_filter})$"]({south:.7f},{west:.7f},{north:.7f},{east:.7f});
);
out body geom;
"""


def _fetch_osm_lines(
    overpass_url: str,
    query: str,
    layer_kind: str,
    timeout_seconds: int,
) -> gpd.GeoDataFrame:
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        overpass_url,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "culvert-ai-ulster-research/0.1",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds + 30) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Overpass API returned HTTP {error.code}: {body[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Overpass API: {error}") from error

    data = json.loads(raw)
    records = []
    for element in data.get("elements", []):
        geometry = element.get("geometry") or []
        if element.get("type") != "way" or len(geometry) < 2:
            continue

        coords = [(point["lon"], point["lat"]) for point in geometry if "lon" in point and "lat" in point]
        if len(coords) < 2:
            continue

        line = LineString(coords)
        if line.is_empty or line.length == 0:
            continue

        tags = element.get("tags") or {}
        records.append(_record_from_tags(element["id"], tags, line, layer_kind))

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _record_from_tags(osm_id: int, tags: dict, line: LineString, layer_kind: str) -> dict:
    if layer_kind == "roads":
        name = tags.get("name") or tags.get("ref") or tags.get("highway") or f"osm road {osm_id}"
        return {
            "osm_id": osm_id,
            "id": osm_id,
            "name": name,
            "highway": tags.get("highway"),
            "ref": tags.get("ref"),
            "surface": tags.get("surface"),
            "maxspeed": tags.get("maxspeed"),
            "bridge": tags.get("bridge"),
            "tunnel": tags.get("tunnel"),
            "road_bridge": _truthy(tags.get("bridge")),
            "road_tunnel": _truthy(tags.get("tunnel")),
            "geometry": line,
        }

    name = tags.get("name") or tags.get("waterway") or f"osm waterway {osm_id}"
    tunnel = tags.get("tunnel")
    return {
        "osm_id": osm_id,
        "id": osm_id,
        "name": name,
        "waterway": tags.get("waterway"),
        "intermittent": tags.get("intermittent"),
        "tunnel": tunnel,
        "covered": tags.get("covered"),
        "man_made": tags.get("man_made"),
        "stream_tunnel": tunnel,
        "stream_culvert": str(tunnel).lower() == "culvert"
        or str(tags.get("man_made")).lower() == "culvert",
        "geometry": line,
    }


def _truthy(value) -> bool:
    return str(value).lower() in {"yes", "true", "1", "bridge", "tunnel", "culvert"}
