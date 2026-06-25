from __future__ import annotations

import json
import math
import time
import urllib.request
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge

from culvert_ai.io import ensure_parent_dir, read_vector


DEFAULT_USGS_3DEP_RESOLUTION = "1"


def usgs_3dep_base_url(resolution: str = DEFAULT_USGS_3DEP_RESOLUTION) -> str:
    return f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/{resolution}/TIFF/current"


def usgs_3dep_tile_url(
    tile_id: str,
    resolution: str = DEFAULT_USGS_3DEP_RESOLUTION,
    base_url: str | None = None,
) -> str:
    base = (base_url or usgs_3dep_base_url(resolution)).rstrip("/")
    return f"{base}/{tile_id}/USGS_{resolution}_{tile_id}.tif"


def dem_tiles_for_bounds(bounds: tuple[float, float, float, float]) -> list[str]:
    """Return USGS 3DEP tile IDs intersecting WGS84 bounds.

    USGS 3DEP current elevation tiles use the southwest 1-degree corner in IDs
    such as n41w075. The epsilon avoids adding the next tile when a max bound
    falls exactly on a degree boundary.
    """

    min_lon, min_lat, max_lon, max_lat = bounds
    epsilon = 1e-9
    lon_start = math.floor(min_lon)
    lon_stop = math.floor(max_lon - epsilon)
    lat_start = math.floor(min_lat)
    lat_stop = math.floor(max_lat - epsilon)

    tiles = []
    for lat in range(lat_start, lat_stop + 1):
        for lon in range(lon_start, lon_stop + 1):
            tiles.append(_tile_id(lat, lon))
    return list(dict.fromkeys(tiles))


def download_usgs_3dep_dem(
    boundary_path: str | Path,
    output_path: str | Path = "data/raw/dem.tif",
    source_dir: str | Path = "data/raw/sources/dem",
    resolution: str = DEFAULT_USGS_3DEP_RESOLUTION,
    base_url: str | None = None,
    buffer_degrees: float = 0.02,
    overwrite: bool = False,
) -> dict:
    """Download and mosaic USGS 3DEP DEM tiles covering a boundary layer."""

    output_path = Path(output_path)
    source_dir = Path(source_dir)
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")

    if output_path.exists() and not overwrite:
        return {
            "dem": output_path,
            "metadata": metadata_path if metadata_path.exists() else None,
            "skipped": True,
            "reason": "DEM already exists. Use --overwrite or REFRESH_DEM=1 to rebuild.",
        }

    boundary = read_vector(boundary_path).to_crs("EPSG:4326")
    bounds = _expanded_bounds(tuple(float(value) for value in boundary.total_bounds), buffer_degrees)
    tile_ids = dem_tiles_for_bounds(bounds)
    if not tile_ids:
        raise ValueError(f"No DEM tiles found for bounds: {bounds}")

    source_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = [
        _download_if_missing(
            usgs_3dep_tile_url(tile_id, resolution=resolution, base_url=base_url),
            source_dir / f"USGS_{resolution}_{tile_id}.tif",
        )
        for tile_id in tile_ids
    ]

    ensure_parent_dir(output_path)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in tile_paths]
        source_crs = sources[0].crs
        if source_crs is None:
            raise ValueError(f"DEM tile is missing a CRS: {tile_paths[0]}")

        crop_bounds = tuple(float(value) for value in boundary.to_crs(source_crs).total_bounds)
        mosaic, transform = merge(sources, bounds=crop_bounds)
        metadata = sources[0].meta.copy()
        metadata.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(mosaic)

    download_metadata = {
        "source": "USGS 3DEP",
        "resolution": resolution,
        "downloaded_at_unix": time.time(),
        "boundary": str(boundary_path),
        "bounds_wgs84": list(bounds),
        "tiles": tile_ids,
        "tile_paths": [str(path) for path in tile_paths],
        "tile_urls": [
            usgs_3dep_tile_url(tile_id, resolution=resolution, base_url=base_url)
            for tile_id in tile_ids
        ],
        "dem": str(output_path),
        "crs": str(metadata.get("crs")),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
    }
    metadata_path.write_text(json.dumps(download_metadata, indent=2), encoding="utf-8")

    return {
        "dem": output_path,
        "metadata": metadata_path,
        "tiles": tile_ids,
        "tile_count": len(tile_ids),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
    }


def _tile_id(lat_floor: int, lon_floor: int) -> str:
    lat_prefix = "n" if lat_floor >= 0 else "s"
    lon_prefix = "e" if lon_floor >= 0 else "w"
    return f"{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}"


def _expanded_bounds(
    bounds: tuple[float, float, float, float],
    buffer_degrees: float,
) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bounds
    buffer = max(0.0, float(buffer_degrees))
    return (min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)


def _download_if_missing(url: str, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")
    if partial_path.exists():
        partial_path.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "culvert-ai-ulster-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=300) as response, partial_path.open(
        "wb"
    ) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    partial_path.replace(output_path)
    return output_path
