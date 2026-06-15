from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from culvert_ai.io import clean_geometry, read_vector, write_vector


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    description: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    focus_places: tuple[str, ...]


REGIONS: dict[str, Region] = {
    "ulster_poughkeepsie": Region(
        key="ulster_poughkeepsie",
        name="Ulster County Poughkeepsie-Area Pilot",
        description=(
            "Ulster County side of the Poughkeepsie/Hudson Valley field area. "
            "Poughkeepsie is in Dutchess County; this region scopes the model to nearby "
            "Ulster County communities and corridors west of the Hudson River."
        ),
        min_lon=-74.35,
        min_lat=41.55,
        max_lon=-73.88,
        max_lat=42.12,
        focus_places=(
            "Highland",
            "Lloyd",
            "Esopus",
            "New Paltz",
            "Marlboro",
            "Plattekill",
            "Rosendale",
            "Kingston south/east approach corridors",
        ),
    )
}


def get_region(region_key: str = "ulster_poughkeepsie") -> Region:
    try:
        return REGIONS[region_key]
    except KeyError as exc:
        valid = ", ".join(sorted(REGIONS))
        raise ValueError(f"Unknown region '{region_key}'. Valid regions: {valid}") from exc


def region_boundary(region_key: str = "ulster_poughkeepsie") -> gpd.GeoDataFrame:
    region = get_region(region_key)
    return gpd.GeoDataFrame(
        [
            {
                "region_key": region.key,
                "name": region.name,
                "description": region.description,
                "focus_places": "; ".join(region.focus_places),
                "geometry": box(region.min_lon, region.min_lat, region.max_lon, region.max_lat),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def write_region_boundary(
    output_path: str | Path,
    region_key: str = "ulster_poughkeepsie",
) -> gpd.GeoDataFrame:
    boundary = region_boundary(region_key)
    write_vector(boundary, output_path)
    return boundary


def filter_to_region(
    data: gpd.GeoDataFrame,
    region_key: str = "ulster_poughkeepsie",
    boundary_path: str | Path | None = None,
    clip: bool = True,
) -> gpd.GeoDataFrame:
    if data.empty:
        return data.copy()
    if data.crs is None:
        raise ValueError("Input data is missing a CRS.")

    boundary = read_vector(boundary_path) if boundary_path else region_boundary(region_key)
    boundary = clean_geometry(boundary).to_crs(data.crs)
    boundary_union = boundary.geometry.unary_union
    matches = data[data.geometry.intersects(boundary_union)].copy()

    if matches.empty or not clip:
        return matches.reset_index(drop=True)

    clipped = gpd.clip(matches, boundary)
    return clean_geometry(clipped).reset_index(drop=True)
