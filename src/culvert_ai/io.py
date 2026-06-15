from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_vector(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    gdf = gpd.read_file(path, layer=layer)
    if "geometry" not in gdf:
        raise ValueError(f"Vector file has no geometry column: {path}")
    if gdf.crs is None:
        raise ValueError(f"Vector file is missing a CRS: {path}")
    return clean_geometry(gdf)


def write_vector(gdf: gpd.GeoDataFrame, path: str | Path, layer: str | None = None) -> None:
    path = Path(path)
    ensure_parent_dir(path)

    if path.suffix.lower() == ".csv":
        table = gdf.copy()
        table["geometry_wkt"] = table.geometry.to_wkt()
        pd.DataFrame(table.drop(columns="geometry")).to_csv(path, index=False)
        return

    if path.suffix.lower() in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
        return

    if path.suffix.lower() != ".gpkg":
        path = path.with_suffix(".gpkg")

    gdf.to_file(path, layer=layer or path.stem, driver="GPKG")


def clean_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def project_layers_to_metric(
    *gdfs: gpd.GeoDataFrame,
) -> tuple[list[gpd.GeoDataFrame], object]:
    """Project layers to one meter-based CRS.

    If the first layer is already projected, its CRS is reused. Otherwise, the CRS is estimated
    from the combined extent, usually as the appropriate UTM zone.
    """

    non_empty = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
    if not non_empty:
        raise ValueError("No non-empty geospatial layers were provided.")

    base_crs = non_empty[0].crs
    if base_crs is None:
        raise ValueError("Input layer is missing a CRS.")

    aligned = [gdf.to_crs(base_crs) if gdf.crs != base_crs else gdf.copy() for gdf in non_empty]
    if getattr(base_crs, "is_projected", False):
        metric_crs = base_crs
    else:
        combined = gpd.GeoDataFrame(
            geometry=pd.concat([layer.geometry for layer in aligned], ignore_index=True),
            crs=base_crs,
        )
        metric_crs = combined.estimate_utm_crs() or "EPSG:3857"

    return [gdf.to_crs(metric_crs) for gdf in aligned], metric_crs


def add_wgs84_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    result = gdf.copy()
    wgs84 = result.to_crs("EPSG:4326")
    result["longitude"] = wgs84.geometry.x
    result["latitude"] = wgs84.geometry.y
    return result


def existing_columns(gdf: gpd.GeoDataFrame, names: Iterable[str]) -> list[str]:
    return [name for name in names if name in gdf.columns]
