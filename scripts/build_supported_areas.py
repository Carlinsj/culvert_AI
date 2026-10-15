"""Publish map destinations from existing roads inside the loaded terrain extent."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]


def main():
    roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs("EPSG:32618")
    with rasterio.open(ROOT / "data/processed/hydrology/usgs_3dep_10m/dem_projected.tif") as dem:
        terrain = gpd.GeoSeries([box(*dem.bounds)], crs=dem.crs).to_crs(roads.crs).iloc[0]
    safe = terrain.buffer(-16500)
    roads = roads.loc[roads.intersects(safe)].copy()
    roads.geometry = roads.geometry.intersection(safe)
    points = roads.geometry.representative_point()
    xy = np.c_[points.x, points.y]
    first = int(cKDTree(xy).query(np.median(xy, axis=0))[1])
    selected = [first]
    for _ in range(2):
        distances = cKDTree(xy[selected]).query(xy)[0]
        index = int(np.argmax(distances))
        if distances[index] < 10000:
            break
        selected.append(index)
    destinations = gpd.GeoSeries(points.iloc[selected], crs=roads.crs).to_crs("EPSG:4326")
    areas = []
    for number, (index, point) in enumerate(zip(selected, destinations), 1):
        road = roads.iloc[index]
        name = next((str(road[c]) for c in ("name", "FULLNAME", "ref") if c in road and str(road[c]) not in ("None", "nan", "")), "")
        areas.append(dict(name=name or f"Supported road area {number}", latitude=point.y, longitude=point.x, radius_m=4800))
    (ROOT / "web/data/supported-areas.json").write_text(json.dumps(dict(
        description="Explore roads within the currently loaded road and terrain coverage.", areas=areas,
    ), indent=2) + "\n")
    print(f"Published {len(areas)} supported map destinations from the existing road network.")


if __name__ == "__main__":
    main()
