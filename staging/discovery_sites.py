"""Label-free physical discovery challenger. No inventory, fitting or model inputs.

Small drainage uses existing conditioned Whitebox-native D8 links, not snapped
inventory or uniformly spaced detections. The ten-metre DEM cannot verify a pipe.
"""

import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from shapely.geometry import LineString

from culvert_ai.crossing_geometry import acute_angle_degrees, local_line_tangent
from staging.hydrology_sites import identifier, line_parts, metric
from staging.retrieval_sites import identity_groups

VERSION = "physical-discovery-v1"
SMALL_AREA_M2 = 10000.0  # one hectare, fixed physical hypothesis, not fitted
MINIMUM_ANGLE = 30.0
# Whitebox native, NOT ESRI: NE,E,SE,S,SW,W,NW,N.
D8 = {
    1: (-1, 1),
    2: (0, 1),
    4: (1, 1),
    8: (1, 0),
    16: (1, -1),
    32: (0, -1),
    64: (-1, -1),
    128: (-1, 0),
}


def drainage_links(
    pointer, areas, transform, row_offset=0, col_offset=0, minimum_area=SMALL_AREA_M2
):
    """Connected raster-centre edges, stable global-cell IDs and upstream areas."""
    if pointer.shape != areas.shape:
        raise ValueError("Pointer and upstream-area grids must align")
    records = []
    for code, (dr, dc) in D8.items():
        rows, cols = np.where((pointer == code) & np.isfinite(areas) & (areas >= minimum_area))
        valid = (
            (rows + dr >= 0)
            & (rows + dr < pointer.shape[0])
            & (cols + dc >= 0)
            & (cols + dc < pointer.shape[1])
        )
        for r, c in zip(rows[valid], cols[valid]):
            nr, nc = r + dr, c + dc
            if not np.isfinite(areas[nr, nc]) or areas[nr, nc] < minimum_area:
                continue
            a, b = transform * (c + 0.5, r + 0.5), transform * (nc + 0.5, nr + 0.5)
            key = f"d8:{r + row_offset}:{c + col_offset}:{nr + row_offset}:{nc + col_offset}"
            records.append(
                dict(link_key=key, upstream_area_m2=float(areas[r, c]), geometry=LineString([a, b]))
            )
    return gpd.GeoDataFrame(
        records,
        columns=["link_key", "upstream_area_m2", "geometry"],
        geometry="geometry",
        crs=32618,
    )


def small_drainage(roads, hydro, bounds):
    """Only read a bounded existing raster window. Never extend terrain support."""
    with (
        rasterio.open(hydro / "flow_direction.tif") as pointer,
        rasterio.open(hydro / "upstream_area_m2.tif") as areas,
    ):
        if (
            pointer.crs.to_epsg() != 32618
            or areas.crs != pointer.crs
            or pointer.transform != areas.transform
            or pointer.shape != areas.shape
        ):
            raise ValueError("D8 and upstream-area rasters require aligned EPSG:32618 grids")
        meta = json.loads((hydro / "hydrology_metadata.json").read_text())
        if meta["routing"]["pointer_scheme"] != "Whitebox native":
            raise ValueError("Unsupported D8 encoding")
        window = (
            from_bounds(*bounds, transform=pointer.transform)
            .round_offsets()
            .round_lengths()
            .intersection(Window(0, 0, pointer.width, pointer.height))
        )
        p = pointer.read(1, window=window, masked=True).astype(float).filled(np.nan)
        a = areas.read(1, window=window, masked=True).astype(float).filled(np.nan)
        links = drainage_links(
            p, a, pointer.window_transform(window), int(window.row_off), int(window.col_off)
        )
    return crossing_links(roads, links), links


def crossing_links(roads, links):
    """Actual transverse road/D8-edge intersections; no nearest-road relocation."""
    records = []
    roads = metric(roads).sort_values("id", kind="stable")
    for road in roads.itertuples():
        for part, line in enumerate(line_parts(road.geometry)):
            for j in sorted(links.sindex.query(line, predicate="intersects")):
                water = links.iloc[j]
                intersection = line.intersection(water.geometry)
                points = (
                    [intersection]
                    if intersection.geom_type == "Point"
                    else [v for v in getattr(intersection, "geoms", []) if v.geom_type == "Point"]
                )
                for point in points:
                    _, road_tangent = local_line_tangent(line, point)
                    _, flow_tangent = local_line_tangent(water.geometry, point)
                    angle = acute_angle_degrees(road_tangent, flow_tangent)
                    if not np.isfinite(angle) or angle < MINIMUM_ANGLE:
                        continue
                    # Require the D8 link endpoints on opposite sides of the
                    # local road, not merely an endpoint touch or parallel ditch.
                    tangent = np.asarray(road_tangent)
                    ends = np.asarray(water.geometry.coords) - np.array([point.x, point.y])
                    side = tangent[0] * ends[:, 1] - tangent[1] * ends[:, 0]
                    if side[0] * side[1] >= -1e-8:
                        continue
                    chainage = line.project(point)
                    key = str(water.link_key)
                    records.append(
                        dict(
                            candidate_id="small_"
                            + identifier(road.id, part, key, f"{chainage:.6f}"),
                            road_id=str(road.id),
                            road_name=getattr(road, "name", str(road.id)),
                            route_id=str(getattr(road, "route_id", road.id)),
                            road_part_index=part,
                            road_chainage_m=chainage,
                            route_part_index=part,
                            route_sample_distance_m=chainage,
                            proposal_kind="small_dem_crossing",
                            evidence_source="conditioned_D8_1ha_transverse_link",
                            channel_key=key,
                            upstream_area_at_link_m2=float(water.upstream_area_m2),
                            crossing_angle_degrees=angle,
                            bilateral_drainage=True,
                            source="route_interval_sample",
                            geometry=point,
                        )
                    )
    columns = [
        "candidate_id",
        "road_id",
        "road_name",
        "route_id",
        "road_part_index",
        "road_chainage_m",
        "route_part_index",
        "route_sample_distance_m",
        "proposal_kind",
        "evidence_source",
        "channel_key",
        "upstream_area_at_link_m2",
        "crossing_angle_degrees",
        "bilateral_drainage",
        "source",
        "geometry",
    ]
    return (
        gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=32618)
        .drop_duplicates("candidate_id")
        .reset_index(drop=True)
    )


def adaptive_proposals(raw):
    """Within physically supported road intervals, select real measured extrema.

    New twenty-metre shoulder extrema permit smaller adjacent depressions than
    the sixty-metre peak exclusion of the baseline. Never fill empty road gaps.
    No scores or inventory influence admission or representative coordinates.
    """
    result = raw.copy()
    result["adaptive_profile"] = False
    result["adaptive_flow"] = False
    probes = result.loc[result.proposal_kind.eq("internal_road_probe")]
    for _, group in probes.groupby(["road_id", "road_part_index"], sort=True):
        group = group.sort_values(["road_chainage_m", "candidate_id"], kind="stable")
        chain = group.road_chainage_m.to_numpy(float)
        elevation = group.elevation_m.to_numpy(float)
        area = group.upstream_area_m2.to_numpy(float)
        for i, index in enumerate(group.index):
            left = np.flatnonzero((chain >= chain[i] - 20 - 1e-8) & (chain < chain[i] - 1e-8))
            right = np.flatnonzero((chain > chain[i] + 1e-8) & (chain <= chain[i] + 20 + 1e-8))
            if not len(left) or not len(right):
                continue
            window = np.r_[left, i, right]
            if np.isfinite(elevation[window]).all():
                dip = min(elevation[left].max(), elevation[right].max()) - elevation[i]
                is_min = elevation[i] <= elevation[window].min() and dip >= 0.15
                tied = [j for j in window if elevation[j] == elevation[i]]
                if is_min and str(group.iloc[i].candidate_id) == min(
                    str(group.iloc[j].candidate_id) for j in tied
                ):
                    result.loc[index, "adaptive_profile"] = True
            if (
                np.isfinite(area[window]).all()
                and area[i] >= SMALL_AREA_M2
                and group.iloc[i].negative_tpi_9x9_m > 0
            ):
                tied = [j for j in window if area[j] == area[i]]
                if (
                    area[i] >= area[window].max()
                    and area[i] > min(area[left].min(), area[right].min())
                    and str(group.iloc[i].candidate_id)
                    == min(str(group.iloc[j].candidate_id) for j in tied)
                ):
                    result.loc[index, "adaptive_flow"] = True
    result["physical_crossing"] = result.proposal_kind.isin(
        ["mapped_crossing", "dem_crossing", "small_dem_crossing"]
    )
    result["discovery_admitted"] = (
        result.physical_crossing | result.adaptive_profile | result.adaptive_flow
    )
    result["localization_evidence"] = [
        json.dumps(
            [
                name
                for name, value in [
                    ("physical_crossing", r.physical_crossing),
                    ("adaptive_profile", r.adaptive_profile),
                    ("adaptive_flow", r.adaptive_flow),
                ]
                if value
            ]
        )
        for r in result.itertuples()
    ]
    result["selection_policy"] = VERSION
    return result


def physical_groups(points):
    """Strict identity grouping keeps separate nearby channels and road parts."""
    return identity_groups(points, radius_m=20)


def ordered(points, score_column="model_score", policy="raw_rf"):
    """Order fixed representatives; never select representative by model score."""
    from culvert_ai.inference import dynamic_route_z

    frame = points.copy()
    if not np.isfinite(frame[score_column]).all():
        raise ValueError("Finite compatible model scores required")
    if policy == "dynamic_route_z":
        frame["priority"] = dynamic_route_z(
            frame.assign(flow_c_score=frame[score_column])
        ).route_z_score.to_numpy()
    elif policy == "raw_rf":
        frame["priority"] = frame[score_column]
    else:
        raise ValueError("Unknown fixed ranking policy")
    frame = frame.sort_values(
        ["priority", score_column, "candidate_id"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    frame["global_rank"] = np.arange(1, len(frame) + 1)
    frame["ranking_policy"] = policy
    return frame
