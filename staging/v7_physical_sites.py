"""Label-free v7 physical crossing hypotheses for surveyed-road diagnostics.

Inventory coordinates are intentionally absent. Dense road probes remain an
internal measurement grid; every returned site has explicit D8 or multiscale
concave-road-profile evidence and a coordinate derived from physical geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point

from staging.discovery_sites import crossing_links, drainage_links
from staging.hydrology_sites import identifier, line_parts, metric


VERSION = "physical-crossings-v7.1-development"
DEFAULT_D8_AREA_M2 = 10_000.0
DIAGNOSTIC_D8_AREAS_M2 = (2_500.0, 5_000.0, 10_000.0)
TERRAIN_GROUP_DISTANCE_M = 15.0
EXACT_CROSSING_DUPLICATE_M = 2.0


def multiscale_d8_crossings(
    roads: gpd.GeoDataFrame,
    hydro: str | Path,
    *,
    thresholds_m2: tuple[float, ...] = DIAGNOSTIC_D8_AREAS_M2,
    padding_m: float = 100.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Extract sub-0.1 km² D8 links once and annotate nested area support."""
    roads = metric(roads)
    hydro = Path(hydro)
    thresholds = tuple(sorted({float(value) for value in thresholds_m2}))
    if not thresholds or thresholds[0] <= 0:
        raise ValueError("Positive drainage-area thresholds are required")
    bounds = roads.total_bounds
    requested = (
        bounds[0] - padding_m,
        bounds[1] - padding_m,
        bounds[2] + padding_m,
        bounds[3] + padding_m,
    )
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
        window = (
            from_bounds(*requested, transform=pointer.transform)
            .round_offsets()
            .round_lengths()
            .intersection(Window(0, 0, pointer.width, pointer.height))
        )
        pointer_values = pointer.read(1, window=window, masked=True).astype(float).filled(np.nan)
        area_values = areas.read(1, window=window, masked=True).astype(float).filled(np.nan)
        links = drainage_links(
            pointer_values,
            area_values,
            pointer.window_transform(window),
            int(window.row_off),
            int(window.col_off),
            thresholds[0],
        )
    crossings = crossing_links(roads, links)
    crossings["threshold_support_m2"] = [
        json.dumps([value for value in thresholds if area >= value])
        for area in crossings.upstream_area_at_link_m2
    ]
    crossings["threshold_support_count"] = [
        sum(area >= value for value in thresholds)
        for area in crossings.upstream_area_at_link_m2
    ]
    crossings["minimum_supported_area_m2"] = thresholds[0]
    crossings["drainage_path_identity"] = crossings.channel_key.astype(str)
    crossings["candidate_source"] = "multiscale_conditioned_d8_crossing"
    crossings["physical_evidence"] = [
        json.dumps(
            {
                "conditioned_d8_link": True,
                "opposite_road_sides": True,
                "upstream_area_m2": float(area),
                "crossing_angle_degrees": float(angle),
            },
            sort_keys=True,
        )
        for area, angle in zip(
            crossings.upstream_area_at_link_m2,
            crossings.crossing_angle_degrees,
            strict=False,
        )
    ]
    crossings["selection_policy"] = VERSION
    road_zone = roads.geometry.buffer(60).union_all()
    supporting_links = links.loc[links.geometry.intersects(road_zone)].copy().reset_index(drop=True)
    return crossings, supporting_links


def _quadratic_low_point(chainage: np.ndarray, elevation: np.ndarray, index: int) -> float:
    """Sub-probe road-chainage localization from three physical DEM samples."""
    if index <= 0 or index >= len(chainage) - 1:
        return float(chainage[index])
    x = chainage[index - 1 : index + 2] - chainage[index]
    z = elevation[index - 1 : index + 2]
    if not np.isfinite(x).all() or not np.isfinite(z).all() or np.max(np.diff(chainage[index - 1:index + 2])) > 10.25:
        return float(chainage[index])
    coefficients = np.polyfit(x, z, 2)
    if coefficients[0] <= 0 or not np.isfinite(coefficients).all():
        return float(chainage[index])
    offset = float(np.clip(-coefficients[1] / (2 * coefficients[0]), x[0], x[-1]))
    return float(chainage[index] + offset)


def terrain_crossing_hypotheses(
    measured_probes: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return local road lows supported by concavity at two DEM scales.

    The 10 m probes are measurements only. Final coordinates are interpolated
    along the road to the local quadratic elevation minimum.
    """
    required = {
        "candidate_id",
        "road_id",
        "road_part_index",
        "road_chainage_m",
        "proposal_kind",
        "elevation_m",
        "negative_tpi_9x9_m",
        "negative_tpi_15x15_m",
        "route_elevation_dip_60m",
        "upstream_area_m2",
    }
    missing = required - set(measured_probes.columns)
    if missing:
        raise ValueError(f"Missing terrain measurements: {sorted(missing)}")
    roads = metric(roads)
    line_lookup = {
        (str(row.id), part): line
        for row in roads.itertuples()
        for part, line in enumerate(line_parts(row.geometry))
    }
    probes = measured_probes.loc[
        measured_probes.proposal_kind.eq("internal_road_probe")
    ].copy()
    selected = []
    for key, group in probes.groupby(["road_id", "road_part_index"], sort=True):
        group = group.sort_values(["road_chainage_m", "candidate_id"], kind="stable")
        chainage = pd.to_numeric(group.road_chainage_m, errors="coerce").to_numpy(float)
        elevation = pd.to_numeric(group.elevation_m, errors="coerce").to_numpy(float)
        tpi9 = pd.to_numeric(group.negative_tpi_9x9_m, errors="coerce").to_numpy(float)
        tpi15 = pd.to_numeric(group.negative_tpi_15x15_m, errors="coerce").to_numpy(float)
        line = line_lookup.get((str(key[0]), int(key[1])))
        if line is None:
            continue
        for position in range(2, len(group) - 2):
            local = elevation[position - 2 : position + 3]
            if not np.isfinite(local).all() or not np.isfinite([tpi9[position], tpi15[position]]).all():
                continue
            local_dip = min(np.max(local[:2]), np.max(local[3:])) - elevation[position]
            is_minimum = elevation[position] <= np.min(local) + 1e-8
            multiscale_concave = tpi9[position] > 0 and tpi15[position] > 0
            if not (is_minimum and local_dip >= 0.15 and multiscale_concave):
                continue
            tied = np.flatnonzero(np.isclose(local, elevation[position])) + position - 2
            candidate_ids = group.iloc[tied].candidate_id.astype(str)
            if str(group.iloc[position].candidate_id) != candidate_ids.min():
                continue
            refined = _quadratic_low_point(chainage, elevation, position)
            source = group.iloc[position].copy()
            source["original_anchor_candidate_id"] = str(source.candidate_id)
            source["road_chainage_m"] = refined
            source["route_sample_distance_m"] = refined
            source["geometry"] = line.interpolate(refined)
            source["candidate_id"] = "v7terrain_" + identifier(
                key[0], key[1], f"{refined:.4f}", source.original_anchor_candidate_id
            )
            source["proposal_kind"] = "multiscale_concave_road_low"
            source["candidate_source"] = "road_profile_terrain_crossing"
            source["evidence_source"] = "road_low_plus_9x9_15x15_concavity"
            source["channel_key"] = "terrain:" + str(source.candidate_id)
            source["drainage_path_identity"] = ""
            source["localization_method"] = "quadratic_dem_low_interpolated_along_road"
            source["localization_shift_m"] = abs(refined - chainage[position])
            source["local_profile_dip_20m"] = float(local_dip)
            source["physical_evidence"] = json.dumps(
                {
                    "local_profile_dip_20m": float(local_dip),
                    "negative_tpi_9x9_m": float(tpi9[position]),
                    "negative_tpi_15x15_m": float(tpi15[position]),
                    "upstream_area_m2": float(source.upstream_area_m2),
                },
                sort_keys=True,
            )
            source["selection_policy"] = VERSION
            selected.append(source)
    if not selected:
        return gpd.GeoDataFrame(columns=[*probes.columns, "geometry"], geometry="geometry", crs=roads.crs)
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=roads.crs).reset_index(drop=True)


def conservative_consolidation(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge only geometry-supported duplicate representations of one crossing."""
    if candidates.empty:
        return candidates.copy()
    frame = metric(candidates)
    if not frame.candidate_id.is_unique:
        raise ValueError("Candidate IDs must be unique before consolidation")
    frame["_explicit"] = ~frame.candidate_source.astype(str).str.contains("terrain")
    frame["_d8"] = frame.candidate_source.astype(str).eq("multiscale_conditioned_d8_crossing")
    frame["_priority"] = np.select(
        [frame._d8, frame._explicit], [0, 1], default=2
    )
    ordered = frame.sort_values(["_priority", "candidate_id"], kind="stable")
    groups: list[list[int]] = []
    for index, row in ordered.iterrows():
        chosen = None
        for group_number, members in enumerate(groups):
            representative = frame.loc[members[0]]
            if (
                str(row.road_id) != str(representative.road_id)
                or int(row.road_part_index) != int(representative.road_part_index)
            ):
                continue
            distance = float(row.geometry.distance(representative.geometry))
            chainage = abs(float(row.road_chainage_m) - float(representative.road_chainage_m))
            if row._explicit and representative._explicit:
                same_resolved_crossing = distance <= EXACT_CROSSING_DUPLICATE_M and chainage <= EXACT_CROSSING_DUPLICATE_M
                d8_area = float(row.get("upstream_area_at_link_m2", np.nan)) if row._d8 else float(representative.get("upstream_area_at_link_m2", np.nan))
                nested_existing_network = (
                    bool(row._d8) != bool(representative._d8)
                    and np.isfinite(d8_area)
                    and d8_area >= 100_000
                    and distance <= TERRAIN_GROUP_DISTANCE_M
                    and chainage <= TERRAIN_GROUP_DISTANCE_M
                )
                if not (same_resolved_crossing or nested_existing_network):
                    continue
            elif distance > TERRAIN_GROUP_DISTANCE_M or chainage > TERRAIN_GROUP_DISTANCE_M:
                continue
            chosen = group_number
            break
        if chosen is None:
            groups.append([index])
        else:
            groups[chosen].append(index)
    records = []
    for members in groups:
        representative = frame.loc[members[0]].copy()
        ids = sorted(frame.loc[members].candidate_id.astype(str))
        sources = sorted(set(frame.loc[members].candidate_source.astype(str)))
        evidence = sorted(set(frame.loc[members].physical_evidence.dropna().astype(str)))
        representative["physical_site_id"] = "v7site_" + identifier(*ids)
        representative["member_candidate_ids"] = json.dumps(ids)
        representative["member_count"] = len(ids)
        representative["support_sources"] = ",".join(sources)
        representative["member_physical_evidence"] = json.dumps(evidence)
        representative["representative_coordinate_rule"] = (
            "conditioned D8 crossing, then existing explicit intersection, then interpolated terrain low"
        )
        records.append(representative)
    return (
        gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs)
        .drop(columns=["_explicit", "_d8", "_priority"])
        .reset_index(drop=True)
    )
