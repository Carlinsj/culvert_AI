"""Versioned, label-free hydrology sites. Never changes frozen production code.

Internal road probes and physical-site proposals are deliberately different
objects. New feature semantics require an explicitly v2 experimental estimator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree

from culvert_ai.candidates import generate_road_route_candidates
from culvert_ai.crossing_geometry import (
    acute_angle_degrees,
    bilateral_continuity,
    local_line_tangent,
)
from culvert_ai.features import build_feature_table
from culvert_ai.inference import add_dem_crossing_context, add_mapped_crossing_context

VERSION = "hydrology-sites-v2"
CRS = "EPSG:32618"
THRESHOLDS = ("0p1km2", "0p5km2", "2km2", "10km2")
WARNING = "Recommendations require field verification."


def identifier(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def metric(frame):
    if frame.crs is None:
        raise ValueError("A coordinate reference system is required.")
    return frame.to_crs(CRS).copy().reset_index(drop=True)


def line_parts(geometry):
    if geometry.geom_type == "LineString":
        return [geometry]
    return [p for p in getattr(geometry, "geoms", []) if p.geom_type == "LineString"]


def raster_sample(path, xy):
    """Missing/nodata/outside stays NaN. Zero is a valid measurement."""
    xy = np.asarray(xy, float)
    with rasterio.open(path) as raster:
        if raster.crs.to_epsg() != 32618:
            raise ValueError(f"Expected metre-based EPSG:32618 raster: {path}")
        values = np.asarray(
            [v.astype(float).filled(np.nan)[0] for v in raster.sample(xy, masked=True)], dtype=float
        )
        if raster.nodata is not None:
            values[np.isclose(values, raster.nodata)] = np.nan
        inside = (
            (xy[:, 0] >= raster.bounds.left)
            & (xy[:, 0] < raster.bounds.right)
            & (xy[:, 1] > raster.bounds.bottom)
            & (xy[:, 1] <= raster.bounds.top)
        )
        values[~inside | ~np.isfinite(values)] = np.nan
    return values


def profile_values(distances, elevations, positions, center_elevations):
    """Existing 20–60m shoulder definition, independent of proposal spacing.

    Reference measurements are dense 10m DEM probes on the *complete* road part.
    Missing shoulders/DEM are unavailable, not artificial zero depressions.
    """
    distances, elevations = np.asarray(distances), np.asarray(elevations)

    def depths(queries, centers):
        result = np.full(len(queries), np.nan)
        for i, (position, center) in enumerate(zip(queries, centers)):
            left = elevations[
                (distances >= position - 60 - 1e-8) & (distances <= position - 20 + 1e-8)
            ]
            right = elevations[
                (distances >= position + 20 - 1e-8) & (distances <= position + 60 + 1e-8)
            ]
            if np.isfinite(center) and np.isfinite(left).any() and np.isfinite(right).any():
                result[i] = max(0.0, min(np.nanmax(left), np.nanmax(right)) - center)
        return result

    reference = depths(distances, elevations)
    positives = np.sort(reference[np.isfinite(reference) & (reference > 0)])
    dip = depths(np.asarray(positions), np.asarray(center_elevations))
    rank = np.where(np.isfinite(dip), 0.0, np.nan)
    if len(positives):
        use = np.isfinite(dip) & (dip > 0)
        rank[use] = np.searchsorted(positives, dip[use], side="right") / len(positives)
    score = 0.70 * (1 - np.exp(-dip / 0.75)) + 0.30 * rank
    score[np.isfinite(dip) & (dip < 0.15)] = 0.0
    return dip, rank, score


def repair_profiles(points, roads, dem_path):
    result = points.copy().reset_index(drop=True)
    lookup = {str(row.id): line_parts(row.geometry) for row in roads.itertuples()}
    output = np.full((len(result), 3), np.nan)
    for road_id, indices in result.groupby(result.road_id.astype(str), sort=True).groups.items():
        parts = lookup.get(road_id, [])
        if not parts:
            continue
        groups = {}
        for i in indices:
            part = min(
                range(len(parts)), key=lambda j: (parts[j].distance(result.geometry.iloc[i]), j)
            )
            groups.setdefault(part, []).append(i)
        for part, use in groups.items():
            line = parts[part]
            distances = np.arange(5.0, line.length, 10.0)
            if not len(distances):
                continue
            dense = [line.interpolate(float(d)) for d in distances]
            positions = [line.project(result.geometry.iloc[i]) for i in use]
            centers = [line.interpolate(float(d)) for d in positions]
            z = raster_sample(dem_path, [(p.x, p.y) for p in dense])
            center_z = raster_sample(dem_path, [(p.x, p.y) for p in centers])
            output[use] = np.column_stack(profile_values(distances, z, positions, center_z))
    for j, name in enumerate(
        ("route_elevation_dip_60m", "route_elevation_dip_rank_pct", "route_low_point_score")
    ):
        result[name] = output[:, j]
    result["road_profile_available"] = np.isfinite(output[:, 0])
    return result


def search_candidates(roads, interval_m=10.0):
    roads = metric(roads)
    if "route_id" not in roads:
        roads["route_id"] = roads.id.astype(str)
    result = generate_road_route_candidates(
        roads,
        interval_m=interval_m,
        lateral_offsets_m=(0.0,),
        include_all_roads=True,
        route_id_column="route_id",
    )
    result["candidate_id"] = [
        "probe_" + identifier(r.road_id, r.route_part_index, f"{r.route_sample_distance_m:.3f}")
        for r in result.itertuples()
    ]
    result["route_id"] = result.matched_route
    result["proposal_kind"] = "internal_road_probe"
    return result


def intersection_points(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    return [p for g in getattr(geometry, "geoms", []) for p in intersection_points(g)]


def detect_events(roads, mapped_streams, drainage_layers, *, minimum_angle=30.0):
    """Actual transverse/bilateral intersections, not river proximity scores.

    No labels, fitted scores, route quotas or inventory distances are accepted.
    All rejected contacts are retained as diagnostic evidence.
    """
    roads, mapped = metric(roads), metric(mapped_streams)
    networks = {key: metric(value) for key, value in drainage_layers.items()}
    fine = networks.get("0p1km2")
    records, rejected = [], []
    layers = [("mapped", mapped), *[("dem_" + key, layer) for key, layer in networks.items()]]
    for road in roads.itertuples():
        for part_index, line in enumerate(line_parts(road.geometry)):
            for source, layer in layers:
                for position in sorted(layer.sindex.query(line, predicate="intersects")):
                    water = layer.geometry.iloc[position]
                    points = intersection_points(line.intersection(water))
                    if not points:
                        rejected.append(
                            dict(
                                road_id=str(road.id),
                                source=source,
                                water_id=int(position),
                                reason="overlapping_or_parallel_geometry",
                            )
                        )
                    for point in points:
                        _, tangent = local_line_tangent(line, point)
                        water_parts = line_parts(water)
                        if not water_parts:
                            continue
                        water_part = min(water_parts, key=lambda p: p.distance(point))
                        _, water_tangent = local_line_tangent(water_part, point)
                        angle = acute_angle_degrees(tangent, water_tangent)
                        bilateral = bilateral_continuity(
                            water, point, tangent, 60.0, side_tolerance_m=2.0
                        )
                        if not np.isfinite(angle) or angle < minimum_angle or bilateral != 1:
                            rejected.append(
                                dict(
                                    road_id=str(road.id),
                                    source=source,
                                    water_id=int(position),
                                    reason="nontransverse_or_one_sided_contact",
                                    angle=angle,
                                )
                            )
                            continue
                        channel = source + ":" + str(position)
                        if fine is not None and len(fine):
                            nearby = sorted(
                                fine.sindex.query(point.buffer(10), predicate="intersects")
                            )
                            if nearby:
                                closest = min(
                                    nearby, key=lambda j: (fine.geometry.iloc[j].distance(point), j)
                                )
                                channel = "dem_fine:" + str(closest)
                        records.append(
                            dict(
                                candidate_id="event_"
                                + identifier(
                                    road.id,
                                    part_index,
                                    source,
                                    position,
                                    f"{point.x:.4f}",
                                    f"{point.y:.4f}",
                                ),
                                road_id=str(road.id),
                                road_name=getattr(road, "name", str(road.id)),
                                route_id=getattr(road, "route_id", str(road.id)),
                                source="route_interval_sample",
                                proposal_kind="mapped_crossing"
                                if source == "mapped"
                                else "dem_crossing",
                                evidence_source=source,
                                water_id=str(position),
                                channel_key=channel,
                                mapped_water_id=str(position) if source == "mapped" else "",
                                road_part_index=part_index,
                                road_chainage_m=float(line.project(point)),
                                crossing_angle_degrees=float(angle),
                                bilateral_drainage=True,
                                bridge_context=str(getattr(road, "road_bridge", "")).lower()
                                in {"true", "yes", "1"},
                                geometry=point,
                            )
                        )
    columns = [
        "candidate_id",
        "road_id",
        "road_name",
        "route_id",
        "source",
        "proposal_kind",
        "evidence_source",
        "water_id",
        "channel_key",
        "mapped_water_id",
        "road_part_index",
        "road_chainage_m",
        "crossing_angle_degrees",
        "bilateral_drainage",
        "bridge_context",
        "geometry",
    ]
    events = gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=CRS)
    return events.drop_duplicates("candidate_id").reset_index(drop=True), pd.DataFrame(rejected)


def physical_measurements(points, roads, streams, hydro):
    """Reuse GIS measurements, but repair profiles and actually measure edge risk."""
    hydro = Path(hydro)
    points, roads, streams = metric(points), metric(roads), metric(streams)
    if not points.candidate_id.is_unique:
        raise ValueError("Candidate IDs must be unique.")
    result = build_feature_table(
        points,
        roads=roads,
        streams=streams,
        dem_path=hydro / "dem_projected.tif",
        density_radius_m=75.0,
        density_radii_m=(100.0, 250.0, 500.0),
    )
    result = add_mapped_crossing_context(result, roads, streams)
    crossing_layers = {
        key: gpd.read_parquet(hydro / f"road_drainage_crossings_{key}.parquet")
        for key in THRESHOLDS
    }
    if any(layer.empty for layer in crossing_layers.values()):
        raise ValueError("Hydrology layers must exist; missing crossings must not become zero.")
    result = add_dem_crossing_context(result, crossing_layers)
    xy = np.c_[result.geometry.x, result.geometry.y]
    for filename, column in (
        ("flow_accumulation_cells.tif", "flow_accumulation_cells"),
        ("upstream_area_m2.tif", "upstream_area_m2"),
        ("edge_originating_flowpaths.tif", "candidate_edge_risk"),
    ):
        result[column] = raster_sample(hydro / filename, xy)
    risk = result.candidate_edge_risk.to_numpy(float)
    for key, crossings in crossing_layers.items():
        crossings = metric(crossings)
        nearest = cKDTree(np.c_[crossings.geometry.x, crossings.geometry.y]).query(xy)[1]
        risk = np.maximum(risk, pd.to_numeric(crossings.iloc[nearest].edge_risk).to_numpy(float))
        drainage = metric(gpd.read_parquet(hydro / f"drainage_{key}.parquet"))
        (left, right), distance = drainage.sindex.nearest(
            result.geometry, return_all=False, return_distance=True
        )
        nearest_points = shapely.get_point(
            shapely.shortest_line(
                result.geometry.iloc[left].array, drainage.geometry.iloc[right].array
            ),
            1,
        )
        edge = raster_sample(
            hydro / "edge_originating_flowpaths.tif",
            np.c_[shapely.get_x(nearest_points), shapely.get_y(nearest_points)],
        )
        reordered = np.full(len(result), np.nan)
        reordered[left] = edge
        risk = np.maximum(risk, reordered)
    result["hydrology_edge_risk"] = risk
    result = repair_profiles(result, roads, hydro / "dem_projected.tif")
    result["dem_data_available"] = result.elevation_m.notna()
    return result


def fit_feature_contract(training_measurements, feature_columns):
    references = {}
    columns = []
    for window in (3, 9, 15, 31):
        columns.extend(
            [
                f"{prefix}_{window}x{window}{suffix}"
                for prefix, suffix in (
                    ("valley_depth", "_m"),
                    ("negative_tpi", "_m"),
                    ("topographic_wetness_proxy", ""),
                    ("terrain_break_score_proxy", ""),
                    ("elevation_relief", "_m"),
                    ("terrain_roughness", "_m"),
                )
            ]
        )
    columns += [
        "road_density_m_per_sqkm",
        *[f"road_density_{r}m_m_per_sqkm" for r in (100, 250, 500)],
    ]
    for column in columns:
        values = (
            pd.to_numeric(training_measurements[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        references[column] = {
            "q05": float(values.quantile(0.05)) if len(values) else None,
            "q95": float(values.quantile(0.95)) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else None,
        }
    return dict(
        version=VERSION,
        feature_columns=list(feature_columns),
        references=references,
        dem_crs=CRS,
        dem_resolution_m=10.0,
        profile_sampling_m=10.0,
        shoulders_m=[20, 60],
        missing_policy="NaN until training-only median imputation; availability retained",
        training_only_reference=True,
    )


def apply_feature_contract(measurements, contract, normalization_context=None):
    if contract.get("version") != VERSION:
        raise ValueError("Repaired features require a v2 experimental contract/model.")
    result = measurements.copy()
    context = measurements if normalization_context is None else normalization_context

    def normalize(column):
        values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        reference = contract["references"][column]
        low, high = reference["q05"], reference["q95"]
        if low is None or high is None:
            return values * np.nan
        if high <= low:
            return values.where(values.isna(), 0.0)
        return ((values - low) / (high - low)).clip(0, 1)

    valleys = []
    terrains = []
    for w in (3, 9, 15, 31):
        depth, tpi, wet = (
            f"valley_depth_{w}x{w}_m",
            f"negative_tpi_{w}x{w}_m",
            f"topographic_wetness_proxy_{w}x{w}",
        )
        low = normalize(depth) / (1 + result.slope_degrees.clip(lower=0))
        result[f"low_slope_valley_score_{w}x{w}"] = low
        # Require every component, rather than silently turning unavailable
        # measurements into a plausible valid zero composite.
        valley = pd.concat([normalize(depth), normalize(tpi), normalize(wet), low], axis=1).mean(
            axis=1, skipna=False
        )
        terrain = pd.concat(
            [
                normalize(f"{prefix}_{w}x{w}{suffix}")
                for prefix, suffix in (
                    ("terrain_break_score_proxy", ""),
                    ("elevation_relief", "_m"),
                    ("terrain_roughness", "_m"),
                )
            ],
            axis=1,
        ).mean(axis=1, skipna=False)
        result[f"dem_valley_position_score_{w}x{w}"] = valley
        result[f"dem_terrain_break_score_{w}x{w}"] = terrain
        valleys.append(valley)
        terrains.append(terrain)
    result["dem_culvert_terrain_score"] = 0.65 * pd.concat(valleys, axis=1).mean(
        axis=1, skipna=False
    ) + 0.35 * pd.concat(terrains, axis=1).mean(axis=1, skipna=False)
    for column in (
        "road_density_m_per_sqkm",
        *[f"road_density_{r}m_m_per_sqkm" for r in (100, 250, 500)],
    ):
        values = pd.to_numeric(result[column], errors="coerce")
        output = pd.Series(np.nan, index=result.index)
        for _, indices in result.groupby("route_id", sort=True).groups.items():
            group = values.loc[indices]
            route = result.loc[indices[0], "route_id"]
            context_values = pd.to_numeric(
                context.loc[context.route_id.eq(route), column], errors="coerce"
            )
            center = context_values.median()
            scale = context_values.quantile(0.75) - context_values.quantile(0.25)
            if not np.isfinite(scale) or scale < 1e-9:
                center, scale = (
                    contract["references"][column]["median"],
                    contract["references"][column]["iqr"],
                )
            if center is not None and scale is not None:
                output.loc[indices] = (
                    (group - center) / scale if scale > 1e-9 else group.where(group.isna(), 0.0)
                )
        result[column + "__robust_route_normalized"] = output
    result["feature_contract"] = VERSION
    return result


def consolidate_events(events, scores=None, radius_m=20.0):
    """Bounded channel/road/chainage grouping; never move a representative.

    Separate water bodies, road parts, parallel roads and hairpin recrossings
    remain separate even if Euclidean distance is small. No single-link chains.
    """
    if events.empty:
        result = events.copy()
        result["physical_site_id"] = pd.Series(dtype=str)
        result["member_candidate_ids"] = pd.Series(dtype=str)
        result["member_count"] = pd.Series(dtype=int)
        return result
    frame = events.copy()
    frame["model_score"] = np.nan if scores is None else np.asarray(scores)
    frame["evidence_order"] = frame.proposal_kind.map(
        {"mapped_crossing": 0, "dem_crossing": 1, "terrain_convergence": 2}
    ).fillna(3)
    ordered = frame.sort_values(["evidence_order", "candidate_id"], kind="stable")
    selected = []
    for _, group in ordered.groupby(["road_id", "road_part_index", "channel_key"], sort=True):
        remaining = list(group.index)
        while remaining:
            representative = remaining[0]
            row = frame.loc[representative]
            members = []
            mapped_ids = set()
            for i in remaining:
                other = frame.loc[i]
                if (
                    row.geometry.distance(other.geometry) > radius_m
                    or abs(row.road_chainage_m - other.road_chainage_m) > radius_m
                ):
                    continue
                mid = str(other.mapped_water_id)
                if mid and mapped_ids and mid not in mapped_ids:
                    continue
                if mid:
                    mapped_ids.add(mid)
                members.append(i)
            record = row.copy()
            record["physical_site_id"] = "hydro_" + identifier(
                row.road_id, row.road_part_index, row.channel_key, row.candidate_id
            )
            record["member_candidate_ids"] = ",".join(sorted(frame.loc[members].candidate_id))
            record["member_count"] = len(members)
            record["support_sources"] = ",".join(sorted(set(frame.loc[members].evidence_source)))
            record["maximum_member_model_score"] = frame.loc[members].model_score.max()
            selected.append(record)
            remaining = [i for i in remaining if i not in members]
    return (
        gpd.GeoDataFrame(selected, geometry="geometry", crs=events.crs)
        .drop(columns="evidence_order")
        .reset_index(drop=True)
    )


def rank_physical_sites(sites):
    result = sites.sort_values(
        ["model_score", "candidate_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["field_verification_notice"] = WARNING
    result["score_interpretation"] = "Uncalibrated relative ranking score, not a probability."
    return result
