"""Label-free location selection, isolated from production and model features.

Flags here are proposal metadata, NEVER new RF inputs. The fixed60m comparison
window and0.15m profile floor come from the existing road-profile definition,
not evaluation inventory. No road quotas, interpolation, jitter or backfilling.
"""

import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import substring

from staging.hybrid_sites import cluster_sites, rank_sites, validate
from staging.hydrology_sites import line_parts

VERSION = "feature-localization-v1"
WINDOW_M = 60.0
PROFILE_FLOOR_M = 0.15
EVIDENCE_COLUMNS = [
    "distance_to_nearest_stream_m",
    "route_elevation_dip_60m",
    "route_low_point_score",
    "road_profile_available",
    "flow_accumulation_cells",
    "upstream_area_m2",
    "slope_degrees",
    "hydrology_edge_risk",
    "negative_tpi_9x9_m",
    "valley_depth_9x9_m",
]


def legacy_terrain_at(points, original_dem):
    """Recover the original raster/window semantics ONLY, not a complete RF input.

    Explicitly leaves batch-normalized composites, road profiles, density and
    crossing context unresolved. Never passes this partial table to frozenRF.
    """
    from culvert_ai.features import add_dem_hydrology_proxies, add_raster_samples

    fields = [
        "slope_degrees",
        *[
            f"{name}_{w}x{w}{suffix}"
            for w in (9, 31)
            for name, suffix in [
                ("elevation_relief", "_m"),
                ("terrain_roughness", "_m"),
                ("topographic_position", "_m"),
                ("valley_depth", "_m"),
                ("negative_tpi", "_m"),
                ("topographic_wetness_proxy", ""),
                ("terrain_break_score_proxy", ""),
            ]
        ],
    ]
    measured = add_dem_hydrology_proxies(
        add_raster_samples(points[["geometry"]], original_dem, "dem")
    )
    return measured[[*fields, "geometry"]]


def local_peaks(anchors, column, window_m=WINDOW_M):
    """True two-sided local extrema along the SAME road/part, bounded by chainage.

    Require a complete window and finite measurements, a lower value on BOTH
    sides. Equal-height neighbouring plateau candidates use stable ID ties.
    A constant field, isolated anchor or request-edge anchor produces no peak.
    """
    found = pd.Series(False, index=anchors.index)
    for _, group in anchors.groupby(["road_id", "road_part_index"], sort=True):
        group = group.sort_values(["road_chainage_m", "candidate_id"], kind="stable")
        pos = group.road_chainage_m.to_numpy(float)
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        ids = group.candidate_id.astype(str).to_numpy()
        for i, (p, v) in enumerate(zip(pos, values)):
            if not np.isfinite(v) or p - pos[0] < window_m - 1e-6 or pos[-1] - p < window_m - 1e-6:
                continue
            lo, hi = np.searchsorted(pos, [p - window_m - 1e-6, p + window_m + 1e-6])
            use = values[lo:hi]
            if not np.isfinite(use).all() or v < use.max():
                continue
            if not (np.any(values[lo:i] < v) and np.any(values[i + 1 : hi] < v)):
                continue
            equal = ids[lo:hi][use == v]
            if ids[i] == min(equal):
                found.loc[group.index[i]] = True
    return found


def proposals(points):
    """Choose evidence locations only; accept no inventory or labels argument."""
    validate(points)
    frame = points.copy()
    anchors = frame.loc[frame.proposal_kind.eq("internal_road_probe")]
    frame["profile_local_peak"] = False
    frame["flow_valley_local_peak"] = False
    frame["ml_local_peak"] = False
    if len(anchors):
        profile = local_peaks(anchors, "route_elevation_dip_60m")
        profile &= anchors.route_elevation_dip_60m.ge(PROFILE_FLOOR_M)
        profile &= anchors.road_profile_available.fillna(False) & anchors.route_low_point_score.gt(
            0
        )
        flow = local_peaks(anchors, "upstream_area_m2")
        flow &= anchors.upstream_area_m2.gt(0) & anchors.negative_tpi_9x9_m.gt(0)
        frame.loc[anchors.index, "profile_local_peak"] = profile
        frame.loc[anchors.index, "flow_valley_local_peak"] = flow
        if "model_score" in anchors:
            frame.loc[anchors.index, "ml_local_peak"] = local_peaks(anchors, "model_score")
    frame["physical_crossing"] = frame.proposal_kind.isin(["mapped_crossing", "dem_crossing"])
    # Strict detect_events crossings are already transverse and two-sided. Legacy
    # mapped nearest approaches are NOT silently admitted as physical crossings.
    frame["originated_at_sampling_anchor"] = frame.proposal_kind.eq("internal_road_probe")
    frame["evidence_admitted"] = (
        frame.physical_crossing | frame.profile_local_peak | frame.flow_valley_local_peak
    )
    frame["localization_policy"] = VERSION
    frame["localization_evidence"] = [
        json.dumps(
            [
                name
                for name in (
                    "physical_crossing",
                    "profile_local_peak",
                    "flow_valley_local_peak",
                    "ml_local_peak",
                )
                if bool(row[name])
            ]
        )
        for _, row in frame.iterrows()
    ]
    return frame


def ranked(points, policy="dynamic_route_z"):
    if len(points):
        result = rank_sites(points, policy)
        lookup = points.set_index("candidate_id").localization_evidence.to_dict()
        result["member_localization_evidence"] = [
            json.dumps(
                sorted(
                    set(
                        item
                        for member in json.loads(ids)
                        for item in json.loads(lookup.get(member, "[]"))
                    )
                )
            )
            for ids in result.member_candidate_ids
        ]
        return result
    result = cluster_sites(points)
    for col in ("global_rank", "priority"):
        result[col] = pd.Series(dtype=float)
    result["ranking_policy"] = policy
    return result


def spacing_links(points, roads, kind):
    """Consecutive chainage, never geographic nearest-neighbour sequencing."""
    lookup = {str(r.id): line_parts(r.geometry) for r in roads.itertuples()}
    rows = []
    for (road, part), group in points.groupby(["road_id", "road_part_index"], sort=True):
        group = group.loc[np.isfinite(group.road_chainage_m)].sort_values(
            ["road_chainage_m", "candidate_id"]
        )
        parts = lookup.get(str(road), [])
        if int(part) >= len(parts):
            continue
        line = parts[int(part)]
        for a, b in zip(list(group.itertuples()), list(group.itertuples())[1:]):
            if b.road_chainage_m <= a.road_chainage_m:
                continue
            geom = substring(line, a.road_chainage_m, b.road_chainage_m)
            if geom.geom_type != "LineString":
                continue
            rows.append(
                dict(
                    road_id=str(road),
                    road_part_index=int(part),
                    from_id=a.candidate_id,
                    to_id=b.candidate_id,
                    distance_m=float(b.road_chainage_m - a.road_chainage_m),
                    measurement="Along-road "
                    + kind
                    + " spacing;not distance between confirmed culverts",
                    geometry=geom,
                )
            )
    return gpd.GeoDataFrame(rows if rows else {"geometry": []}, geometry="geometry", crs=32618)


def check_model(fitted, frozen_columns):
    """Fail closed on frozenv1, wrong order, missing preprocessing or estimator."""
    if (
        not fitted.get("experimental")
        or fitted.get("contract", {}).get("version") != "hydrology-sites-v2"
    ):
        raise ValueError("Frozen RF is not compatible with the v2 feature representation.")
    columns = fitted.get("feature_columns")
    if (
        columns != frozen_columns
        or len(columns) != 48
        or columns != fitted["contract"].get("feature_columns")
    ):
        raise ValueError("The exact ordered48-field schema is required.")
    if fitted["model"].n_estimators != 800 or not set(columns).issubset(fitted["medians"].index):
        raise ValueError("Expected fixed800-tree v2 RF and saved training medians.")
    return columns
