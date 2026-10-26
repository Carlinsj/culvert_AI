"""Label-free physical-site consolidation for the final v6 research queue.

The policy never reads evaluation coordinates. It keeps the frozen baseline
population, admits independently generated terrain extrema and explicit
drainage crossings, and collapses only unresolved cross-generator duplicates.
"""

from __future__ import annotations

import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


VERSION = "compact-physical-sites-v6-experimental"
DEM_RESOLUTION_M = 10.0
EXACT_DUPLICATE_M = 2.0
EXPLICIT_CROSSING_KINDS = {
    "mapped_crossing",
    "dem_crossing",
    "small_dem_crossing",
    "terrain_convergence",
}


def _stable_id(member_ids: list[str]) -> str:
    payload = "|".join(sorted(map(str, member_ids))).encode()
    return "compact_" + hashlib.sha256(payload).hexdigest()[:24]


def _first_text(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        result = result.mask(result.eq("") & values.ne(""), values)
    return result


def eligible_physical_union(
    candidates: gpd.GeoDataFrame, baseline_candidate_ids: set[str]
) -> gpd.GeoDataFrame:
    """Keep baseline anchors, multiscale extrema, and explicit crossings only."""
    ids = candidates.candidate_id.astype(str)
    source = candidates.get("candidate_source", pd.Series("", index=candidates.index)).fillna("")
    kind = candidates.get("proposal_kind", pd.Series("", index=candidates.index)).fillna("")
    baseline = ids.isin(baseline_candidate_ids)
    terrain = source.eq("multiscale_terrain_extremum")
    crossing = kind.isin(EXPLICIT_CROSSING_KINDS)
    result = candidates.loc[baseline | terrain | crossing].copy()
    result["eligibility_reason"] = np.select(
        [baseline.loc[result.index], terrain.loc[result.index], crossing.loc[result.index]],
        ["preserved_baseline", "multiscale_terrain_extremum", "explicit_physical_crossing"],
        default="excluded",
    )
    return result


def compact_cross_generator_sites(
    candidates: gpd.GeoDataFrame,
    baseline_ranks: dict[str, int],
    *,
    dem_resolution_m: float = DEM_RESOLUTION_M,
    exact_duplicate_m: float = EXACT_DUPLICATE_M,
) -> gpd.GeoDataFrame:
    """Collapse compatible cross-generator proposals without moving a site.

    Proposals must share a road and road part. Different non-empty channel
    identities never merge. Same-generator sites merge only when they are
    effectively coincident (<=2 m); cross-generator proposals may merge within
    one 10 m DEM cell. Complete-span chainage and Euclidean limits prevent
    single-link chains and hairpin-road merges.
    """
    if candidates.empty:
        return candidates.copy()
    if candidates.crs is None or candidates.crs.to_epsg() != 32618:
        raise ValueError("Physical consolidation requires EPSG:32618 metre coordinates")
    required = {"candidate_id", "road_chainage_m", "geometry", "existing_score"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing consolidation fields: {sorted(missing)}")

    frame = candidates.copy()
    frame["_road"] = _first_text(frame, ("road_id", "route_id", "road_name"))
    frame["_part"] = pd.to_numeric(
        frame.get("road_part_index", pd.Series(-1, index=frame.index)), errors="coerce"
    ).fillna(-1).astype(int)
    frame["_chainage"] = pd.to_numeric(frame.road_chainage_m, errors="coerce")
    channel = _first_text(frame, ("channel_key",))
    water = _first_text(frame, ("mapped_water_id",))
    frame["_identity"] = np.where(channel.ne(""), "channel:" + channel, np.where(water.ne(""), "water:" + water, ""))
    source = frame.get("candidate_source", pd.Series("", index=frame.index)).fillna("")
    frame["_source_family"] = np.where(
        source.eq("multiscale_terrain_extremum"), "terrain_extremum", "existing_anchor"
    )
    frame["_baseline_rank"] = frame.candidate_id.astype(str).map(baseline_ranks)
    frame["_evidence_count"] = pd.to_numeric(
        frame.get("physical_evidence_count", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    frame["_has_identity"] = frame._identity.ne("").astype(int)
    frame = frame.sort_values(
        ["_road", "_part", "_chainage", "candidate_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    group_ids: list[int] = []
    group_id = 0
    key: tuple[str, int] | None = None
    first_chainage = np.nan
    first_geometry = None
    source_families: set[str] = set()
    identities: set[str] = set()
    for road, part, chainage, geometry, family, identity in zip(
        frame._road,
        frame._part,
        frame._chainage,
        frame.geometry,
        frame._source_family,
        frame._identity,
    ):
        current_key = (str(road), int(part))
        chainage_distance = (
            float(chainage) - float(first_chainage)
            if np.isfinite(chainage) and np.isfinite(first_chainage)
            else np.inf
        )
        geometry_distance = (
            float(geometry.distance(first_geometry)) if first_geometry is not None else np.inf
        )
        identity_conflict = bool(identity and identities and identity not in identities)
        same_source = family in source_families
        allowed_span = exact_duplicate_m if same_source else dem_resolution_m
        starts_group = (
            current_key != key
            or chainage_distance > allowed_span + 1e-8
            or geometry_distance > allowed_span + 1e-8
            or identity_conflict
        )
        if starts_group:
            group_id += 1
            key = current_key
            first_chainage = float(chainage) if np.isfinite(chainage) else np.nan
            first_geometry = geometry
            source_families = set()
            identities = set()
        source_families.add(str(family))
        if identity:
            identities.add(str(identity))
        group_ids.append(group_id)
    frame["_compact_group"] = group_ids

    members = frame.groupby("_compact_group", sort=True).candidate_id.apply(
        lambda values: [str(value) for value in values]
    )
    aggregates = frame.groupby("_compact_group", sort=True).agg(
        source_site_count=("candidate_id", "size"),
        group_baseline_rank=("_baseline_rank", "min"),
        group_max_existing_score=("existing_score", "max"),
        group_min_v3_rank=("global_rank", "min"),
        source_family_count=("_source_family", "nunique"),
    )
    representatives = (
        frame.sort_values(
            ["_compact_group", "_baseline_rank", "_has_identity", "_evidence_count", "existing_score", "candidate_id"],
            ascending=[True, True, False, False, False, True],
            kind="stable",
            na_position="last",
        )
        .drop_duplicates("_compact_group")
        .copy()
    )
    representatives = representatives.join(aggregates, on="_compact_group")
    representatives["member_candidate_ids"] = representatives._compact_group.map(
        lambda group: json.dumps(members.loc[group])
    )
    representatives["physical_site_id"] = representatives._compact_group.map(
        lambda group: _stable_id(members.loc[group])
    )
    prior_members = pd.to_numeric(
        representatives.get("member_count", pd.Series(1, index=representatives.index)), errors="coerce"
    ).fillna(1).astype(int)
    representatives["member_count"] = np.maximum(
        prior_members, representatives.source_site_count.astype(int)
    )
    representatives["grouping_policy"] = (
        "same road/part; <=10m cross-generator or <=2m same-generator; "
        "different channel identities preserved; complete-span and Euclidean bounded"
    )
    representatives["representative_selection"] = (
        "existing member only: earliest baseline rank, then channel identity, "
        "physical evidence count, frozen-RF score, stable candidate ID"
    )
    representatives["model_version"] = VERSION
    representatives["model_score"] = representatives.group_max_existing_score
    representatives["_baseline_group"] = representatives.group_baseline_rank.notna()
    representatives = representatives.sort_values(
        ["_baseline_group", "group_baseline_rank", "group_max_existing_score", "group_min_v3_rank", "candidate_id"],
        ascending=[False, True, False, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    representatives["global_rank"] = np.arange(1, len(representatives) + 1)
    drop = [column for column in representatives.columns if column.startswith("_")]
    return gpd.GeoDataFrame(
        representatives.drop(columns=drop), geometry="geometry", crs=candidates.crs
    )
