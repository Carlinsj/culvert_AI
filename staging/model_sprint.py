"""Experimental, label-safe helpers for the 2026-10-17 model sprint.

The module deliberately has no workbook/inventory dependency. Candidate admission
uses only measured GIS/DEM evidence. Evaluation labels belong in the runner.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d


VERSION = "culvert-retrieval-v3-experimental"
PROFILE_HALF_WINDOW_M = 60.0
EXPECTED_SAMPLE_SPACING_M = 10.0
GROUP_SPAN_M = 20.0

PHYSICAL_SIGNALS = (
    "route_low_point_score",
    "flow_accumulation_cells",
    "negative_tpi_3x3_m",
    "negative_tpi_9x9_m",
    "negative_tpi_15x15_m",
    "negative_tpi_31x31_m",
    "terrain_break_score_proxy_3x3",
    "terrain_break_score_proxy_9x9",
    "terrain_break_score_proxy_15x15",
    "terrain_break_score_proxy_31x31",
)

GIS_FEATURES = (
    "distance_to_nearest_road_m",
    "distance_to_nearest_stream_m",
    "stream_density_100m_m_per_sqkm",
    "stream_density_250m_m_per_sqkm",
    "stream_density_500m_m_per_sqkm",
    "road_density_100m_m_per_sqkm",
    "road_density_250m_m_per_sqkm",
    "road_density_500m_m_per_sqkm",
    "local_road_stream_separation_m",
    "local_road_stream_angle_degrees",
    "distance_to_nearest_mapped_crossing_m",
    "mapped_crossing_count_250m",
    "mapped_crossing_count_500m",
    "crossing_angle_perpendicularity",
)
HYDRO_FEATURES = (
    "flow_accumulation_cells",
    "upstream_area_m2",
    "hydrology_edge_risk",
    "distance_to_dem_road_drainage_crossing_0p1km2_m",
    "nearest_crossing_upstream_area_0p1km2_m2",
    "nearest_crossing_angle_0p1km2_degrees",
    "distance_to_dem_road_drainage_crossing_0p5km2_m",
    "nearest_crossing_upstream_area_0p5km2_m2",
    "nearest_crossing_angle_0p5km2_degrees",
    "distance_to_dem_road_drainage_crossing_2km2_m",
    "nearest_crossing_upstream_area_2km2_m2",
    "nearest_crossing_angle_2km2_degrees",
    "distance_to_dem_road_drainage_crossing_10km2_m",
    "nearest_crossing_upstream_area_10km2_m2",
    "nearest_crossing_angle_10km2_degrees",
)
TERRAIN_FEATURES = (
    "elevation_m",
    "slope_degrees",
    "elevation_relief_3x3_m",
    "elevation_relief_9x9_m",
    "elevation_relief_15x15_m",
    "elevation_relief_31x31_m",
    "negative_tpi_3x3_m",
    "negative_tpi_9x9_m",
    "negative_tpi_15x15_m",
    "negative_tpi_31x31_m",
    "topographic_wetness_proxy_3x3",
    "topographic_wetness_proxy_9x9",
    "topographic_wetness_proxy_15x15",
    "topographic_wetness_proxy_31x31",
    "terrain_break_score_proxy_3x3",
    "terrain_break_score_proxy_9x9",
    "terrain_break_score_proxy_15x15",
    "terrain_break_score_proxy_31x31",
    "dem_valley_position_score",
    "dem_terrain_break_score",
    "dem_culvert_terrain_score",
)
ROAD_PROFILE_FEATURES = (
    "route_elevation_dip_60m",
    "route_elevation_dip_rank_pct",
    "route_low_point_score",
    "road_profile_available",
    "candidate_edge_risk",
    "dem_data_available",
)
FEATURE_FAMILIES = {
    "gis": GIS_FEATURES,
    "gis_hydro": GIS_FEATURES + HYDRO_FEATURES,
    "gis_hydro_terrain": GIS_FEATURES + HYDRO_FEATURES + TERRAIN_FEATURES,
    "all_physical": GIS_FEATURES + HYDRO_FEATURES + TERRAIN_FEATURES + ROAD_PROFILE_FEATURES,
}


def _stable_id(prefix: str, values: list[str]) -> str:
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _peak_flags(values: np.ndarray, size: int) -> np.ndarray:
    """Centered local maxima; NaN never becomes zero and plateaus emit once."""
    finite = np.isfinite(values)
    maximum = maximum_filter1d(
        np.where(finite, values, -np.inf), size=size, mode="constant", cval=-np.inf
    )
    result = finite & (values > 0) & np.isclose(values, maximum, rtol=0, atol=1e-12)
    # Input is sorted by chainage and candidate ID, giving deterministic plateau ties.
    result &= ~np.r_[False, np.isclose(values[1:], values[:-1], rtol=0, atol=1e-12)]
    return result


def multiscale_extrema(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create label-free terrain hypotheses at measured 10 m road anchors.

    The 120 m centered window is a fixed physical scale (60 m on either side),
    not a fitted threshold. All coordinates are existing measurement anchors.
    """
    missing = set(PHYSICAL_SIGNALS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing physical inputs: {sorted(missing)}")
    probes = raw.loc[raw.proposal_kind.eq("internal_road_probe")].copy().reset_index(drop=True)
    flags = np.zeros((len(probes), len(PHYSICAL_SIGNALS)), dtype=bool)
    size = int(round(2 * PROFILE_HALF_WINDOW_M / EXPECTED_SAMPLE_SPACING_M)) + 1
    for indices in probes.groupby(["road_id", "road_part_index"], sort=False).groups.values():
        group = probes.loc[list(indices)].sort_values(
            ["road_chainage_m", "candidate_id"], kind="stable"
        )
        group_positions = group.index.to_numpy()
        signal_values = group.loc[:, PHYSICAL_SIGNALS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        chainage = pd.to_numeric(group.road_chainage_m, errors="coerce").to_numpy(float)
        # Query-cell clipping can leave large gaps within one road identity. Never
        # treat values across those gaps as physical neighbours.
        breaks = np.flatnonzero(
            ~np.isfinite(np.diff(chainage))
            | (np.diff(chainage) > EXPECTED_SAMPLE_SPACING_M + 0.25)
        ) + 1
        for run in np.split(np.arange(len(group)), breaks):
            positions = group_positions[run]
            for column_index in range(len(PHYSICAL_SIGNALS)):
                flags[positions, column_index] = _peak_flags(
                    signal_values[run, column_index], size
                )
    # Flow alone is not evidence: require a concave local valley.
    flags[:, 1] &= pd.to_numeric(
        probes["negative_tpi_9x9_m"], errors="coerce"
    ).fillna(0).to_numpy(float) > 0
    admitted = flags.any(axis=1)
    result = probes.loc[admitted].copy()
    selected = flags[admitted]
    result["physical_evidence_count"] = selected.sum(axis=1)
    result["physical_evidence_sources"] = [
        json.dumps([name for name, keep in zip(PHYSICAL_SIGNALS, row) if keep])
        for row in selected
    ]
    result["physical_score"] = 0.0
    for column in PHYSICAL_SIGNALS:
        result["physical_score"] += (
            result.groupby(["road_id", "road_part_index"])[column]
            .rank(pct=True)
            .fillna(0)
        )
    result["candidate_source"] = "multiscale_terrain_extremum"
    result["quality_missing_count"] = result[list(PHYSICAL_SIGNALS)].isna().sum(axis=1)
    result["selection_policy"] = VERSION
    return result


def represented_candidate_ids(hybrid: pd.DataFrame) -> set[str]:
    represented = set(hybrid.candidate_id.astype(str))
    if "member_candidate_ids" not in hybrid:
        return represented
    for value in hybrid.member_candidate_ids.dropna():
        try:
            represented.update(map(str, json.loads(value)))
        except (TypeError, json.JSONDecodeError):
            continue
    return represented


def group_extrema(extrema: gpd.GeoDataFrame, span_m: float = GROUP_SPAN_M) -> gpd.GeoDataFrame:
    """Complete-span grouping on the same road part; never proximity-only across roads."""
    if extrema.empty:
        return extrema.copy()
    ordered = extrema.sort_values(
        ["road_id", "road_part_index", "road_chainage_m", "candidate_id"], kind="stable"
    ).copy()
    groups: list[int] = []
    group_id = 0
    current_key = None
    first_chainage = 0.0
    for road_id, part, chainage in zip(
        ordered.road_id.astype(str), ordered.road_part_index, ordered.road_chainage_m
    ):
        key = (road_id, part)
        if key != current_key or float(chainage) - first_chainage > span_m + 1e-8:
            group_id += 1
            current_key = key
            first_chainage = float(chainage)
        groups.append(group_id)
    ordered["_physical_group"] = groups
    members = ordered.groupby("_physical_group", sort=True).candidate_id.apply(list)
    selected = (
        ordered.sort_values(
            ["_physical_group", "physical_evidence_count", "physical_score", "candidate_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates("_physical_group")
        .copy()
    )
    selected["member_candidate_ids"] = selected._physical_group.map(
        lambda value: json.dumps(members.loc[value])
    )
    selected["member_count"] = selected._physical_group.map(members.map(len))
    selected["physical_site_id"] = [
        _stable_id("terrain_site", json.loads(value)) for value in selected.member_candidate_ids
    ]
    selected["representative_coordinate_rule"] = (
        "existing anchor with most independent evidence, then physical score, then stable ID"
    )
    return selected.drop(columns="_physical_group").reset_index(drop=True)


def build_candidate_sites(
    hybrid: gpd.GeoDataFrame, raw: gpd.GeoDataFrame,
    baseline: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Union prior hybrid sites with new reversible multi-scale terrain sites."""
    anchors = hybrid.copy()
    if baseline is not None:
        baseline = baseline.copy()
        baseline_ids = set(baseline.candidate_id.astype(str))
        anchors = pd.concat(
            [baseline, hybrid.loc[~hybrid.candidate_id.astype(str).isin(baseline_ids)]],
            ignore_index=True, sort=False,
        )
        anchors = gpd.GeoDataFrame(anchors, geometry="geometry", crs=hybrid.crs)
    extrema = multiscale_extrema(raw)
    extrema = extrema.loc[
        ~extrema.candidate_id.astype(str).isin(represented_candidate_ids(anchors))
    ].copy()
    grouped = group_extrema(extrema)
    anchors["candidate_source"] = np.where(
        anchors.candidate_id.astype(str).isin(set(hybrid.candidate_id.astype(str))),
        "existing_hybrid", "preserved_baseline_anchor",
    )
    if "quality_missing_count" not in anchors:
        anchors["quality_missing_count"] = anchors.reindex(columns=PHYSICAL_SIGNALS).isna().sum(axis=1)
    combined = pd.concat([anchors, grouped], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=hybrid.crs)
    if not combined.candidate_id.is_unique:
        raise ValueError("Candidate IDs must remain unique after the physical union")
    return combined, extrema


def numeric_matrix(frame: pd.DataFrame, features: list[str] | tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def feature_parity(training: pd.DataFrame, inference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, columns in FEATURE_FAMILIES.items():
        for column in columns:
            train = pd.to_numeric(
                training[column] if column in training else pd.Series(np.nan, index=training.index),
                errors="coerce",
            ).astype(float)
            infer = pd.to_numeric(
                inference[column] if column in inference else pd.Series(np.nan, index=inference.index),
                errors="coerce",
            ).astype(float)
            train_finite = train[np.isfinite(train)]
            infer_finite = infer[np.isfinite(infer)]
            rows.append(
                dict(
                    family=family,
                    feature=column,
                    training_available=column in training,
                    inference_available=column in inference,
                    training_missing_rate=float(1 - len(train_finite) / max(len(train), 1)),
                    inference_missing_rate=float(1 - len(infer_finite) / max(len(infer), 1)),
                    training_unique=int(train_finite.nunique()),
                    inference_unique=int(infer_finite.nunique()),
                    training_median=float(train_finite.median()) if len(train_finite) else np.nan,
                    inference_median=float(infer_finite.median()) if len(infer_finite) else np.nan,
                    training_p05=float(train_finite.quantile(.05)) if len(train_finite) else np.nan,
                    training_p95=float(train_finite.quantile(.95)) if len(train_finite) else np.nan,
                    inference_p05=float(infer_finite.quantile(.05)) if len(infer_finite) else np.nan,
                    inference_p95=float(infer_finite.quantile(.95)) if len(infer_finite) else np.nan,
                    near_constant_training=bool(train_finite.nunique() <= 1),
                    near_constant_inference=bool(infer_finite.nunique() <= 1),
                )
            )
    return pd.DataFrame(rows).drop_duplicates("feature").reset_index(drop=True)


@dataclass(frozen=True)
class RankResult:
    frame: gpd.GeoDataFrame
    policy: str


def rank_sites(frame: gpd.GeoDataFrame, score_column: str, policy: str) -> RankResult:
    ranked = frame.copy()
    if policy == "raw":
        ranked["research_priority"] = ranked[score_column]
    elif policy == "soft_local_diversity":
        ordered = ranked.sort_values(
            [score_column, "candidate_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        # Fixed, mild penalty for earlier same-road hypotheses within 40 m.
        priority = ordered[score_column].to_numpy(float).copy()
        chain = pd.to_numeric(ordered.road_chainage_m, errors="coerce").to_numpy(float)
        roads = list(zip(ordered.road_id.astype(str), ordered.road_part_index))
        selected: list[int] = []
        remaining = set(range(len(ordered)))
        for _ in range(min(1000, len(ordered))):
            i = max(remaining, key=lambda j: (priority[j], -j))
            selected.append(i)
            remaining.remove(i)
            for j in remaining:
                if roads[j] == roads[i] and np.isfinite(chain[j]) and abs(chain[j] - chain[i]) <= 40:
                    priority[j] -= 0.1
        tail = [i for i in range(len(ordered)) if i not in set(selected)]
        ranked = ordered.iloc[selected + tail].copy()
        ranked["research_priority"] = priority[selected + tail]
    else:
        raise ValueError(f"Unknown ranking policy: {policy}")
    ranked = ranked.sort_values(
        ["research_priority", score_column, "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["global_rank"] = np.arange(1, len(ranked) + 1)
    ranked["ranking_policy"] = policy
    return RankResult(ranked, policy)
