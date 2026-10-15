"""Label-free consensus features and precision-first culvert ranking.

This module operates only on frozen candidate-side signals.  It deliberately
does not accept labels, official inventory coordinates, or match distances.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


LOCAL_WINDOW_M = 100.0
NEAR_EQUIVALENT_DELTA = 0.03


def _numeric(frame: pd.DataFrame, column: str, fill: float = np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(fill, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)


def _bounded(frame: pd.DataFrame, column: str, fill: float = 0.0) -> pd.Series:
    return _numeric(frame, column, fill).clip(0.0, 1.0)


def _proximity(frame: pd.DataFrame, column: str, scale_m: float) -> pd.Series:
    distance = _numeric(frame, column, np.inf).clip(lower=0.0)
    return np.exp(-distance / scale_m)


def _route_percentile(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    working = pd.DataFrame({"route": frame["route"].astype(str), "value": values}, index=frame.index)
    return working.groupby("route", sort=False)["value"].rank(method="average", pct=True).fillna(0.0)


def _perpendicularity(frame: pd.DataFrame, column: str) -> pd.Series:
    angle = _numeric(frame, column)
    return (1.0 - (angle - 90.0).abs() / 90.0).clip(0.0, 1.0).fillna(0.0)


def _local_competition(frame: pd.DataFrame, base_score: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in (
        "local_model_percentile_100m",
        "local_gap_to_best_100m",
        "local_gap_to_second_best_100m",
        "local_competitor_count_100m",
        "local_near_equivalent_count_100m",
        "local_strong_maximum",
        "local_prominence_score",
        "local_ambiguity_penalty",
    ):
        result[column] = 0.0

    for _, indices in frame.groupby("route", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        xy = frame.loc[positions, ["x_m", "y_m"]].to_numpy(float)
        scores = base_score.loc[positions].to_numpy(float)
        tree = cKDTree(xy)
        neighborhoods = tree.query_ball_point(xy, LOCAL_WINDOW_M)
        for local_pos, neighbors in enumerate(neighborhoods):
            neighbors = np.asarray(neighbors, dtype=int)
            local_scores = scores[neighbors]
            current = scores[local_pos]
            descending = np.sort(local_scores)[::-1]
            best = float(descending[0])
            second = float(descending[1]) if len(descending) > 1 else best
            percentile = float(np.mean(local_scores <= current))
            gap_best = current - best
            gap_second = current - second
            near_equivalent = int(np.count_nonzero(np.abs(local_scores - current) <= NEAR_EQUIVALENT_DELTA) - 1)
            strong_maximum = float(gap_best >= -1e-12 and (len(descending) == 1 or gap_second >= 0.02))
            best_closeness = math.exp(min(gap_best, 0.0) / 0.05)
            second_separation = float(np.clip(max(gap_second, 0.0) / 0.10, 0.0, 1.0))
            uniqueness = 1.0 / (1.0 + max(near_equivalent, 0))
            prominence = 0.35 * percentile + 0.25 * best_closeness + 0.20 * second_separation + 0.20 * uniqueness
            ambiguity = 0.60 * (1.0 - uniqueness) + 0.40 * (1.0 - strong_maximum)
            target = positions[local_pos]
            result.at[target, "local_model_percentile_100m"] = percentile
            result.at[target, "local_gap_to_best_100m"] = gap_best
            result.at[target, "local_gap_to_second_best_100m"] = gap_second
            result.at[target, "local_competitor_count_100m"] = max(len(neighbors) - 1, 0)
            result.at[target, "local_near_equivalent_count_100m"] = max(near_equivalent, 0)
            result.at[target, "local_strong_maximum"] = strong_maximum
            result.at[target, "local_prominence_score"] = prominence
            result.at[target, "local_ambiguity_penalty"] = ambiguity
    return result


def add_consensus_precision_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return candidates with fixed, interpretable consensus features and score."""
    required = {"candidate_id", "route", "x_m", "y_m", "flow_c_score", "route_z_score"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing required consensus-ranking fields: {missing}")
    frame = candidates.copy().reset_index(drop=True)

    frame["flow_c_route_percentile"] = _route_percentile(frame, _numeric(frame, "flow_c_score", 0.0))
    frame["route_z_route_percentile"] = _route_percentile(frame, _numeric(frame, "route_z_score", 0.0))
    # Bounded monotone transform: it keeps ROUTE-Z ordering while preventing a
    # handful of extreme z-scores from overwhelming independent evidence.
    frame["route_z_support"] = 0.5 + 0.5 * np.tanh(_numeric(frame, "route_z_score", 0.0) / 2.0)
    frame["model_support_score"] = (
        0.60 * frame["route_z_support"]
        + 0.25 * frame["flow_c_route_percentile"]
        + 0.15 * frame["route_z_route_percentile"]
    )

    frame["crossing_proximity"] = _proximity(frame, "minimum_any_crossing_distance_m", 75.0)
    frame["mapped_crossing_proximity"] = _proximity(frame, "distance_to_nearest_mapped_crossing_m", 75.0)
    frame["dem_crossing_proximity"] = _proximity(frame, "minimum_dem_crossing_distance_m", 75.0)
    frame["mapped_dem_agreement"] = _proximity(frame, "mapped_dem_crossing_distance_agreement_m", 100.0)
    frame["drainage_threshold_agreement"] = (_numeric(frame, "dem_crossing_threshold_support_count_50m", 0.0) / 4.0).clip(0.0, 1.0)
    frame["flow_accumulation_support"] = _route_percentile(frame, _numeric(frame, "log1p_flow_accumulation_cells", 0.0))
    frame["upstream_area_support"] = _route_percentile(frame, _numeric(frame, "log1p_maximum_upstream_area_5x5_m2", 0.0))
    frame["hydrology_support_score"] = (
        0.20 * frame["crossing_proximity"]
        + 0.12 * frame["mapped_crossing_proximity"]
        + 0.14 * frame["dem_crossing_proximity"]
        + 0.16 * frame["mapped_dem_agreement"]
        + 0.16 * frame["drainage_threshold_agreement"]
        + 0.10 * frame["flow_accumulation_support"]
        + 0.12 * frame["upstream_area_support"]
    )

    frame["mapped_crossing_perpendicularity"] = _perpendicularity(frame, "local_road_stream_angle_degrees")
    dem_angles = [
        _perpendicularity(frame, f"nearest_crossing_angle_{key}_degrees")
        for key in ("0p1km2", "0p5km2", "2km2", "10km2")
    ]
    frame["dem_crossing_perpendicularity"] = pd.concat(dem_angles, axis=1).max(axis=1)
    frame["crossing_perpendicularity"] = frame[["mapped_crossing_perpendicularity", "dem_crossing_perpendicularity"]].max(axis=1)
    frame["mapped_dem_both_near"] = _bounded(frame, "mapped_dem_both_within_50m")
    frame["drainage_crosses_road"] = (
        frame["crossing_proximity"].ge(math.exp(-50.0 / 75.0))
        & frame["crossing_perpendicularity"].ge(0.50)
    ).astype(float)
    frame["road_low_point_support"] = _bounded(frame, "route_low_point_score")
    frame["road_crossing_support_score"] = (
        0.35 * frame["crossing_proximity"]
        + 0.20 * frame["crossing_perpendicularity"]
        + 0.20 * frame["drainage_crosses_road"]
        + 0.15 * frame["road_low_point_support"]
        + 0.10 * frame["mapped_dem_both_near"]
    )

    frame["terrain_low_position_support"] = _route_percentile(frame, -_numeric(frame, "topographic_position_9x9_m", 0.0))
    frame["terrain_relief_support"] = _route_percentile(frame, _numeric(frame, "elevation_relief_31x31_m", 0.0))
    frame["terrain_roughness_support"] = _route_percentile(frame, _numeric(frame, "terrain_roughness_31x31_m", 0.0))
    frame["terrain_wetness_support"] = _route_percentile(frame, _numeric(frame, "topographic_wetness_proxy_31x31", 0.0))
    slope = _numeric(frame, "slope_degrees", 12.0).clip(lower=0.0)
    frame["terrain_slope_support"] = np.exp(-(slope - 4.0).abs() / 8.0)
    frame["terrain_support_score"] = (
        0.30 * frame["road_low_point_support"]
        + 0.20 * frame["terrain_low_position_support"]
        + 0.15 * frame["terrain_wetness_support"]
        + 0.15 * frame["terrain_relief_support"]
        + 0.10 * frame["terrain_roughness_support"]
        + 0.10 * frame["terrain_slope_support"]
    )

    local_base = 0.55 * frame["route_z_support"] + 0.45 * frame["flow_c_route_percentile"]
    frame = pd.concat([frame, _local_competition(frame, local_base)], axis=1)

    family_columns = [
        "model_support_score", "hydrology_support_score", "road_crossing_support_score",
        "terrain_support_score", "local_prominence_score",
    ]
    frame["supporting_family_count"] = frame[family_columns].ge(0.65).sum(axis=1)
    frame["strong_flow_plus_crossing"] = (
        frame["flow_c_route_percentile"].ge(0.90) & frame["road_crossing_support_score"].ge(0.65)
    ).astype(float)
    frame["strong_flow_plus_mapped_dem"] = (
        frame["flow_c_route_percentile"].ge(0.90) & frame["mapped_dem_agreement"].ge(0.65)
    ).astype(float)
    frame["strong_drainage_plus_low_point"] = (
        frame["hydrology_support_score"].ge(0.65) & frame["road_low_point_support"].ge(0.65)
    ).astype(float)
    pair_agreement = frame[[
        "strong_flow_plus_crossing", "strong_flow_plus_mapped_dem", "strong_drainage_plus_low_point"
    ]].mean(axis=1)
    frame["consensus_strength_score"] = 0.70 * (frame["supporting_family_count"] / 5.0) + 0.30 * pair_agreement

    sorted_families = np.sort(frame[family_columns].to_numpy(float), axis=1)
    frame["single_family_penalty"] = (
        (sorted_families[:, -1] >= 0.75) & (sorted_families[:, -2] < 0.50)
    ).astype(float)
    frame["weak_crossing_penalty"] = (1.0 - frame["road_crossing_support_score"]).clip(0.0, 1.0)

    # Fixed before evaluation. Positive weights sum to one. Penalties explicitly
    # target ambiguity and one-family candidates without removing candidates.
    frame["precision_score"] = 100.0 * (
        0.40 * frame["model_support_score"]
        + 0.14 * frame["hydrology_support_score"]
        + 0.14 * frame["road_crossing_support_score"]
        + 0.08 * frame["terrain_support_score"]
        + 0.14 * frame["local_prominence_score"]
        + 0.10 * frame["consensus_strength_score"]
        - 0.05 * frame["local_ambiguity_penalty"]
        - 0.04 * frame["single_family_penalty"]
        - 0.03 * frame["weak_crossing_penalty"]
    ).clip(0.0, 1.0)

    reason_flags = pd.DataFrame({
        "strong_flow_c": frame["flow_c_route_percentile"].ge(0.90),
        "strong_route_z": frame["route_z_support"].ge(0.80),
        "dem_mapped_agreement": frame["mapped_dem_agreement"].ge(0.65),
        "strong_drainage_crossing": frame["road_crossing_support_score"].ge(0.65),
        "terrain_low_point": frame["terrain_support_score"].ge(0.65),
        "route_local_maximum": frame["local_strong_maximum"].eq(1.0),
        "ambiguous_local_competition": frame["local_ambiguity_penalty"].ge(0.60),
        "weak_crossing_support": frame["road_crossing_support_score"].lt(0.40),
    })
    frame["ranking_reason"] = reason_flags.apply(
        lambda row: "|".join(row.index[row.to_numpy(bool)]) or "moderate_multi_signal_support", axis=1
    )
    frame["major_supporting_signal_categories"] = frame[family_columns].apply(
        lambda row: "|".join(name.removesuffix("_support_score").removesuffix("_score")
                              for name, value in row.items() if value >= 0.65) or "none_strong",
        axis=1,
    )
    return frame

