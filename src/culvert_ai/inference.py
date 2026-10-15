"""Route-general FLOW-C + dynamic ROUTE-Z inference.

The functions in this module never fit a model and never read authoritative
culvert locations.  Route identifiers are used only to define the unlabeled
score distribution for normalization and to label output rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from culvert_ai.candidates import CandidateSettings, generate_candidates, generate_road_route_candidates
from culvert_ai.features import build_feature_table
from culvert_ai.flow_c_bundle import (
    REQUIRED_FEATURE_CRS,
    assert_route_general_feature_columns,
    load_frozen_flow_c_bundle,
)
from culvert_ai.io import add_wgs84_coordinates, clean_geometry, project_layers_to_metric


SITE_RADIUS_M = 20.0
FROZEN_INTERVAL_M = 75.0
DEM_THRESHOLDS = ("0p1km2", "0p5km2", "2km2", "10km2")


@dataclass(frozen=True)
class RouteZConfig:
    """Stability and fallback rules for dynamic route-local normalization."""

    minimum_standard_candidates: int = 8
    minimum_robust_candidates: int = 4
    minimum_scale: float = 1e-6
    global_center: float | None = None
    global_scale: float | None = None


def dynamic_route_z(
    candidates: pd.DataFrame,
    *,
    score_column: str = "flow_c_score",
    route_column: str = "route_id",
    config: RouteZConfig | None = None,
) -> pd.DataFrame:
    """Normalize each route from its own unlabeled FLOW-C score distribution.

    Stable routes use mean / population standard deviation.  Small routes use
    median / IQR when that estimate is stable.  Remaining routes use frozen
    global score statistics supplied with the model bundle; if those statistics
    are unavailable, they receive neutral z=0 and retain FLOW-C as the tie-break.
    """
    config = config or RouteZConfig()
    if score_column not in candidates or route_column not in candidates:
        raise ValueError(f"Required columns are missing: {score_column}, {route_column}")
    result = candidates.copy()
    result[route_column] = result[route_column].fillna("<unassigned>").astype(str)
    result["route_z_score"] = 0.0
    result["route_z_method"] = "neutral_fallback"
    result["route_z_candidate_count"] = 0
    result["route_z_center"] = np.nan
    result["route_z_scale"] = np.nan

    has_global = (
        config.global_center is not None
        and config.global_scale is not None
        and np.isfinite(config.global_center)
        and np.isfinite(config.global_scale)
        and float(config.global_scale) >= config.minimum_scale
    )
    for _, indices in result.groupby(route_column, sort=False).groups.items():
        values = pd.to_numeric(result.loc[indices, score_column], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        count = len(values)
        method = "neutral_fallback"
        center = np.nan
        scale = np.nan
        if len(finite) >= config.minimum_standard_candidates:
            candidate_center = float(np.mean(finite))
            candidate_scale = float(np.std(finite, ddof=0))
            if candidate_scale >= config.minimum_scale:
                center, scale, method = candidate_center, candidate_scale, "route_mean_std"
        if method == "neutral_fallback" and len(finite) >= config.minimum_robust_candidates:
            q25, candidate_center, q75 = np.quantile(finite, [0.25, 0.50, 0.75])
            candidate_scale = float((q75 - q25) / 1.349)
            if candidate_scale >= config.minimum_scale:
                center, scale, method = float(candidate_center), candidate_scale, "route_median_iqr"
        if method == "neutral_fallback" and has_global:
            center = float(config.global_center)
            scale = float(config.global_scale)
            method = "frozen_global_fallback"

        normalized = np.zeros(count, dtype=float)
        if method != "neutral_fallback":
            normalized = (values - center) / scale
            normalized[~np.isfinite(normalized)] = 0.0
        result.loc[indices, "route_z_score"] = normalized
        result.loc[indices, "route_z_method"] = method
        result.loc[indices, "route_z_candidate_count"] = count
        result.loc[indices, "route_z_center"] = center
        result.loc[indices, "route_z_scale"] = scale
    return result


def score_frozen_flow_c(
    features: gpd.GeoDataFrame,
    model_bundle: str | Path | Mapping,
) -> tuple[gpd.GeoDataFrame, Mapping]:
    """Score candidates with a pre-existing audited FLOW-C model bundle."""
    bundle = load_frozen_flow_c_bundle(model_bundle)
    feature_columns = [str(column) for column in bundle["feature_columns"]]
    assert_route_general_feature_columns(feature_columns)
    if features.crs is None or features.crs.to_string().upper() != REQUIRED_FEATURE_CRS:
        raise ValueError(
            f"FLOW-C features must use the frozen {REQUIRED_FEATURE_CRS} CRS; "
            f"received {features.crs}."
        )
    missing_features = sorted(set(feature_columns) - set(features.columns))
    if missing_features:
        raise ValueError(
            "The arbitrary-road feature pipeline did not produce every frozen FLOW-C field: "
            + ", ".join(missing_features)
        )
    x = features.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    fill = bundle["fill_values"]
    x = x.fillna(pd.Series(fill).reindex(feature_columns))
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x), dtype=float)[:, 1]
    else:  # pragma: no cover - bundle validation checks this before scoring
        raise TypeError("Frozen FLOW-C estimator must implement predict_proba().")
    scored = features.copy()
    scored["flow_c_score"] = score
    return scored, bundle


def rank_flow_c_candidates(
    scored_candidates: gpd.GeoDataFrame,
    *,
    route_column: str = "route_id",
    route_z_config: RouteZConfig | None = None,
    site_radius_m: float = SITE_RADIUS_M,
) -> gpd.GeoDataFrame:
    """Create one global queue from any set of already FLOW-C-scored roads."""
    if site_radius_m != SITE_RADIUS_M:
        raise ValueError("The physical-site radius is frozen at 20 m.")
    if "candidate_id" not in scored_candidates or "flow_c_score" not in scored_candidates:
        raise ValueError("candidate_id and flow_c_score are required.")
    normalized = dynamic_route_z(
        scored_candidates,
        score_column="flow_c_score",
        route_column=route_column,
        config=route_z_config,
    )
    ordered = normalized.sort_values(
        ["route_z_score", "flow_c_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    _, xy = _metric_copy_and_xy(ordered)
    tree = cKDTree(xy)
    suppressed = np.zeros(len(ordered), dtype=bool)
    selected: list[int] = []
    for position in range(len(ordered)):
        if suppressed[position]:
            continue
        selected.append(position)
        neighbors = np.asarray(tree.query_ball_point(xy[position], site_radius_m), dtype=int)
        suppressed[neighbors] = True
    sites = ordered.iloc[selected].copy().reset_index(drop=True)
    sites["global_rank"] = np.arange(1, len(sites) + 1)
    sites["physical_site_id"] = [f"site_{index:06d}" for index in range(1, len(sites) + 1)]
    sites["final_priority"] = sites["route_z_score"]
    sites["priority_percentile"] = 1.0 - (
        (sites["global_rank"] - 1) / max(len(sites), 1)
    )
    sites["route_local_rank"] = (
        sites.groupby(route_column, sort=False)["global_rank"]
        .rank(method="first")
        .astype(int)
    )
    if "latitude" not in sites or "longitude" not in sites:
        sites = add_wgs84_coordinates(sites)
    standard = [
        "global_rank", route_column, "latitude", "longitude", "candidate_id",
        "physical_site_id", "flow_c_score", "route_z_score", "final_priority",
        "route_z_method", "route_z_candidate_count", "route_z_center", "route_z_scale",
        "priority_percentile", "route_local_rank", "geometry",
    ]
    # These values support careful, feature-based UI summaries. They are
    # presentation metadata only and never participate in post-score reranking.
    explainable = (
        "distance_to_nearest_stream_m",
        "route_low_point_score",
        "local_road_stream_separation_m",
        "distance_to_nearest_mapped_crossing_m",
        "dem_culvert_terrain_score",
        "hydrology_edge_risk",
        "log1p_distance_to_dem_road_drainage_crossing_0p1km2_m",
        "log1p_distance_to_dem_road_drainage_crossing_2km2_m",
    )
    optional = [
        column
        for column in ("road_id", "road_name", "source", *explainable)
        if column in sites
    ]
    return gpd.GeoDataFrame(sites.loc[:, standard[:-1] + optional + ["geometry"]], geometry="geometry", crs=sites.crs)


def predict_culverts_from_features(
    features: gpd.GeoDataFrame,
    *,
    flow_c_model: str | Path | Mapping,
    route_column: str = "route_id",
    route_z_config: RouteZConfig | None = None,
) -> gpd.GeoDataFrame:
    """Standard inference boundary when generic GIS/DEM features already exist."""
    scored, bundle = score_frozen_flow_c(features, flow_c_model)
    config = route_z_config or _route_z_config_from_bundle(bundle)
    return rank_flow_c_candidates(scored, route_column=route_column, route_z_config=config)


def predict_culverts(
    road_geometry: gpd.GeoDataFrame,
    *,
    streams: gpd.GeoDataFrame,
    dem_path: str | Path,
    flow_c_model: str | Path | Mapping,
    route_id_column: str | None = None,
    flow_accumulation_path: str | Path | None = None,
    drainage_area_path: str | Path | None = None,
    landcover_path: str | Path | None = None,
    dem_crossings: Mapping[str, gpd.GeoDataFrame] | None = None,
    feature_enricher: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None,
    route_z_config: RouteZConfig | None = None,
) -> gpd.GeoDataFrame:
    """Run frozen candidates -> generic features -> FLOW-C -> ROUTE-Z -> sites.

    ``feature_enricher`` is the extension point for the frozen DEM hydrology
    extractor when its crossing layers are produced outside this package.  It
    must be label-free and must not change candidate coordinates.
    """
    bundle = load_frozen_flow_c_bundle(flow_c_model)
    roads = clean_geometry(road_geometry)
    streams = clean_geometry(streams)
    candidates = generate_road_route_candidates(
        roads,
        interval_m=FROZEN_INTERVAL_M,
        lateral_offsets_m=(0.0,),
        include_all_roads=True,
        route_id_column=route_id_column,
    )
    if candidates.empty:
        raise ValueError("The supplied road geometry produced no interval candidates.")
    candidates["route_id"] = candidates["matched_route"].fillna(candidates["road_id"]).astype(str)
    features = build_feature_table(
        candidates,
        known_culverts=None,
        negative_culverts=None,
        roads=roads,
        streams=streams,
        dem_path=dem_path,
        flow_accumulation_path=flow_accumulation_path,
        drainage_area_path=drainage_area_path,
        landcover_path=landcover_path,
        density_radius_m=75.0,
        density_radii_m=(100.0, 250.0, 500.0),
    )
    features = add_mapped_crossing_context(features, roads, streams)
    features = add_dem_crossing_context(features, dem_crossings or {})
    features = add_route_local_density_normalization(features, route_column="route_id")
    if feature_enricher is not None:
        original_geometry = features.geometry.copy()
        features = feature_enricher(features.copy())
        if len(features) != len(original_geometry) or not features.geometry.reset_index(drop=True).equals(
            original_geometry.reset_index(drop=True)
        ):
            raise ValueError("feature_enricher must preserve candidate rows and coordinates.")
    return predict_culverts_from_features(
        features,
        flow_c_model=bundle,
        route_column="route_id",
        route_z_config=route_z_config,
    )


def filter_predictions_by_route(
    predictions: gpd.GeoDataFrame,
    route_ids: str | list[str] | set[str],
    *,
    route_column: str = "route_id",
) -> gpd.GeoDataFrame:
    """Presentation-only filtering that preserves the original global ranks."""
    selected = {str(route_ids)} if isinstance(route_ids, str) else {str(value) for value in route_ids}
    return predictions.loc[predictions[route_column].astype(str).isin(selected)].copy()


def add_route_local_density_normalization(
    features: gpd.GeoDataFrame,
    *,
    route_column: str = "route_id",
) -> gpd.GeoDataFrame:
    """Reproduce label-free robust road-density fields on arbitrary routes."""
    result = features.copy()
    density_columns = (
        "road_density_100m_m_per_sqkm", "road_density_250m_m_per_sqkm",
        "road_density_500m_m_per_sqkm", "road_density_m_per_sqkm",
    )
    for column in density_columns:
        if column not in result:
            continue
        output = pd.Series(0.0, index=result.index)
        all_values = pd.to_numeric(result[column], errors="coerce")
        global_median = float(all_values.median())
        global_iqr = float(all_values.quantile(0.75) - all_values.quantile(0.25))
        for _, indices in result.groupby(route_column, sort=False).groups.items():
            values = all_values.loc[indices]
            center = float(values.median())
            scale = float(values.quantile(0.75) - values.quantile(0.25))
            if not np.isfinite(scale) or scale < 1e-9:
                center, scale = global_median, global_iqr
            if not np.isfinite(scale) or scale < 1e-9:
                output.loc[indices] = 0.0
            else:
                output.loc[indices] = (values - center) / scale
        result[f"{column}__robust_route_normalized"] = output.fillna(0.0)
    return result


def add_mapped_crossing_context(
    features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    streams: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach mapped-water crossing geometry without inventory information."""
    result = features.copy()
    crossings = generate_candidates(
        roads,
        streams,
        CandidateSettings(snap_tolerance_m=20.0, min_spacing_m=20.0),
    )
    if crossings.empty:
        result["local_road_stream_separation_m"] = np.inf
        result["local_road_stream_angle_degrees"] = np.nan
        result["distance_to_nearest_mapped_crossing_m"] = np.inf
        result["mapped_crossing_count_250m"] = 0
        result["mapped_crossing_count_500m"] = 0
        return result
    (points_m, crossings_m), _ = project_layers_to_metric(result, crossings)
    point_xy = np.column_stack([points_m.geometry.x, points_m.geometry.y])
    crossing_xy = np.column_stack([crossings_m.geometry.x, crossings_m.geometry.y])
    tree = cKDTree(crossing_xy)
    distance, nearest = tree.query(point_xy)
    result["local_road_stream_separation_m"] = distance
    result["distance_to_nearest_mapped_crossing_m"] = distance
    result["local_road_stream_angle_degrees"] = pd.to_numeric(
        crossings_m.iloc[nearest]["crossing_angle_degrees"], errors="coerce"
    ).to_numpy()
    result["mapped_crossing_count_250m"] = tree.query_ball_point(point_xy, 250.0, return_length=True)
    result["mapped_crossing_count_500m"] = tree.query_ball_point(point_xy, 500.0, return_length=True)
    return result


def add_dem_crossing_context(
    features: gpd.GeoDataFrame,
    crossing_layers: Mapping[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Attach precomputed, label-free DEM road/drainage crossing layers."""
    result = features.copy()
    for threshold in DEM_THRESHOLDS:
        layer = crossing_layers.get(threshold)
        distance_column = f"distance_to_dem_road_drainage_crossing_{threshold}_m"
        area_column = f"nearest_crossing_upstream_area_{threshold}_m2"
        angle_column = f"nearest_crossing_angle_{threshold}_degrees"
        if layer is None or layer.empty:
            result[distance_column] = np.inf
            result[area_column] = 0.0
            result[angle_column] = np.nan
        else:
            (points_m, layer_m), _ = project_layers_to_metric(result, clean_geometry(layer))
            point_xy = np.column_stack([points_m.geometry.x, points_m.geometry.y])
            layer_xy = np.column_stack([layer_m.geometry.x, layer_m.geometry.y])
            distance, nearest = cKDTree(layer_xy).query(point_xy)
            nearest_rows = layer_m.iloc[nearest]
            result[distance_column] = distance
            result[area_column] = _first_numeric_column(
                nearest_rows, ("upstream_area_m2", "maximum_upstream_area_5x5_m2"), 0.0
            )
            result[angle_column] = _first_numeric_column(
                nearest_rows, ("crossing_angle_degrees", "angle_degrees"), np.nan
            )
        result[f"log1p_{distance_column}"] = np.log1p(
            pd.to_numeric(result[distance_column], errors="coerce").clip(lower=0.0)
        )
        result[f"log1p_{area_column}"] = np.log1p(
            pd.to_numeric(result[area_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    if "hydrology_edge_risk" not in result:
        result["hydrology_edge_risk"] = 0
    return result


def _route_z_config_from_bundle(bundle: Mapping) -> RouteZConfig:
    metadata = bundle.get("route_z_global", {})
    return RouteZConfig(
        global_center=metadata.get("center"),
        global_scale=metadata.get("scale"),
    )


def _metric_copy_and_xy(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    if frame.crs is None:
        raise ValueError("Candidate geometry must have a CRS for the fixed 20 m site rule.")
    (metric,), _ = project_layers_to_metric(frame)
    return metric, np.column_stack([metric.geometry.x, metric.geometry.y])


def _first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...], fill: float) -> np.ndarray:
    for column in columns:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").fillna(fill).to_numpy(float)
    return np.full(len(frame), fill, dtype=float)
