from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.windows import Window

from culvert_ai.io import add_wgs84_coordinates, clean_geometry, project_layers_to_metric


KNOWN_PATTERN_RADII_M = (250.0, 500.0, 1000.0)
FIELD_ROUTE_INFERENCE_MAX_DISTANCE_M = 40.0
FIELD_ROUTE_LOCAL_SUPPORT_DECAY_M = 750.0
FIELD_ROUTE_LOCAL_SUPPORT_MAX_DISTANCE_M = 3000.0
ROUTE_PROFILE_DIP_WINDOW_M = 60.0
ROUTE_PROFILE_DIP_SHOULDER_M = 20.0
ROUTE_TOKEN_RE = re.compile(
    r"\b(?P<prefix>NY|US|I|CR)\s*-?\s*"
    r"(?:(?:HWY|HIGHWAY|RTE|ROUTE|RT)\s*-?\s*)?"
    r"(?P<number>\d+[A-Z]?)\b",
    re.IGNORECASE,
)
GENERIC_ROUTE_TOKEN_RE = re.compile(
    r"\b(?:ROUTE|RTE|RT)\s*-?\s*(?P<number>\d+[A-Z]?)\b",
    re.IGNORECASE,
)


def build_feature_table(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame | None = None,
    negative_culverts: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | None = None,
    streams: gpd.GeoDataFrame | None = None,
    dem_path: str | Path | None = None,
    flow_accumulation_path: str | Path | None = None,
    drainage_area_path: str | Path | None = None,
    landcover_path: str | Path | None = None,
    positive_radius_m: float = 10.0,
    negative_radius_m: float = 10.0,
    density_radius_m: float = 75.0,
    density_radii_m: tuple[float, ...] | None = None,
) -> gpd.GeoDataFrame:
    (features,), metric_crs = project_layers_to_metric(clean_geometry(candidates))
    features = features.copy()
    features["x_m"] = features.geometry.x
    features["y_m"] = features.geometry.y
    features = add_candidate_derived_features(features)

    density_radii = _density_radii(density_radius_m, density_radii_m)
    known_m = None
    negative_m = None

    if known_culverts is not None:
        known_m = clean_geometry(known_culverts).to_crs(metric_crs)
        features = add_known_culvert_labels(features, known_m, positive_radius_m)
        features = add_known_culvert_pattern_features(
            features,
            known_m,
            positive_radius_m=positive_radius_m,
        )

    if negative_culverts is not None:
        negative_m = clean_geometry(negative_culverts).to_crs(metric_crs)
        features = add_negative_culvert_labels(features, negative_m, negative_radius_m)

    if known_m is not None or negative_m is not None:
        features = add_field_route_feedback_features(
            features,
            known_culverts=known_m,
            negative_culverts=negative_m,
        )

    if roads is not None:
        roads_m = clean_geometry(roads).to_crs(metric_crs)
        features["distance_to_nearest_road_m"] = _nearest_distance(features.geometry, roads_m)
        for radius in density_radii:
            column = _density_column("road", radius)
            features[column] = _line_density(features.geometry, roads_m, radius)
        features["road_density_m_per_sqkm"] = features[_density_column("road", density_radius_m)]

    if streams is not None:
        streams_m = clean_geometry(streams).to_crs(metric_crs)
        features["distance_to_nearest_stream_m"] = _nearest_distance(features.geometry, streams_m)
        for radius in density_radii:
            column = _density_column("stream", radius)
            features[column] = _line_density(features.geometry, streams_m, radius)
        features["stream_density_m_per_sqkm"] = features[
            _density_column("stream", density_radius_m)
        ]

    if dem_path:
        features = add_raster_samples(features, dem_path, prefix="dem")
        features = add_dem_hydrology_proxies(features)
        features = add_dem_culvert_terrain_features(features)
        features = add_route_profile_dem_features(features)
        if known_m is not None:
            features = add_approved_known_dem_similarity_features(
                features,
                positive_radius_m=positive_radius_m,
            )
            features = add_known_culvert_pattern_score(features)

    if flow_accumulation_path:
        features = add_raster_samples(features, flow_accumulation_path, prefix="flow_accumulation")
        features = add_hydrology_raster_features(features, "flow_accumulation")

    if drainage_area_path:
        features = add_raster_samples(features, drainage_area_path, prefix="drainage_area")
        features = add_hydrology_raster_features(features, "drainage_area")

    if landcover_path:
        features = add_raster_samples(features, landcover_path, prefix="landcover")

    features = add_training_sample_weights(features)
    features = add_wgs84_coordinates(features)
    return features.reset_index(drop=True)


def add_candidate_derived_features(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    features = candidates.copy()
    if "road_stream_distance_m" in features.columns:
        distances = pd.to_numeric(features["road_stream_distance_m"], errors="coerce").clip(lower=0)
        features["log_road_stream_distance_m"] = np.log1p(distances)
        features["is_exact_road_stream_intersection"] = (distances <= 0.01).astype(int)
        features["road_stream_proximity_signal"] = (1.0 / (1.0 + distances / 20.0)).fillna(0.0)

    if "crossing_angle_degrees" in features.columns:
        angle = pd.to_numeric(features["crossing_angle_degrees"], errors="coerce")
        features["crossing_angle_abs_from_90"] = (90 - angle).abs()
        features["crossing_angle_perpendicularity"] = (
            1.0 - (features["crossing_angle_abs_from_90"] / 90.0)
        ).clip(0, 1)

    if {"road_stream_proximity_signal", "crossing_angle_perpendicularity"}.issubset(
        features.columns
    ):
        features["crossing_geometry_signal"] = (
            0.65 * features["road_stream_proximity_signal"].fillna(0.0)
            + 0.35 * features["crossing_angle_perpendicularity"].fillna(0.0)
        ).clip(0, 1)

    if "source" in features.columns:
        source = features["source"].fillna("").astype(str).str.lower()
        features["source_exact_intersection"] = source.eq("exact_road_stream_intersection").astype(
            int
        )
        features["source_nearest_approach"] = source.eq("nearest_road_stream_approach").astype(int)
        features["source_route_interval_sample"] = source.eq("route_interval_sample").astype(int)
        features["source_field_report_observed"] = source.eq(
            "field_report_observed_culvert"
        ).astype(int)
        features["source_field_observed_non_culvert"] = source.eq(
            "field_observed_non_culvert"
        ).astype(int)

    for column in ("road_name", "stream_name", "matched_route"):
        if column in features.columns:
            text = features[column].fillna("").astype(str).str.strip()
            features[f"has_{column}"] = (text != "").astype(int)

    for column in ("road_bridge", "road_tunnel", "stream_culvert", "stream_tunnel"):
        if column in features.columns:
            features[f"{column}_flag"] = _boolean_score(features[column])

    return features


def add_known_culvert_labels(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame,
    positive_radius_m: float,
) -> gpd.GeoDataFrame:
    labeled = candidates.copy()
    if known_culverts.empty:
        labeled["dist_to_known_culvert_m"] = np.nan
        labeled["is_culvert"] = 0
        return labeled

    known_union = known_culverts.geometry.unary_union
    labeled["dist_to_known_culvert_m"] = labeled.geometry.apply(
        lambda geom: float(geom.distance(known_union))
    )
    labeled["is_culvert"] = 0
    for known_geom in known_culverts.geometry:
        distances = labeled.geometry.distance(known_geom)
        nearby = distances[distances <= positive_radius_m]
        if nearby.empty:
            continue
        selected_index = _best_training_match(labeled.loc[nearby.index], nearby)
        labeled.at[selected_index, "is_culvert"] = 1
    labeled = add_nearest_known_culvert_metadata(labeled, known_culverts)
    return labeled


def _best_training_match(candidates: gpd.GeoDataFrame, distances: pd.Series):
    source_priority = _known_label_source_priority(candidates)
    order = pd.DataFrame(
        {
            "distance": distances,
            "source_priority": source_priority,
        },
        index=candidates.index,
    ).sort_values(["source_priority", "distance"], kind="mergesort")
    return order.index[0]


def _known_label_source_priority(candidates: gpd.GeoDataFrame) -> pd.Series:
    if "source" not in candidates.columns:
        return pd.Series(5, index=candidates.index)

    source = candidates["source"].fillna("").astype(str)
    priority = pd.Series(5, index=candidates.index)
    priority.loc[source == "exact_road_stream_intersection"] = 0
    priority.loc[source == "nearest_road_stream_approach"] = 1
    priority.loc[source == "route_interval_sample"] = 2
    priority.loc[source == "field_report_observed_culvert"] = 3
    priority.loc[source == "field_observed_non_culvert"] = 9
    return priority


def add_nearest_known_culvert_metadata(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    enriched = candidates.copy()
    metadata_columns = {
        "report_date": "nearest_field_report_date",
        "route": "nearest_field_report_route",
        "culvert_id": "nearest_field_report_culvert_id",
        "source_file": "nearest_field_report_source_file",
        "feedback_type": "nearest_field_feedback_type",
        "layout_source": "nearest_field_feedback_layout_source",
        "feedback_source": "nearest_field_feedback_source",
        "observation_id": "nearest_field_feedback_observation_id",
    }

    for output_column in metadata_columns.values():
        enriched[output_column] = ""

    if known_culverts.empty:
        return enriched

    known_reset = known_culverts.reset_index(drop=True)
    for row_index, geometry in enriched.geometry.items():
        distances = known_reset.geometry.distance(geometry)
        nearest_index = int(distances.idxmin())
        nearest = known_reset.iloc[nearest_index]
        for source_column, output_column in metadata_columns.items():
            if source_column in nearest.index and pd.notna(nearest[source_column]):
                enriched.at[row_index, output_column] = str(nearest[source_column])

    return enriched


def add_known_culvert_pattern_features(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame,
    positive_radius_m: float,
    radii_m: tuple[float, ...] = KNOWN_PATTERN_RADII_M,
) -> gpd.GeoDataFrame:
    """Add bounded support from approved known culvert neighborhoods.

    These columns describe context around already-approved culverts. They are
    useful for evidence ranking, but supervised training excludes them to avoid
    learning direct proximity to labels.
    """

    enriched = candidates.copy()
    radii = tuple(sorted(float(radius) for radius in radii_m if float(radius) > 0))
    for radius in radii:
        radius_label = _radius_label(radius)
        enriched[f"approved_known_culvert_count_{radius_label}m"] = 0
        enriched[f"approved_known_culvert_density_{radius_label}m_per_sqkm"] = 0.0
    enriched["nearest_known_culvert_distance_decay"] = 0.0
    enriched["nearest_known_route_match"] = 0
    enriched["nearest_known_source_abu"] = 0
    enriched["nearest_known_source_doc"] = 0
    enriched["approved_known_source_score"] = 0.0
    enriched["approved_known_culvert_corridor_score"] = 0.0
    enriched["approved_known_culvert_pattern_score"] = 0.0

    if known_culverts.empty:
        return enriched

    known_reset = known_culverts.reset_index(drop=True)
    known_routes = [_known_route_tokens(row) for _, row in known_reset.iterrows()]
    known_source_scores = [_known_source_score(row) for _, row in known_reset.iterrows()]
    known_is_abu = [_known_source_is_abu(row) for _, row in known_reset.iterrows()]
    known_is_doc = [_known_source_is_doc(row) for _, row in known_reset.iterrows()]

    count_values = {radius: [] for radius in radii}
    density_values = {radius: [] for radius in radii}
    decay_values: list[float] = []
    route_match_values: list[int] = []
    source_score_values: list[float] = []
    abu_values: list[int] = []
    doc_values: list[int] = []
    corridor_scores: list[float] = []

    positive_radius = max(0.0, float(positive_radius_m))
    for _, row in enriched.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            for radius in radii:
                count_values[radius].append(0)
                density_values[radius].append(0.0)
            decay_values.append(0.0)
            route_match_values.append(0)
            source_score_values.append(0.0)
            abu_values.append(0)
            doc_values.append(0)
            corridor_scores.append(0.0)
            continue

        distances = known_reset.geometry.distance(geometry)
        nearest_index = int(distances.idxmin())
        nearest_distance = float(distances.iloc[nearest_index])
        outside_known_match = nearest_distance > positive_radius

        for radius in radii:
            count = int((distances <= radius).sum())
            area_sqkm = np.pi * radius * radius / 1_000_000
            count_values[radius].append(count)
            density_values[radius].append(count / area_sqkm if area_sqkm else 0.0)

        distance_decay = (
            float(np.exp(-((nearest_distance - positive_radius) / 650.0)))
            if outside_known_match
            else 0.0
        )
        distance_decay = float(np.clip(distance_decay, 0.0, 1.0))
        candidate_routes = _candidate_route_tokens(row)
        route_match = int(bool(candidate_routes & known_routes[nearest_index]))
        source_score = known_source_scores[nearest_index]

        count_500 = int((distances <= 500.0).sum())
        count_1000 = int((distances <= 1000.0).sum())
        density_signal = min(
            (0.50 * min(count_500, 2) / 2.0)
            + (0.50 * min(count_1000, 4) / 4.0),
            1.0,
        )
        route_signal = distance_decay if route_match else 0.0
        corridor_score = (
            source_score
            * (0.55 * distance_decay + 0.25 * density_signal + 0.20 * route_signal)
            if outside_known_match
            else 0.0
        )

        decay_values.append(distance_decay)
        route_match_values.append(route_match)
        source_score_values.append(source_score)
        abu_values.append(int(known_is_abu[nearest_index]))
        doc_values.append(int(known_is_doc[nearest_index]))
        corridor_scores.append(float(np.clip(corridor_score, 0.0, 1.0)))

    for radius in radii:
        radius_label = _radius_label(radius)
        enriched[f"approved_known_culvert_count_{radius_label}m"] = count_values[radius]
        enriched[f"approved_known_culvert_density_{radius_label}m_per_sqkm"] = density_values[
            radius
        ]
    enriched["nearest_known_culvert_distance_decay"] = decay_values
    enriched["nearest_known_route_match"] = route_match_values
    enriched["nearest_known_source_abu"] = abu_values
    enriched["nearest_known_source_doc"] = doc_values
    enriched["approved_known_source_score"] = source_score_values
    enriched["approved_known_culvert_corridor_score"] = corridor_scores
    return add_known_culvert_pattern_score(enriched)


def add_known_culvert_pattern_score(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    enriched = points.copy()
    corridor = _numeric_score(enriched, "approved_known_culvert_corridor_score")
    dem_similarity = _numeric_score(enriched, "approved_known_dem_similarity_score")
    if corridor.max() <= 0 and dem_similarity.max() <= 0:
        if "approved_known_culvert_pattern_score" not in enriched.columns:
            enriched["approved_known_culvert_pattern_score"] = 0.0
        return enriched

    pattern = corridor.copy()
    if dem_similarity.max() > 0:
        pattern = (0.82 * corridor + 0.18 * corridor * dem_similarity).clip(0, 1)
    enriched["approved_known_culvert_pattern_score"] = pattern.fillna(0.0).clip(0, 1)
    return enriched


def add_field_route_feedback_features(
    candidates: gpd.GeoDataFrame,
    known_culverts: gpd.GeoDataFrame | None,
    negative_culverts: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame:
    """Learn route-level field precision after assigning feedback to its nearest road profile."""

    enriched = candidates.copy()
    output_columns = (
        "field_route_positive_reviews",
        "field_route_negative_reviews",
        "field_route_feedback_reviews",
        "field_route_feedback_precision",
        "field_route_positive_support",
        "field_route_negative_support",
        "field_route_feedback_score",
    )
    for column in output_columns:
        enriched[column] = 0.0

    route_keys = enriched.apply(_candidate_feedback_route_key, axis=1)
    route_reference = enriched[
        enriched.get("source", pd.Series("", index=enriched.index))
        .fillna("")
        .astype(str)
        .eq("route_interval_sample")
        & route_keys.ne("")
    ].copy()
    if route_reference.empty:
        return enriched
    route_reference["_field_route_key"] = route_keys.loc[route_reference.index]

    positives = _field_feedback_rows(known_culverts, positive=True)
    negatives = _field_feedback_rows(negative_culverts, positive=False)
    positive_assignments = _infer_feedback_route_assignments(positives, route_reference)
    negative_assignments = _infer_feedback_route_assignments(negatives, route_reference)
    if positive_assignments.empty and negative_assignments.empty:
        return enriched

    positive_counts = _weighted_feedback_counts(positive_assignments, positive=True)
    negative_counts = _weighted_feedback_counts(negative_assignments, positive=False)
    candidate_groups = route_keys[route_keys.ne("")].groupby(route_keys).groups

    for route_key, candidate_indices in candidate_groups.items():
        positive_rows = positive_assignments[
            positive_assignments["_field_route_key"].eq(route_key)
        ]
        negative_rows = negative_assignments[
            negative_assignments["_field_route_key"].eq(route_key)
        ]
        positive_count = float(positive_counts.get(route_key, 0.0))
        negative_count = float(negative_counts.get(route_key, 0.0))
        review_count = positive_count + negative_count
        if review_count <= 0:
            continue

        route_precision = (positive_count + 2.0) / (review_count + 4.0)
        indices = list(candidate_indices)
        positive_support = _feedback_distance_support(
            enriched.loc[indices].geometry,
            positive_rows,
            positive=True,
        )
        negative_support = _feedback_distance_support(
            enriched.loc[indices].geometry,
            negative_rows,
            positive=False,
        )
        support_total = positive_support + negative_support
        local_precision = (
            positive_support + 2.0 * route_precision
        ) / (support_total + 2.0)
        feedback_score = (0.65 * route_precision + 0.35 * local_precision).clip(0, 1)

        enriched.loc[indices, "field_route_positive_reviews"] = positive_count
        enriched.loc[indices, "field_route_negative_reviews"] = negative_count
        enriched.loc[indices, "field_route_feedback_reviews"] = review_count
        enriched.loc[indices, "field_route_feedback_precision"] = route_precision
        enriched.loc[indices, "field_route_positive_support"] = positive_support.to_numpy()
        enriched.loc[indices, "field_route_negative_support"] = negative_support.to_numpy()
        enriched.loc[indices, "field_route_feedback_score"] = feedback_score.to_numpy()

    return enriched


def add_negative_culvert_labels(
    candidates: gpd.GeoDataFrame,
    negative_culverts: gpd.GeoDataFrame,
    negative_radius_m: float,
) -> gpd.GeoDataFrame:
    labeled = candidates.copy()
    labeled["field_denied"] = 0
    labeled["dist_to_denied_culvert_m"] = np.nan
    labeled["nearest_denied_observation_id"] = ""
    labeled["nearest_denied_notes"] = ""
    labeled["nearest_denied_label"] = ""
    labeled["nearest_denied_feedback_type"] = ""
    labeled["nearest_denied_layout_source"] = ""

    if negative_culverts.empty:
        return labeled

    if "is_culvert" not in labeled.columns:
        labeled["is_culvert"] = 0

    negative_reset = negative_culverts.reset_index(drop=True)
    negative_by_candidate_id = {}
    if "candidate_id" in negative_reset.columns:
        for negative_index, candidate_id in (
            negative_reset["candidate_id"].fillna("").astype(str).items()
        ):
            if candidate_id:
                negative_by_candidate_id[candidate_id] = negative_reset.iloc[int(negative_index)]

    direct_feedback_observation_ids: set[str] = set()
    if {"source", "field_feedback_observation_id"}.issubset(labeled.columns):
        direct_rows = labeled[
            labeled["source"].fillna("").astype(str).eq("field_observed_non_culvert")
        ]
        direct_feedback_observation_ids = set(
            direct_rows["field_feedback_observation_id"].fillna("").astype(str)
        ) - {""}

    for row_index, geometry in labeled.geometry.items():
        distances = negative_reset.geometry.distance(geometry)
        nearest_index = int(distances.idxmin())
        distance = float(distances.iloc[nearest_index])
        nearest = negative_reset.iloc[nearest_index]
        labeled.at[row_index, "dist_to_denied_culvert_m"] = distance
        candidate_id = (
            str(labeled.at[row_index, "candidate_id"]) if "candidate_id" in labeled else ""
        )
        candidate_source = ""
        if "source" in labeled:
            source_value = labeled.at[row_index, "source"]
            candidate_source = "" if pd.isna(source_value) else str(source_value)
        direct_feedback_negative = candidate_source == "field_observed_non_culvert"
        exact_negative = negative_by_candidate_id.get(candidate_id)
        nearest_observation_value = nearest.get("observation_id", "")
        nearest_observation_id = (
            "" if pd.isna(nearest_observation_value) else str(nearest_observation_value)
        )
        has_direct_feedback_row = nearest_observation_id in direct_feedback_observation_ids
        if exact_negative is not None:
            nearest = exact_negative
            miss_distance_m = _optional_float(nearest.get("miss_distance_m"))
            if miss_distance_m is not None:
                labeled.at[row_index, "dist_to_denied_culvert_m"] = miss_distance_m
            mark_negative = True
        elif has_direct_feedback_row and not direct_feedback_negative:
            mark_negative = False
        else:
            mark_negative = distance <= negative_radius_m and (
                direct_feedback_negative or not _is_missed_prediction(nearest)
            )

        if mark_negative:
            _mark_negative_label(labeled, row_index, nearest)

    return labeled


def _mark_negative_label(labeled: gpd.GeoDataFrame, row_index, nearest: pd.Series) -> None:
    labeled.at[row_index, "field_denied"] = 1
    labeled.at[row_index, "is_culvert"] = 0
    if "observation_id" in nearest.index and pd.notna(nearest["observation_id"]):
        labeled.at[row_index, "nearest_denied_observation_id"] = str(nearest["observation_id"])
    if "notes" in nearest.index and pd.notna(nearest["notes"]):
        labeled.at[row_index, "nearest_denied_notes"] = str(nearest["notes"])
    metadata_columns = {
        "label": "nearest_denied_label",
        "feedback_type": "nearest_denied_feedback_type",
        "layout_source": "nearest_denied_layout_source",
    }
    for source_column, output_column in metadata_columns.items():
        if source_column in nearest.index and pd.notna(nearest[source_column]):
            labeled.at[row_index, output_column] = str(nearest[source_column])


def _is_missed_prediction(row: pd.Series) -> bool:
    return str(row.get("label", "") or "").strip() == "missed_prediction"


def _optional_float(value) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def add_raster_samples(
    points: gpd.GeoDataFrame,
    raster_path: str | Path,
    prefix: str,
) -> gpd.GeoDataFrame:
    import rasterio

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    enriched = points.copy()
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster is missing a CRS: {raster_path}")

        sample_points = enriched.to_crs(src.crs)
        value_col = "elevation_m" if prefix == "dem" else f"{prefix}_value"
        slope_col = "slope_degrees" if prefix == "dem" else f"{prefix}_local_slope_degrees"
        values: list[float] = []
        stats_by_window = {
            3: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
            9: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
            15: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
            31: {"slope": [], "mean": [], "relief": [], "std": [], "tpi": [], "valley_depth": []},
        }

        for point in sample_points.geometry:
            value = _sample_value(src, point.x, point.y)
            values.append(value)
            for window_size, output in stats_by_window.items():
                stats = _local_raster_stats(src, point.x, point.y, window_size=window_size)
                output["slope"].append(stats["slope_degrees"])
                output["mean"].append(stats["mean"])
                output["relief"].append(stats["relief"])
                output["std"].append(stats["std"])
                output["tpi"].append(stats["topographic_position"])
                output["valley_depth"].append(stats["valley_depth"])

        enriched[value_col] = values
        if prefix == "dem":
            enriched[slope_col] = stats_by_window[3]["slope"]
            for window_size, output in stats_by_window.items():
                enriched[f"elevation_mean_{window_size}x{window_size}_m"] = output["mean"]
                enriched[f"elevation_relief_{window_size}x{window_size}_m"] = output["relief"]
                enriched[f"terrain_roughness_{window_size}x{window_size}_m"] = output["std"]
                enriched[f"topographic_position_{window_size}x{window_size}_m"] = output["tpi"]
                enriched[f"valley_depth_{window_size}x{window_size}_m"] = output["valley_depth"]

    return enriched


def add_dem_hydrology_proxies(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    enriched = points.copy()
    if "slope_degrees" not in enriched.columns:
        return enriched

    slope = pd.to_numeric(enriched["slope_degrees"], errors="coerce").clip(lower=0)
    slope_damping = 1.0 / (1.0 + slope)
    for window_size in (3, 9, 15, 31):
        relief_col = f"elevation_relief_{window_size}x{window_size}_m"
        roughness_col = f"terrain_roughness_{window_size}x{window_size}_m"
        tpi_col = f"topographic_position_{window_size}x{window_size}_m"
        valley_col = f"valley_depth_{window_size}x{window_size}_m"
        if not {relief_col, roughness_col, tpi_col, valley_col}.issubset(enriched.columns):
            continue

        relief = pd.to_numeric(enriched[relief_col], errors="coerce").clip(lower=0)
        roughness = pd.to_numeric(enriched[roughness_col], errors="coerce").clip(lower=0)
        tpi = pd.to_numeric(enriched[tpi_col], errors="coerce")
        valley_depth = pd.to_numeric(enriched[valley_col], errors="coerce").clip(lower=0)

        enriched[f"valley_depth_relief_ratio_{window_size}x{window_size}"] = (
            valley_depth / relief.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        enriched[f"topographic_wetness_proxy_{window_size}x{window_size}"] = (
            np.log1p(valley_depth) * slope_damping
        )
        enriched[f"low_slope_valley_score_{window_size}x{window_size}"] = (
            _robust_0_to_1(valley_depth) * slope_damping
        )
        enriched[f"terrain_break_score_proxy_{window_size}x{window_size}"] = np.log1p(
            relief
        ) * np.log1p(roughness)
        enriched[f"negative_tpi_{window_size}x{window_size}_m"] = (-tpi).clip(lower=0)

    return enriched


def add_dem_culvert_terrain_features(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build label-free terrain composites from DEM local statistics."""

    enriched = points.copy()
    valley_scores = []
    terrain_scores = []
    for window_size in (3, 9, 15, 31):
        valley_pieces = []
        for column in (
            f"valley_depth_{window_size}x{window_size}_m",
            f"negative_tpi_{window_size}x{window_size}_m",
            f"topographic_wetness_proxy_{window_size}x{window_size}",
        ):
            if column in enriched.columns:
                valley_pieces.append(_robust_0_to_1(enriched[column]))
        low_slope_col = f"low_slope_valley_score_{window_size}x{window_size}"
        if low_slope_col in enriched.columns:
            valley_pieces.append(_numeric_score(enriched, low_slope_col))

        valley_col = f"dem_valley_position_score_{window_size}x{window_size}"
        enriched[valley_col] = _mean_numeric_pieces(enriched, valley_pieces)
        valley_scores.append(enriched[valley_col])

        terrain_pieces = []
        for column in (
            f"terrain_break_score_proxy_{window_size}x{window_size}",
            f"elevation_relief_{window_size}x{window_size}_m",
            f"terrain_roughness_{window_size}x{window_size}_m",
        ):
            if column in enriched.columns:
                terrain_pieces.append(_robust_0_to_1(enriched[column]))

        terrain_col = f"dem_terrain_break_score_{window_size}x{window_size}"
        enriched[terrain_col] = _mean_numeric_pieces(enriched, terrain_pieces)
        terrain_scores.append(enriched[terrain_col])

    enriched["dem_valley_position_score"] = _mean_numeric_pieces(enriched, valley_scores)
    enriched["dem_terrain_break_score"] = _mean_numeric_pieces(enriched, terrain_scores)
    enriched["dem_culvert_terrain_score"] = (
        0.65 * enriched["dem_valley_position_score"]
        + 0.35 * enriched["dem_terrain_break_score"]
    ).clip(0, 1)
    return enriched


def add_route_profile_dem_features(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find DEM depressions along sampled roads instead of at arbitrary map intervals."""

    enriched = points.copy()
    enriched["route_elevation_dip_60m"] = 0.0
    enriched["route_elevation_dip_rank_pct"] = 0.0
    enriched["route_low_point_score"] = 0.0
    required = {"source", "route_sample_distance_m", "elevation_m"}
    if not required.issubset(enriched.columns):
        return enriched

    route_mask = enriched["source"].fillna("").astype(str).eq("route_interval_sample")
    route = enriched[route_mask]
    if route.empty:
        return enriched

    profile_keys = route.apply(_route_profile_key, axis=1)
    sample_distances = pd.to_numeric(route["route_sample_distance_m"], errors="coerce")
    elevations = pd.to_numeric(route["elevation_m"], errors="coerce")

    for profile_key, profile_indices in profile_keys.groupby(profile_keys).groups.items():
        finite_indices = [
            index
            for index in profile_indices
            if pd.notna(sample_distances.loc[index]) and pd.notna(elevations.loc[index])
        ]
        if len(finite_indices) < 3:
            continue
        ordered = sorted(finite_indices, key=lambda index: float(sample_distances.loc[index]))
        distances = np.asarray([float(sample_distances.loc[index]) for index in ordered])
        profile_elevations = np.asarray([float(elevations.loc[index]) for index in ordered])
        dips = _route_profile_dips(
            distances,
            profile_elevations,
            window_m=ROUTE_PROFILE_DIP_WINDOW_M,
            shoulder_m=ROUTE_PROFILE_DIP_SHOULDER_M,
        )
        dip_series = pd.Series(dips, index=ordered)
        positive_dips = dip_series[dip_series > 0]
        ranks = pd.Series(0.0, index=ordered)
        if not positive_dips.empty:
            ranks.loc[positive_dips.index] = positive_dips.rank(pct=True)
        absolute_score = 1.0 - np.exp(-dip_series.clip(lower=0.0) / 0.75)
        low_point_score = (
            0.70 * absolute_score + 0.30 * ranks
        ).where(dip_series >= 0.15, 0.0).clip(0, 1)

        enriched.loc[ordered, "route_elevation_dip_60m"] = dip_series.to_numpy()
        enriched.loc[ordered, "route_elevation_dip_rank_pct"] = ranks.to_numpy()
        enriched.loc[ordered, "route_low_point_score"] = low_point_score.to_numpy()

    return enriched


def _route_profile_dips(
    distances: np.ndarray,
    elevations: np.ndarray,
    window_m: float,
    shoulder_m: float,
) -> np.ndarray:
    dips = np.zeros(len(distances), dtype=float)
    for position, (distance, elevation) in enumerate(zip(distances, elevations)):
        left_start = int(np.searchsorted(distances, distance - window_m, side="left"))
        left_end = int(np.searchsorted(distances, distance - shoulder_m, side="right"))
        right_start = int(np.searchsorted(distances, distance + shoulder_m, side="left"))
        right_end = int(np.searchsorted(distances, distance + window_m, side="right"))
        left = elevations[left_start:left_end]
        right = elevations[right_start:right_end]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if not len(left) or not len(right) or not np.isfinite(elevation):
            continue
        shoulder_elevation = min(float(np.max(left)), float(np.max(right)))
        dips[position] = max(0.0, shoulder_elevation - float(elevation))
    return dips


def add_approved_known_dem_similarity_features(
    points: gpd.GeoDataFrame,
    positive_radius_m: float,
) -> gpd.GeoDataFrame:
    """Compare candidate DEM terrain to the approved known culvert terrain profile."""

    enriched = points.copy()
    enriched["approved_known_dem_similarity_score"] = 0.0
    if "is_culvert" not in enriched.columns:
        return enriched

    pattern_columns = [
        column
        for column in (
            "dem_culvert_terrain_score",
            "dem_valley_position_score",
            "dem_terrain_break_score",
            "dem_valley_position_score_9x9",
            "dem_valley_position_score_15x15",
            "dem_valley_position_score_31x31",
            "dem_terrain_break_score_9x9",
            "dem_terrain_break_score_15x15",
            "dem_terrain_break_score_31x31",
        )
        if column in enriched.columns
    ]
    if not pattern_columns:
        return enriched

    known = pd.to_numeric(enriched["is_culvert"], errors="coerce").fillna(0).astype(int) == 1
    known_values = enriched.loc[known, pattern_columns].apply(pd.to_numeric, errors="coerce")
    known_values = known_values.dropna(how="all")
    if known_values.empty:
        return enriched

    profile = known_values.median(axis=0)
    candidate_values = enriched[pattern_columns].apply(pd.to_numeric, errors="coerce").fillna(
        profile
    )
    difference = (candidate_values - profile).abs().mean(axis=1)
    similarity = (1.0 - difference).clip(0, 1).fillna(0.0)

    if "dist_to_known_culvert_m" in enriched.columns:
        distance = pd.to_numeric(enriched["dist_to_known_culvert_m"], errors="coerce")
        similarity = similarity.where(distance > float(positive_radius_m), 0.0)

    enriched["approved_known_dem_similarity_score"] = similarity
    return enriched


def add_hydrology_raster_features(points: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
    enriched = points.copy()
    value_col = f"{prefix}_value"
    if value_col not in enriched.columns:
        return enriched

    values = pd.to_numeric(enriched[value_col], errors="coerce").clip(lower=0)
    enriched[f"{prefix}_log"] = np.log1p(values)
    enriched[f"{prefix}_rank_pct"] = values.rank(pct=True).fillna(0.0)
    return enriched


def add_training_sample_weights(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make explicit field feedback outweigh unlabeled background samples."""

    weighted = points.copy()
    weights = pd.Series(0.05, index=weighted.index, dtype=float)

    is_positive = _numeric_flag(weighted, "is_culvert")
    is_denied = _numeric_flag(weighted, "field_denied")
    is_feedback_positive = is_positive & _field_observation_match(weighted)
    is_field_positive = is_positive & ~is_feedback_positive
    is_abu_positive = is_feedback_positive & _feedback_type_match(weighted, "abu")
    is_cbu_positive = is_feedback_positive & _feedback_type_match(weighted, "cbu")
    is_mvd_positive = is_feedback_positive & _feedback_type_match(weighted, "mvd")
    is_missed_negative = is_denied & (
        _feedback_type_match(weighted, "mvd_origin")
        | _string_contains(weighted, "nearest_denied_notes", "confirmed culvert was")
    )

    weights.loc[is_field_positive] = 6.0
    weights.loc[is_feedback_positive] = 18.0
    weights.loc[is_abu_positive] = 24.0
    weights.loc[is_cbu_positive] = 20.0
    weights.loc[is_mvd_positive] = 32.0
    weights.loc[is_denied] = 24.0
    weights.loc[is_missed_negative] = 32.0

    weighted["training_sample_weight"] = weights
    return weighted


def _numeric_flag(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return pd.to_numeric(table[column], errors="coerce").fillna(0).astype(int) == 1


def _field_observation_match(table: pd.DataFrame) -> pd.Series:
    source_columns = [
        column
        for column in ("field_report_source_file", "nearest_field_report_source_file")
        if column in table.columns
    ]
    if not source_columns:
        return pd.Series(False, index=table.index)

    result = pd.Series(False, index=table.index)
    for column in source_columns:
        result |= (
            table[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains("field_observations.geojson", regex=False)
        )
    return result


def _string_contains(table: pd.DataFrame, column: str, needle: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return table[column].fillna("").astype(str).str.contains(needle, case=False, regex=False)


def _feedback_type_match(table: pd.DataFrame, feedback_type: str) -> pd.Series:
    result = pd.Series(False, index=table.index)
    for column in (
        "field_feedback_type",
        "nearest_field_feedback_type",
        "nearest_denied_feedback_type",
    ):
        if column in table.columns:
            result |= table[column].fillna("").astype(str).str.lower().eq(feedback_type)
    return result


def _density_radii(
    base_radius_m: float, extra_radii_m: tuple[float, ...] | None
) -> tuple[float, ...]:
    radii = {float(base_radius_m), 50.0, 100.0, 250.0}
    if extra_radii_m:
        radii.update(float(radius) for radius in extra_radii_m)
    return tuple(sorted(radius for radius in radii if radius > 0))


def _density_column(layer: str, radius_m: float) -> str:
    radius_label = int(radius_m) if float(radius_m).is_integer() else radius_m
    return f"{layer}_density_{radius_label}m_m_per_sqkm"


def _radius_label(radius_m: float) -> str:
    return str(int(radius_m)) if float(radius_m).is_integer() else str(radius_m).replace(".", "_")


def _nearest_distance(points: Iterable, targets: gpd.GeoDataFrame) -> list[float]:
    if targets.empty:
        return [np.nan for _point in points]
    target_union = targets.geometry.unary_union
    return [float(point.distance(target_union)) for point in points]


def _line_density(points: Iterable, lines: gpd.GeoDataFrame, radius_m: float) -> list[float]:
    area_sqkm = np.pi * radius_m * radius_m / 1_000_000
    densities = []
    line_geometries = lines.geometry.to_numpy()

    for point in points:
        buffer = point.buffer(radius_m)
        total_length_m = 0.0
        for position in _query_positions(lines, buffer):
            segment = line_geometries[int(position)].intersection(buffer)
            if not segment.is_empty:
                total_length_m += float(segment.length)
        densities.append(total_length_m / area_sqkm if area_sqkm else 0.0)

    return densities


def _query_positions(gdf: gpd.GeoDataFrame, geometry) -> list[int]:
    try:
        return list(gdf.sindex.query(geometry, predicate="intersects"))
    except Exception:
        return list(range(len(gdf)))


def _boolean_score(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .isin({"1", "true", "yes", "y", "bridge", "tunnel", "culvert", "covered"})
        .astype(int)
    )


def _robust_0_to_1(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    low = numeric.quantile(0.05)
    high = numeric.quantile(0.95)
    if pd.isna(low) or pd.isna(high) or high <= low:
        return pd.Series(0.0, index=values.index)
    return ((numeric - low) / (high - low)).clip(0, 1).fillna(0.0)


def _numeric_score(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(0.0, index=table.index)
    return pd.to_numeric(table[column], errors="coerce").fillna(0.0).clip(0, 1)


def _mean_numeric_pieces(table: pd.DataFrame, pieces: list[pd.Series]) -> pd.Series:
    if not pieces:
        return pd.Series(0.0, index=table.index)
    stacked = pd.concat(pieces, axis=1)
    return stacked.mean(axis=1).fillna(0.0).clip(0, 1)


def _candidate_route_tokens(row: pd.Series) -> set[str]:
    tokens: set[str] = set()
    for column in ("matched_route", "road_name", "road_id", "nearest_field_report_route"):
        if column in row.index and pd.notna(row[column]):
            tokens |= _route_tokens_from_value(row[column])
    return tokens


def _candidate_feedback_route_key(row: pd.Series) -> str:
    tokens: set[str] = set()
    for column in ("matched_route", "road_name"):
        if column in row.index and pd.notna(row[column]):
            tokens |= _route_tokens_from_value(row[column])
    prefixed = sorted(token for token in tokens if re.search(r"[A-Z]", token))
    if prefixed:
        return prefixed[0]
    return sorted(tokens)[0] if tokens else ""


def _route_profile_key(row: pd.Series) -> str:
    road_id = _clean_key_value(row.get("road_id", ""))
    route_key = _candidate_feedback_route_key(row)
    part = _clean_key_value(row.get("route_part_index", 0)) or "0"
    offset = _clean_key_value(row.get("route_lateral_offset_m", 0)) or "0"
    return f"{road_id or route_key}|part:{part}|offset:{offset}"


def _clean_key_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    return " ".join(text.split())


def _field_feedback_rows(
    points: gpd.GeoDataFrame | None,
    positive: bool,
) -> gpd.GeoDataFrame:
    if points is None or points.empty:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(points, "crs", None))
    feedback = points.copy()
    if positive:
        field_mask = pd.Series(False, index=feedback.index)
        if "feedback_type" in feedback.columns:
            field_mask |= (
                feedback["feedback_type"]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin({"abu", "cbu", "mvd"})
            )
        for column in ("observation_id", "field_culvert_id"):
            if column in feedback.columns:
                field_mask |= feedback[column].fillna("").astype(str).str.strip().ne("")
        if "source_file" in feedback.columns:
            field_mask |= (
                feedback["source_file"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains("field_observations.geojson", regex=False)
            )
        feedback = feedback[field_mask]
    return feedback.reset_index(drop=True)


def _infer_feedback_route_assignments(
    feedback: gpd.GeoDataFrame,
    route_reference: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if feedback.empty:
        return gpd.GeoDataFrame(geometry=[], crs=route_reference.crs)

    reference_columns = [
        "_field_route_key",
        "geometry",
    ]
    reference = route_reference[reference_columns].reset_index(drop=True)
    points = feedback.copy().reset_index(drop=True)
    points["_feedback_row_id"] = np.arange(len(points))
    joined = gpd.sjoin_nearest(
        points,
        reference,
        how="inner",
        max_distance=FIELD_ROUTE_INFERENCE_MAX_DISTANCE_M,
        distance_col="_field_route_distance_m",
    )
    if joined.empty:
        return joined
    return (
        joined.sort_values(["_feedback_row_id", "_field_route_distance_m"])
        .drop_duplicates("_feedback_row_id")
        .reset_index(drop=True)
    )


def _weighted_feedback_counts(
    assignments: gpd.GeoDataFrame,
    positive: bool,
) -> dict[str, float]:
    if assignments.empty:
        return {}
    weights = _feedback_assignment_weights(assignments, positive=positive)
    if not positive:
        feedback_type = assignments.get(
            "feedback_type",
            pd.Series("", index=assignments.index),
        ).fillna("").astype(str).str.lower()
        weights = weights.where(~feedback_type.eq("mvd_origin"), 0.25)
    return (
        pd.DataFrame(
            {
                "route_key": assignments["_field_route_key"].astype(str),
                "weight": weights,
            }
        )
        .groupby("route_key")["weight"]
        .sum()
        .to_dict()
    )


def _feedback_distance_support(
    candidate_geometry: gpd.GeoSeries,
    assignments: gpd.GeoDataFrame,
    positive: bool,
) -> pd.Series:
    support = pd.Series(0.0, index=candidate_geometry.index)
    if assignments.empty:
        return support
    weights = _feedback_assignment_weights(assignments, positive=positive)
    for (_, feedback), weight in zip(assignments.iterrows(), weights):
        distance = candidate_geometry.distance(feedback.geometry)
        contribution = np.exp(
            -distance / FIELD_ROUTE_LOCAL_SUPPORT_DECAY_M
        ).where(distance <= FIELD_ROUTE_LOCAL_SUPPORT_MAX_DISTANCE_M, 0.0)
        support += contribution.fillna(0.0) * float(weight)
    return support


def _feedback_assignment_weights(
    assignments: gpd.GeoDataFrame,
    positive: bool,
) -> pd.Series:
    feedback_type = assignments.get(
        "feedback_type",
        pd.Series("", index=assignments.index),
    ).fillna("").astype(str).str.lower()
    emphasized = {"mvd"} if positive else {"mvd_origin"}
    return feedback_type.isin(emphasized).map({True: 1.25, False: 1.0}).astype(float)


def _known_route_tokens(row: pd.Series) -> set[str]:
    tokens: set[str] = set()
    for column in ("route", "road_name", "culvert_id", "field_culvert_id"):
        if column in row.index and pd.notna(row[column]):
            tokens |= _route_tokens_from_value(row[column])
    return tokens


def _route_tokens_from_value(value) -> set[str]:
    text = _normalized_route_text(value)
    tokens = set()
    for match in ROUTE_TOKEN_RE.finditer(text):
        number = match.group("number").upper()
        tokens.add(f"{match.group('prefix').upper()}{number}")
        tokens.add(number)
    for match in GENERIC_ROUTE_TOKEN_RE.finditer(text):
        tokens.add(match.group("number").upper())
    if tokens:
        return tokens

    bare = re.fullmatch(r"\s*(\d+[A-Z]?)\s*", text)
    return {bare.group(1).upper()} if bare else set()


def _normalized_route_text(value) -> str:
    text = str(value or "").upper()
    replacements = [
        (r"\bU\.S\.\b", "US"),
        (r"\bUS\s+ROUTE\b", "US"),
        (r"\bUNITED\s+STATES\s+ROUTE\b", "US"),
        (r"\bINTERSTATE\b", "I"),
        (r"\bI\s*-\s*", "I"),
        (r"\bSTATE\s+(?:RTE|RT|ROUTE)\b", "NY"),
        (r"\bNYS\s+(?:RTE|RT|ROUTE)\b", "NY"),
        (r"\bNEW\s+YORK\s+(?:RTE|RT|ROUTE)\b", "NY"),
        (r"\bCOUNTY\s+(?:ROAD|ROUTE|RTE|RT)\b", "CR"),
        (r"\bCO\s*(?:RD|RTE|RT)\b", "CR"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def _known_source_score(row: pd.Series) -> float:
    confidence = (
        _optional_float(row.get("label_confidence"))
        if "label_confidence" in row.index
        else None
    )
    if _known_source_is_abu(row):
        return 1.0
    if _known_source_is_doc(row):
        return float(np.clip(max(confidence or 0.0, 0.85), 0.0, 1.0))
    return float(np.clip(confidence if confidence is not None else 0.80, 0.0, 1.0))


def _known_source_is_abu(row: pd.Series) -> bool:
    text = " ".join(
        str(row.get(column, "") or "")
        for column in ("source_file", "label", "observation_id", "field_culvert_id")
    ).lower()
    return (
        "field_observations.geojson" in text
        or "confirmed_field_observation" in text
        or str(row.get("field_culvert_id", "") or "").strip() != ""
        or str(row.get("observation_id", "") or "").strip() != ""
    )


def _known_source_is_doc(row: pd.Series) -> bool:
    if _known_source_is_abu(row):
        return False
    source_file = str(row.get("source_file", "") or "").strip()
    return source_file != ""


def _sample_value(src, x: float, y: float) -> float:
    try:
        value = next(src.sample([(x, y)]))[0]
    except Exception:
        return np.nan

    if src.nodata is not None and value == src.nodata:
        return np.nan
    return float(value)


def _local_raster_stats(src, x: float, y: float, window_size: int = 3) -> dict[str, float]:
    try:
        row, col = src.index(x, y)
    except Exception:
        return _empty_raster_stats()

    half = window_size // 2
    if row < half or col < half or row >= src.height - half or col >= src.width - half:
        return _empty_raster_stats()

    window = Window(col - half, row - half, window_size, window_size)
    data = src.read(1, window=window, masked=True).astype(float)
    if data.shape != (window_size, window_size):
        return _empty_raster_stats()

    filled = data.filled(np.nan)
    if np.isnan(filled).any():
        return _empty_raster_stats()

    center = float(filled[half, half])
    mean = float(np.mean(filled))
    xres, yres = _raster_resolution_m(src, x, y)
    dz_dy, dz_dx = np.gradient(filled, yres, xres)
    rise_run = np.sqrt(dz_dx[half, half] ** 2 + dz_dy[half, half] ** 2)
    return {
        "slope_degrees": float(np.degrees(np.arctan(rise_run))),
        "mean": mean,
        "relief": float(np.max(filled) - np.min(filled)),
        "std": float(np.std(filled)),
        "topographic_position": center - mean,
        "valley_depth": max(0.0, mean - center),
    }


def _raster_resolution_m(src, x: float, y: float) -> tuple[float, float]:
    xres = abs(float(src.transform.a))
    yres = abs(float(src.transform.e))
    if src.crs and src.crs.is_geographic:
        latitude = np.radians(float(y))
        meters_per_degree_latitude = (
            111132.92
            - 559.82 * np.cos(2 * latitude)
            + 1.175 * np.cos(4 * latitude)
        )
        meters_per_degree_longitude = (
            111412.84 * np.cos(latitude)
            - 93.5 * np.cos(3 * latitude)
        )
        return (
            max(xres * abs(float(meters_per_degree_longitude)), 1e-6),
            max(yres * abs(float(meters_per_degree_latitude)), 1e-6),
        )

    unit_factor = 1.0
    try:
        linear_units_factor = src.crs.linear_units_factor
        unit_factor = float(
            linear_units_factor[1]
            if isinstance(linear_units_factor, tuple)
            else linear_units_factor
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return max(xres * unit_factor, 1e-6), max(yres * unit_factor, 1e-6)


def _empty_raster_stats() -> dict[str, float]:
    return {
        "slope_degrees": np.nan,
        "mean": np.nan,
        "relief": np.nan,
        "std": np.nan,
        "topographic_position": np.nan,
        "valley_depth": np.nan,
    }
