from __future__ import annotations

from html import escape
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from culvert_ai.io import ensure_parent_dir


DEFAULT_WEIGHTS = {
    "road_stream_proximity_score": 0.25,
    "drainage_strength_score": 0.20,
    "valley_position_score": 0.16,
    "crossing_geometry_score": 0.10,
    "terrain_break_score": 0.13,
    "road_context_score": 0.10,
    "osm_culvert_tag_score": 0.06,
    "field_report_support_score": 0.08,
}


def score_unlabeled_candidates(
    features: gpd.GeoDataFrame,
    weights: dict[str, float] | None = None,
) -> gpd.GeoDataFrame:
    """Rank likely culvert locations without local known culvert labels.

    This is not a supervised model. It is an expert/weak-supervision score designed for the
    real-world case where crews do not yet know where the culverts are.
    """

    weights = weights or DEFAULT_WEIGHTS
    scored = features.copy()

    scored["road_stream_proximity_score"] = _road_stream_proximity_score(scored)
    scored["drainage_strength_score"] = _drainage_strength_score(scored)
    scored["valley_position_score"] = _valley_position_score(scored)
    scored["crossing_geometry_score"] = _crossing_geometry_score(scored)
    scored["terrain_break_score"] = _terrain_break_score(scored)
    scored["road_context_score"] = _road_context_score(scored)
    scored["osm_culvert_tag_score"] = _osm_culvert_tag_score(scored)
    scored["field_report_support_score"] = _field_report_support_score(scored)
    scored["non_culvert_structure_penalty"] = _non_culvert_structure_penalty(scored)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Scoring weights must sum to a positive value.")

    score = np.zeros(len(scored), dtype=float)
    for column, weight in weights.items():
        if column not in scored.columns:
            continue
        score += scored[column].fillna(0.0).astype(float).clip(0, 1) * float(weight)

    scored["culvert_likelihood_score"] = (100 * score / total_weight) - (
        20 * scored["non_culvert_structure_penalty"]
    )
    scored["culvert_likelihood_score"] = scored["culvert_likelihood_score"].clip(0, 100)
    if "is_culvert" in scored.columns:
        known = pd.to_numeric(scored["is_culvert"], errors="coerce").fillna(0).astype(int) == 1
        scored.loc[known, "culvert_likelihood_score"] = scored.loc[
            known, "culvert_likelihood_score"
        ].clip(lower=95)
    scored = scored.sort_values("culvert_likelihood_score", ascending=False).reset_index(drop=True)
    scored["priority_rank"] = np.arange(1, len(scored) + 1)
    scored["priority_percentile"] = 1.0 - ((scored["priority_rank"] - 1) / max(len(scored), 1))
    scored["priority_bucket"] = pd.cut(
        scored["culvert_likelihood_score"],
        bins=[-0.01, 35, 55, 75, 100],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    scored["evidence_summary"] = scored.apply(_evidence_summary, axis=1)
    scored["google_earth_url"] = scored.apply(_google_earth_url, axis=1)
    return scored


def build_discovery_ranking(
    evidence_predictions: gpd.GeoDataFrame,
    supervised_predictions: gpd.GeoDataFrame | None = None,
    evidence_weight: float = 0.40,
    model_weight: float = 0.60,
    known_radius_m: float = 75.0,
) -> gpd.GeoDataFrame:
    """Create a field-work ranking that prioritizes not-yet-observed candidates.

    Field reports are training labels and validation evidence, but they should not dominate the
    work queue for crews. This ranking keeps known matches visible while ranking unknown locations
    first by a blend of model probability and interpretable GIS evidence.
    """

    if evidence_predictions.empty:
        raise ValueError("Evidence prediction table has no rows.")
    if evidence_weight < 0 or model_weight < 0 or evidence_weight + model_weight <= 0:
        raise ValueError("Discovery ranking weights must be non-negative and sum above zero.")

    ranked = evidence_predictions.copy()
    if supervised_predictions is not None and not supervised_predictions.empty:
        ranked = _attach_supervised_probability(ranked, supervised_predictions)

    evidence_score = _score_0_to_1(ranked, "culvert_likelihood_score", scale=100.0)
    model_probability = _score_0_to_1(ranked, "culvert_probability", scale=1.0)
    model_rank_score = _model_rank_score(model_probability)
    has_model = model_rank_score.notna()

    blended = evidence_score.copy()
    total_weight = evidence_weight + model_weight
    weighted_signal = (
        evidence_weight * evidence_score.loc[has_model] + model_weight * model_rank_score.loc[has_model]
    ) / total_weight
    agreement_signal = np.sqrt(
        evidence_score.loc[has_model].clip(0, 1) * model_rank_score.loc[has_model].clip(0, 1)
    )
    blended.loc[has_model] = (
        0.55 * agreement_signal + 0.25 * evidence_score.loc[has_model] + 0.20 * weighted_signal
    ).clip(0, 1)

    known = _known_field_match_mask(ranked, known_radius_m=known_radius_m)
    ranked["is_known_field_match"] = known.astype(int)
    ranked["discovery_status"] = np.where(known, "known_field_match", "undiscovered_candidate")
    ranked["evidence_score"] = (evidence_score.fillna(0.0) * 100).clip(0, 100)
    ranked["model_probability_score"] = (model_probability.fillna(0.0) * 100).clip(0, 100)
    ranked["model_rank_score"] = (model_rank_score.fillna(0.0) * 100).clip(0, 100)
    ranked["discovery_score"] = (blended.fillna(evidence_score).fillna(0.0) * 100).clip(0, 100)

    sort_table = ranked.assign(_known_sort=known.astype(int))
    sort_table = sort_table.sort_values(
        ["_known_sort", "discovery_score", "culvert_likelihood_score"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["_known_sort"])

    sort_table = sort_table.reset_index(drop=True)
    sort_table["discovery_rank"] = np.arange(1, len(sort_table) + 1)
    sort_table["priority_rank"] = sort_table["discovery_rank"]
    sort_table["priority_percentile"] = 1.0 - (
        (sort_table["priority_rank"] - 1) / max(len(sort_table), 1)
    )
    sort_table["priority_bucket"] = pd.cut(
        sort_table["discovery_score"],
        bins=[-0.01, 35, 55, 75, 100],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    sort_table["evidence_summary"] = sort_table.apply(_discovery_evidence_summary, axis=1)
    if "google_earth_url" not in sort_table.columns:
        sort_table["google_earth_url"] = sort_table.apply(_google_earth_url, axis=1)
    return sort_table


def write_google_earth_kml(
    scored: gpd.GeoDataFrame,
    output_path: str | Path,
    max_points: int = 250,
) -> None:
    output_path = Path(output_path)
    ensure_parent_dir(output_path)
    wgs84 = scored.to_crs("EPSG:4326").head(max_points)

    placemarks = []
    for _, row in wgs84.iterrows():
        name = f"Rank {int(row['priority_rank'])}: {row.get('priority_bucket', 'candidate')}"
        score = row.get("discovery_score", row.get("culvert_likelihood_score", 0))
        status = row.get("discovery_status", "candidate")
        description = (
            f"Score: {score:.1f}<br/>"
            f"Status: {escape(str(status))}<br/>"
            f"Evidence: {escape(str(row.get('evidence_summary', '')))}<br/>"
            f"Road: {escape(str(row.get('road_name', 'unknown')))}<br/>"
            f"Stream/drainage: {escape(str(row.get('stream_name', 'unknown')))}"
        )
        point = row.geometry
        placemarks.append(
            "\n".join(
                [
                    "    <Placemark>",
                    f"      <name>{escape(name)}</name>",
                    f"      <description>{description}</description>",
                    "      <Point>",
                    f"        <coordinates>{point.x:.8f},{point.y:.8f},0</coordinates>",
                    "      </Point>",
                    "    </Placemark>",
                ]
            )
        )

    kml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "  <Document>",
            "    <name>Culvert AI Priority Review</name>",
            *placemarks,
            "  </Document>",
            "</kml>",
        ]
    )
    output_path.write_text(kml, encoding="utf-8")


def _road_stream_proximity_score(table: pd.DataFrame) -> pd.Series:
    if "road_stream_distance_m" not in table.columns:
        return _zero(table)

    distance = table["road_stream_distance_m"].astype(float).clip(lower=0)
    distance_score = 1.0 / (1.0 + distance / 20.0)
    exact = table.get("is_exact_road_stream_intersection", pd.Series(0, index=table.index)).fillna(0)
    return (0.75 * distance_score + 0.25 * exact.astype(float)).clip(0, 1)


def _drainage_strength_score(table: pd.DataFrame) -> pd.Series:
    pieces = []
    if "stream_order" in table.columns:
        pieces.append(_percentile(table["stream_order"]))
    for column in (
        "flow_accumulation_log",
        "flow_accumulation_rank_pct",
        "drainage_area_log",
        "drainage_area_rank_pct",
        "stream_density_100m_m_per_sqkm",
        "stream_density_250m_m_per_sqkm",
        "stream_density_500m_m_per_sqkm",
        "stream_density_m_per_sqkm",
    ):
        if column in table.columns:
            pieces.append(_percentile(table[column]))
    return _mean_score(table, pieces)


def _valley_position_score(table: pd.DataFrame) -> pd.Series:
    pieces = []
    if "valley_depth_9x9_m" in table.columns:
        pieces.append(_percentile(table["valley_depth_9x9_m"]))
    if "topographic_position_9x9_m" in table.columns:
        pieces.append(_inverse_percentile(table["topographic_position_9x9_m"]))
    if "topographic_wetness_proxy_9x9" in table.columns:
        pieces.append(_percentile(table["topographic_wetness_proxy_9x9"]))
    if "low_slope_valley_score_9x9" in table.columns:
        pieces.append(table["low_slope_valley_score_9x9"].fillna(0).clip(0, 1))
    if "valley_depth_15x15_m" in table.columns:
        pieces.append(_percentile(table["valley_depth_15x15_m"]))
    if "topographic_position_15x15_m" in table.columns:
        pieces.append(_inverse_percentile(table["topographic_position_15x15_m"]))
    if "topographic_wetness_proxy_15x15" in table.columns:
        pieces.append(_percentile(table["topographic_wetness_proxy_15x15"]))
    if "negative_tpi_31x31_m" in table.columns:
        pieces.append(_percentile(table["negative_tpi_31x31_m"]))
    return _mean_score(table, pieces)


def _crossing_geometry_score(table: pd.DataFrame) -> pd.Series:
    if "crossing_angle_degrees" not in table.columns:
        return _zero(table)
    angle = table["crossing_angle_degrees"].astype(float)
    return (1.0 - ((90.0 - angle).abs() / 90.0)).clip(0, 1).fillna(0)


def _terrain_break_score(table: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in (
        "slope_degrees",
        "elevation_relief_3x3_m",
        "terrain_roughness_3x3_m",
        "elevation_relief_9x9_m",
        "terrain_roughness_9x9_m",
        "terrain_break_score_proxy_9x9",
        "elevation_relief_15x15_m",
        "terrain_roughness_15x15_m",
        "terrain_break_score_proxy_15x15",
        "elevation_relief_31x31_m",
        "terrain_roughness_31x31_m",
        "terrain_break_score_proxy_31x31",
    ):
        if column in table.columns:
            pieces.append(_percentile(table[column]))
    return _mean_score(table, pieces)


def _road_context_score(table: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in (
        "road_density_100m_m_per_sqkm",
        "road_density_250m_m_per_sqkm",
        "road_density_500m_m_per_sqkm",
        "road_density_m_per_sqkm",
    ):
        if column in table.columns:
            pieces.append(_percentile(table[column]))
    return _mean_score(table, pieces)


def _evidence_summary(row: pd.Series) -> str:
    evidence = []
    thresholds = [
        ("road_stream_proximity_score", "road-drainage crossing"),
        ("drainage_strength_score", "strong drainage signal"),
        ("valley_position_score", "valley/low-point terrain"),
        ("crossing_geometry_score", "culvert-like crossing angle"),
        ("terrain_break_score", "terrain break or relief"),
        ("road_context_score", "road corridor context"),
        ("osm_culvert_tag_score", "mapped culvert/tunnel signal"),
        ("field_report_support_score", "field report match"),
    ]
    for column, label in thresholds:
        if float(row.get(column, 0) or 0) >= 0.6:
            evidence.append(label)
    return "; ".join(evidence) if evidence else "weak evidence; review only if nearby"


def _osm_culvert_tag_score(table: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in ("stream_culvert",):
        if column in table.columns:
            pieces.append(_boolean_score(table[column]))
    if "stream_tunnel" in table.columns:
        pieces.append(
            table["stream_tunnel"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin({"culvert", "yes", "covered"})
            .astype(float)
        )
    return _mean_score(table, pieces)


def _field_report_support_score(table: pd.DataFrame) -> pd.Series:
    pieces = []
    if "is_culvert" in table.columns:
        pieces.append(pd.to_numeric(table["is_culvert"], errors="coerce").fillna(0).clip(0, 1))
    if "dist_to_known_culvert_m" in table.columns:
        distance = pd.to_numeric(table["dist_to_known_culvert_m"], errors="coerce")
        pieces.append((1.0 / (1.0 + distance.clip(lower=0) / 35.0)).fillna(0.0).clip(0, 1))
    return _mean_score(table, pieces)


def _attach_supervised_probability(
    evidence_predictions: gpd.GeoDataFrame,
    supervised_predictions: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if "culvert_probability" not in supervised_predictions.columns:
        return evidence_predictions

    ranked = evidence_predictions.drop(columns=["culvert_probability"], errors="ignore")
    if "candidate_id" in ranked.columns and "candidate_id" in supervised_predictions.columns:
        probability = supervised_predictions[["candidate_id", "culvert_probability"]].drop_duplicates(
            "candidate_id"
        )
        return ranked.merge(probability, on="candidate_id", how="left")

    if len(ranked) == len(supervised_predictions):
        ranked["culvert_probability"] = supervised_predictions["culvert_probability"].to_numpy()
    return ranked


def _score_0_to_1(table: pd.DataFrame, column: str, scale: float) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return (pd.to_numeric(table[column], errors="coerce") / scale).clip(0, 1)


def _model_rank_score(model_probability: pd.Series) -> pd.Series:
    if model_probability.notna().sum() <= 1:
        return model_probability
    return model_probability.rank(pct=True).fillna(0.0).clip(0, 1)


def _known_field_match_mask(table: pd.DataFrame, known_radius_m: float) -> pd.Series:
    known = pd.Series(False, index=table.index)
    if "is_culvert" in table.columns:
        known |= pd.to_numeric(table["is_culvert"], errors="coerce").fillna(0).astype(int) == 1
    if "dist_to_known_culvert_m" in table.columns:
        distance = pd.to_numeric(table["dist_to_known_culvert_m"], errors="coerce")
        known |= distance.notna() & (distance <= known_radius_m)
    return known


def _discovery_evidence_summary(row: pd.Series) -> str:
    base = str(row.get("evidence_summary", "") or "")
    status = row.get("discovery_status")
    if status == "known_field_match":
        return "known field-report match; use for model validation"
    if base and base != "weak evidence; review only if nearby":
        return base
    probability = float(row.get("model_probability_score", 0) or 0)
    evidence = float(row.get("evidence_score", 0) or 0)
    if probability >= 70 and evidence >= 45:
        return "model and GIS evidence agree"
    if probability >= 70:
        return "model-led undiscovered candidate"
    if evidence >= 55:
        return "GIS evidence-led undiscovered candidate"
    return base or "undiscovered candidate; lower confidence"


def _non_culvert_structure_penalty(table: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in ("road_bridge", "road_tunnel"):
        if column in table.columns:
            pieces.append(_boolean_score(table[column]))
    return _mean_score(table, pieces)


def _boolean_score(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "bridge", "tunnel"}).astype(
        float
    )


def _google_earth_url(row: pd.Series) -> str:
    lat = row.get("latitude")
    lon = row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        return ""
    return f"https://earth.google.com/web/search/{lat:.7f},{lon:.7f}"


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return numeric.rank(pct=True).fillna(0.0)


def _inverse_percentile(series: pd.Series) -> pd.Series:
    return 1.0 - _percentile(series)


def _mean_score(table: pd.DataFrame, pieces: list[pd.Series]) -> pd.Series:
    if not pieces:
        return _zero(table)
    stacked = pd.concat(pieces, axis=1)
    return stacked.mean(axis=1).fillna(0.0).clip(0, 1)


def _zero(table: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=table.index)
