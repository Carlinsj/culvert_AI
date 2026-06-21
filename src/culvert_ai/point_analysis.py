from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from culvert_ai.io import add_wgs84_coordinates, ensure_parent_dir, read_vector, write_vector


POINT_COLUMNS = [
    "point_id",
    "latitude",
    "longitude",
    "nearest_road_distance_m",
    "nearest_road_name",
    "nearest_stream_distance_m",
    "nearest_stream_name",
    "nearest_candidate_distance_m",
    "nearest_candidate_id",
    "nearest_candidate_score",
    "nearest_candidate_status",
    "matched_existing_candidate",
    "inside_analysis_extent",
    "analysis_flag",
    "cluster_id",
]

DEFAULT_TRAINING_FLAGS = ("matched_existing_candidate",)
TRAINING_POINT_COLUMNS = [
    "point_id",
    "latitude",
    "longitude",
    "nearest_road_distance_m",
    "nearest_road_name",
    "nearest_stream_distance_m",
    "nearest_stream_name",
    "nearest_candidate_distance_m",
    "nearest_candidate_id",
    "analysis_flag",
    "training_label",
    "label_source",
    "label_confidence",
]


def write_point_only_layer(
    points_path: str | Path,
    output_path: str | Path,
    csv_output: str | Path | None = None,
) -> dict:
    points = _point_only(read_vector(points_path))
    write_vector(points, output_path)
    result = {"points": Path(output_path), "rows": int(len(points))}
    if csv_output:
        ensure_parent_dir(csv_output)
        points.drop(columns="geometry").to_csv(csv_output, index=False)
        result["points_csv"] = Path(csv_output)
    return result


def analyze_extracted_points(
    points_path: str | Path,
    output_geojson: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
    roads_path: str | Path | None = None,
    streams_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
    match_radius_m: float = 75.0,
    cluster_radius_m: float = 750.0,
) -> dict:
    points = _point_only(read_vector(points_path))
    if points.empty:
        raise ValueError(f"No points found in {points_path}.")

    roads = read_vector(roads_path) if roads_path and Path(roads_path).exists() else None
    streams = read_vector(streams_path) if streams_path and Path(streams_path).exists() else None
    candidates = (
        read_vector(candidates_path) if candidates_path and Path(candidates_path).exists() else None
    )

    analyzed = points.copy()
    analysis_layers = [layer for layer in (roads, streams, candidates) if layer is not None and not layer.empty]
    if analysis_layers:
        analyzed["inside_analysis_extent"] = _inside_combined_extent(points, analysis_layers)
    else:
        analyzed["inside_analysis_extent"] = True

    if roads is not None and not roads.empty:
        analyzed = _attach_nearest_line(analyzed, roads, prefix="road")
    else:
        analyzed["nearest_road_distance_m"] = np.nan
        analyzed["nearest_road_name"] = ""

    if streams is not None and not streams.empty:
        analyzed = _attach_nearest_line(analyzed, streams, prefix="stream")
    else:
        analyzed["nearest_stream_distance_m"] = np.nan
        analyzed["nearest_stream_name"] = ""

    if candidates is not None and not candidates.empty:
        analyzed = _attach_nearest_candidate(analyzed, candidates)
    else:
        analyzed["nearest_candidate_distance_m"] = np.nan
        analyzed["nearest_candidate_id"] = ""
        analyzed["nearest_candidate_score"] = np.nan
        analyzed["nearest_candidate_status"] = ""

    analyzed["matched_existing_candidate"] = (
        pd.to_numeric(analyzed["nearest_candidate_distance_m"], errors="coerce") <= match_radius_m
    )
    analyzed["cluster_id"] = _cluster_ids(analyzed, cluster_radius_m)
    analyzed["analysis_flag"] = analyzed.apply(_analysis_flag, axis=1)
    analyzed = analyzed[[column for column in POINT_COLUMNS if column in analyzed.columns] + ["geometry"]]

    write_vector(analyzed, output_geojson)
    ensure_parent_dir(output_csv)
    analyzed.drop(columns="geometry").to_csv(output_csv, index=False)

    summary = _summary(analyzed, match_radius_m, cluster_radius_m)
    ensure_parent_dir(output_json)
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ensure_parent_dir(output_markdown)
    Path(output_markdown).write_text(_markdown(summary), encoding="utf-8")

    return {
        "analyzed_points_geojson": Path(output_geojson),
        "analyzed_points_csv": Path(output_csv),
        "summary_json": Path(output_json),
        "summary_markdown": Path(output_markdown),
        "rows": int(len(analyzed)),
        "matched_existing_candidates": int(analyzed["matched_existing_candidate"].sum()),
        "outside_analysis_extent": int((~analyzed["inside_analysis_extent"]).sum()),
    }


def write_high_confidence_training_points(
    analysis_path: str | Path,
    output_path: str | Path,
    csv_output: str | Path | None = None,
    accepted_flags: tuple[str, ...] = DEFAULT_TRAINING_FLAGS,
) -> dict:
    points = read_vector(analysis_path)
    missing = [column for column in ("analysis_flag", "inside_analysis_extent") if column not in points.columns]
    if missing:
        raise ValueError(f"Analysis layer is missing required columns: {', '.join(missing)}")

    inside = points["inside_analysis_extent"].map(_truthy)
    accepted = points["analysis_flag"].astype(str).isin(accepted_flags)
    training = points[inside & accepted].copy()
    if training.empty:
        raise ValueError(
            "No extracted points passed the high-confidence training filter. "
            "Inspect the point analysis report before training."
        )

    training["training_label"] = "culvert"
    training["label_source"] = "field_report_coordinate_geospatial_qc"
    training["label_confidence"] = np.where(
        training["analysis_flag"].astype(str) == "matched_existing_candidate",
        0.95,
        0.85,
    )
    selected = [column for column in TRAINING_POINT_COLUMNS if column in training.columns]
    training = training[[*selected, "geometry"]].to_crs("EPSG:4326")

    write_vector(training, output_path)
    result = {
        "training_points": Path(output_path),
        "rows": int(len(training)),
        "accepted_flags": list(accepted_flags),
        "rejected_rows": int(len(points) - len(training)),
        "analysis_flags": {
            str(flag): int(count)
            for flag, count in points["analysis_flag"].astype(str).value_counts().items()
        },
    }
    if csv_output:
        ensure_parent_dir(csv_output)
        training.drop(columns="geometry").to_csv(csv_output, index=False)
        result["training_points_csv"] = Path(csv_output)
    return result


def _point_only(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    wgs84 = add_wgs84_coordinates(points.to_crs("EPSG:4326"))
    cleaned = gpd.GeoDataFrame(
        {
            "point_id": [f"pt_{index + 1:04d}" for index in range(len(wgs84))],
            "latitude": wgs84["latitude"].astype(float),
            "longitude": wgs84["longitude"].astype(float),
        },
        geometry=wgs84.geometry,
        crs="EPSG:4326",
    )
    return cleaned.sort_values(["latitude", "longitude"]).reset_index(drop=True)


def _inside_combined_extent(points: gpd.GeoDataFrame, layers: list[gpd.GeoDataFrame]) -> pd.Series:
    bounds = np.array([layer.to_crs("EPSG:4326").total_bounds for layer in layers])
    margin = 0.02
    minx, miny = bounds[:, 0].min() - margin, bounds[:, 1].min() - margin
    maxx, maxy = bounds[:, 2].max() + margin, bounds[:, 3].max() + margin
    wgs84 = points.to_crs("EPSG:4326")
    return (
        (wgs84.geometry.x >= minx)
        & (wgs84.geometry.x <= maxx)
        & (wgs84.geometry.y >= miny)
        & (wgs84.geometry.y <= maxy)
    )


def _attach_nearest_line(points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
    points_m = points.to_crs(lines.estimate_utm_crs() or "EPSG:3857")
    lines_m = lines.to_crs(points_m.crs).reset_index(drop=True)
    rows = []
    for point in points_m.geometry:
        distances = lines_m.geometry.distance(point)
        nearest_idx = int(distances.idxmin())
        nearest = lines_m.iloc[nearest_idx]
        rows.append(
            {
                f"nearest_{prefix}_distance_m": float(distances.iloc[nearest_idx]),
                f"nearest_{prefix}_name": _feature_name(nearest),
            }
        )
    result = points.copy()
    for column in rows[0]:
        result[column] = [row[column] for row in rows]
    return result


def _attach_nearest_candidate(
    points: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    points_m = points.to_crs(candidates.estimate_utm_crs() or candidates.crs)
    candidates_m = candidates.to_crs(points_m.crs).reset_index(drop=True)
    rows = []
    for point in points_m.geometry:
        distances = candidates_m.geometry.distance(point)
        nearest_idx = int(distances.idxmin())
        nearest = candidates_m.iloc[nearest_idx]
        rows.append(
            {
                "nearest_candidate_distance_m": float(distances.iloc[nearest_idx]),
                "nearest_candidate_id": str(nearest.get("candidate_id", "")),
                "nearest_candidate_score": _candidate_score(nearest),
                "nearest_candidate_status": str(nearest.get("discovery_status", "")),
            }
        )
    result = points.copy()
    for column in rows[0]:
        result[column] = [row[column] for row in rows]
    return result


def _cluster_ids(points: gpd.GeoDataFrame, cluster_radius_m: float) -> list[int]:
    points_m = points.to_crs(points.estimate_utm_crs() or "EPSG:3857")
    coords = np.column_stack([points_m.geometry.x, points_m.geometry.y])
    if len(coords) == 0:
        return []
    labels = DBSCAN(eps=cluster_radius_m, min_samples=2).fit_predict(coords)
    return [int(label) for label in labels]


def _summary(points: gpd.GeoDataFrame, match_radius_m: float, cluster_radius_m: float) -> dict:
    road_dist = pd.to_numeric(points["nearest_road_distance_m"], errors="coerce")
    stream_dist = pd.to_numeric(points["nearest_stream_distance_m"], errors="coerce")
    candidate_dist = pd.to_numeric(points["nearest_candidate_distance_m"], errors="coerce")
    clusters = points[points["cluster_id"] >= 0].groupby("cluster_id")
    cluster_rows = []
    for cluster_id, cluster in clusters:
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "points": int(len(cluster)),
                "centroid_latitude": float(cluster.geometry.y.mean()),
                "centroid_longitude": float(cluster.geometry.x.mean()),
            }
        )

    flags = points["analysis_flag"].value_counts().to_dict()
    return {
        "rows": int(len(points)),
        "match_radius_m": float(match_radius_m),
        "cluster_radius_m": float(cluster_radius_m),
        "bounds": [float(value) for value in points.total_bounds],
        "inside_analysis_extent": int(points["inside_analysis_extent"].sum()),
        "outside_analysis_extent": int((~points["inside_analysis_extent"]).sum()),
        "matched_existing_candidates": int(points["matched_existing_candidate"].sum()),
        "unmatched_existing_candidates": int((~points["matched_existing_candidate"]).sum()),
        "nearest_road_distance_m": _distance_stats(road_dist),
        "nearest_stream_distance_m": _distance_stats(stream_dist),
        "nearest_candidate_distance_m": _distance_stats(candidate_dist),
        "analysis_flags": {str(key): int(value) for key, value in flags.items()},
        "clusters": sorted(cluster_rows, key=lambda row: row["points"], reverse=True),
    }


def _distance_stats(values: pd.Series) -> dict:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {}
    return {
        "min": float(valid.min()),
        "median": float(valid.median()),
        "mean": float(valid.mean()),
        "p90": float(valid.quantile(0.9)),
        "max": float(valid.max()),
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Extracted Point Analysis",
        "",
        "This analysis treats the imported coordinates as a point set, independent of report-level text.",
        "",
        f"- Points: {summary['rows']}",
        f"- Matched to existing model candidates within {summary['match_radius_m']:.0f} m: "
        f"{summary['matched_existing_candidates']}",
        f"- Outside current road/stream/model analysis extent: {summary['outside_analysis_extent']}",
        f"- Cluster radius: {summary['cluster_radius_m']:.0f} m",
        "",
        "## Distance Summary",
        "",
        _stats_line("Nearest road", summary["nearest_road_distance_m"]),
        _stats_line("Nearest stream", summary["nearest_stream_distance_m"]),
        _stats_line("Nearest model candidate", summary["nearest_candidate_distance_m"]),
        "",
        "## Flags",
        "",
    ]
    for flag, count in summary["analysis_flags"].items():
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Largest Clusters", ""])
    for cluster in summary["clusters"][:10]:
        lines.append(
            f"- Cluster {cluster['cluster_id']}: {cluster['points']} points near "
            f"{cluster['centroid_latitude']:.6f}, {cluster['centroid_longitude']:.6f}"
        )
    lines.append("")
    return "\n".join(lines)


def _stats_line(label: str, stats: dict) -> str:
    if not stats:
        return f"- {label}: unavailable"
    return (
        f"- {label}: median {stats['median']:.1f} m, p90 {stats['p90']:.1f} m, "
        f"max {stats['max']:.1f} m"
    )


def _analysis_flag(row: pd.Series) -> str:
    if not bool(row.get("inside_analysis_extent", True)):
        return "outside_current_analysis_extent"
    candidate_distance = float(row.get("nearest_candidate_distance_m", np.inf))
    road_distance = float(row.get("nearest_road_distance_m", np.inf))
    stream_distance = float(row.get("nearest_stream_distance_m", np.inf))
    if candidate_distance <= 75:
        return "matched_existing_candidate"
    if road_distance <= 50 and stream_distance <= 100:
        return "road_stream_context_no_candidate_match"
    if road_distance > 250:
        return "far_from_current_road_layer"
    return "needs_manual_review"


def _feature_name(row: pd.Series) -> str:
    for column in ("FULLNAME", "name", "Name", "road_name", "stream_name", "LINEARID", "id"):
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    return ""


def _candidate_score(row: pd.Series) -> float | None:
    for column in ("discovery_score", "culvert_likelihood_score", "culvert_probability"):
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
