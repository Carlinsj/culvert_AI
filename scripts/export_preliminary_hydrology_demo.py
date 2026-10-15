#!/usr/bin/env python3
"""Build the small, local-only hydrology presentation bundle used by the web UI."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "reports/experiments/nysdot_ulster_dem_hydrology_groundwork_v2_20260917"
HYDROLOGY = ROOT / "data/processed/hydrology/usgs_3dep_10m"
WEB_DATA = ROOT / "web/data"
OFFICIAL = ROOT / "data/processed/nysdot/nysdot_small_culverts_r8_ulster_2026_07_14.geoparquet"

THRESHOLDS = (
    (0.1, "0p1km2", "Small headwater"),
    (0.5, "0p5km2", "Medium headwater"),
    (2.0, "2km2", "Established small drainage"),
    (10.0, "10km2", "Larger drainage"),
)


def _number(value):
    if pd.isna(value):
        return None
    return float(value)


def _point_feature(geometry, properties):
    point = shapely.set_precision(geometry, 1e-5, mode="pointwise")
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(point),
    }


def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(EXPERIMENT / "descriptive_validation.csv")
    run_summary = json.loads((EXPERIMENT / "run_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((HYDROLOGY / "suitability_gate.json").read_text(encoding="utf-8"))
    model_summary = json.loads((WEB_DATA / "model_summary.json").read_text(encoding="utf-8"))

    rows = []
    for row in validation.to_dict(orient="records"):
        rows.append(
            {
                "threshold_km2": float(row["threshold_km2"]),
                "official_count": int(row["official_count"]),
                "unlabeled_candidate_count": int(row["unlabeled_candidate_count"]),
                "mapped_stream_sample_count": int(row["mapped_stream_sample_count"]),
                "official_within_m": {
                    str(distance): _number(row[f"official_within_{distance}m_percent"])
                    for distance in (10, 25, 50, 100)
                },
                "unlabeled_within_m": {
                    str(distance): _number(row[f"unlabeled_candidates_within_{distance}m_percent"])
                    for distance in (10, 25, 50, 100)
                },
                "mapped_stream_within_m": {
                    str(distance): _number(row[f"mapped_stream_samples_within_{distance}m_percent"])
                    for distance in (10, 25, 50, 100)
                },
            }
        )

    presentation = {
        "title": "Preliminary DEM hydrology results",
        "experiment_id": run_summary["experiment_id"],
        "status": "Groundwork passed; v7 predictive experiment not yet run",
        "prediction_warning": (
            "The new DEM hydrology products have not produced a culvert prediction yet. "
            "They passed physical and coverage checks and are ready to be tested in v7."
        ),
        "what_was_used": {
            "dem": "Official USGS 3DEP seamless 1/3 arc-second DEM",
            "resolution": "10 m projected grid",
            "processing_extent": "Ulster County plus a 30 km external buffer",
            "conditioning": "WhiteboxTools least-cost breaching, then filling remaining depressions",
            "routing": "D8 flow direction and contributing area",
            "candidate_count": 80620,
            "official_culvert_count": 714,
            "thresholds_km2": [value for value, _, _ in THRESHOLDS],
        },
        "coverage": {
            "county_nodata_before_percent": 29.59,
            "county_nodata_after_percent": gate["county_internal_nodata_percent"],
            "candidate_nodata_before_percent": 24.68,
            "candidate_nodata_after_percent": gate["candidate_nodata_percent"],
            "hydrology_edge_risk_percent": 15.1873,
        },
        "validation": rows,
        "current_ui_model": {
            "label": "Existing field-review model; does not include the new hydrology group",
            "model": model_summary.get("selected_model"),
            "labeled_rows": model_summary.get("rows"),
            "positive_labels": model_summary.get("positive_labels"),
            "negative_labels": model_summary.get("negative_labels"),
            "spatial_average_precision": model_summary.get("spatial_holdout_average_precision"),
            "spatial_roc_auc": model_summary.get("spatial_holdout_roc_auc"),
            "operating_threshold": model_summary.get("operating_threshold"),
            "caveat": (
                "These are existing-model holdout metrics, not v7 results. Prior work found "
                "that whole-route generalization and site-level ranking remain the bottleneck."
            ),
        },
        "map_note": (
            "The map overlay is presentation-only. Lines are simplified to 25 m for browser "
            "performance; the analytical rasters and vectors remain unchanged."
        ),
        "compute": {
            "active_wall_minutes": run_summary["execution_timing"]["approximate_total_active_wall_minutes"],
            "processed_output_gb": round(run_summary["output_bytes"] / 1_000_000_000, 2),
        },
    }
    (WEB_DATA / "preliminary_results.json").write_text(
        json.dumps(presentation, indent=2), encoding="utf-8"
    )

    official = gpd.read_parquet(OFFICIAL)[["culvert_id", "route", "geometry"]].to_crs("EPSG:32618")
    drainage_geometries = {}
    features = []

    for threshold, key, label in THRESHOLDS:
        drainage = gpd.read_parquet(HYDROLOGY / f"drainage_{key}.parquet").to_crs("EPSG:32618")
        simplified = shapely.simplify(drainage.geometry.array, 25.0, preserve_topology=False)
        merged = shapely.union_all(simplified)
        drainage_geometries[key] = shapely.union_all(drainage.geometry.array)
        merged_wgs84 = gpd.GeoSeries([merged], crs="EPSG:32618").to_crs("EPSG:4326").iloc[0]
        merged_wgs84 = shapely.set_precision(merged_wgs84, 1e-5, mode="pointwise")
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "drainage",
                    "threshold_km2": threshold,
                    "threshold_key": key,
                    "label": label,
                },
                "geometry": mapping(merged_wgs84),
            }
        )

        crossings = gpd.read_parquet(
            HYDROLOGY / f"road_drainage_crossings_{key}.parquet"
        ).to_crs("EPSG:4326")
        for crossing in crossings.itertuples(index=False):
            features.append(
                _point_feature(
                    crossing.geometry,
                    {
                        "kind": "crossing",
                        "threshold_km2": threshold,
                        "threshold_key": key,
                        "upstream_area_km2": _number(crossing.upstream_area_m2 / 1_000_000.0),
                        "crossing_angle_degrees": _number(crossing.crossing_angle_degrees),
                        "edge_risk": bool(crossing.edge_risk),
                    },
                )
            )

    official_wgs84 = official.to_crs("EPSG:4326")
    distance_columns = {
        key: official.geometry.distance(drainage_geometries[key]).to_numpy()
        for _, key, _ in THRESHOLDS
    }
    for index, culvert in enumerate(official_wgs84.itertuples(index=False)):
        properties = {
            "kind": "official",
            "culvert_id": str(culvert.culvert_id),
            "route": "" if pd.isna(culvert.route) else str(culvert.route),
        }
        for _, key, _ in THRESHOLDS:
            properties[f"distance_{key}_m"] = round(float(distance_columns[key][index]), 1)
        features.append(_point_feature(culvert.geometry, properties))

    bundle = {
        "type": "FeatureCollection",
        "name": "Preliminary DEM hydrology presentation overlay",
        "features": features,
    }
    (WEB_DATA / "hydrology_demo.geojson").write_text(
        json.dumps(bundle, separators=(",", ":")), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "results": "web/data/preliminary_results.json",
                "map": "web/data/hydrology_demo.geojson",
                "features": len(features),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
