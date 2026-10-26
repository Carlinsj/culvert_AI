"""Read-only recorded-versus-predicted comparison datasets for the research UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.hydrology_service import ROOT, geojson

SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
PHYSICAL = ROOT / "reports/experiments/physical_discovery_20261016"
SELECTIVE = ROOT / "reports/experiments/selective_admission_20261016"
SPRINT = ROOT / "reports/experiments/model_improvement_sprint_20261017"
V4 = ROOT / "reports/experiments/physical_ranker_v4_20261019"
V5 = ROOT / "reports/experiments/baseline_preserving_v5_20261019"
V6 = ROOT / "reports/experiments/final_physical_optimization_20261019"
V7 = ROOT / "reports/experiments/v7_physical_crossings_final_20261020"
SURVEYED_V4 = ROOT / "reports/experiments/surveyed_route_supervision_v4_20261020"

AREAS = {
    "rock": (41.880519, -74.110507, 1500.0, "Rock Ledge"),
    "beaver": (42.07647, -74.207172, 2500.0, "Beaver Kill"),
}
MODELS = {
    "baseline_rf": {
        "label": "Original RF · 13,065 physical sites",
        "path": SELECTIVE / "queue_BASELINE-RF.parquet",
        "version": "existing-baseline-rf",
    },
    "baseline_hgb": {
        "label": "Existing HGB comparison · same 13,065 sites",
        "path": SELECTIVE / "queue_BASELINE-HGB.parquet",
        "version": "existing-baseline-hgb",
    },
    "v3": {
        "label": "Experimental v3 RF · 68,302 hypotheses",
        "path": SPRINT / "ranked_physical_sites.parquet",
        "version": "culvert-retrieval-v3-experimental",
    },
    "v5": {
        "label": "Experimental v5 baseline-preserving reranker · 13,065 sites",
        "path": V5 / "ranked_sites.parquet",
        "version": "baseline-preserving-physical-reranker-v5-experimental",
    },
    "v6": {
        "label": "Experimental v6 compact physical queue · 50,826 sites",
        "path": V6 / "ranked_compact_physical_sites.parquet",
        "version": "compact-physical-sites-v6-experimental",
    },
    "v7": {
        "label": "v7 physical crossings · 142 sites · surveyed routes",
        "path": V7 / "ranked_v7_frozen_rf.parquet",
        "version": "physical-crossings-v7.1-development",
    },
}


def _bool(value) -> bool:
    return bool(value) and str(value).lower() not in {"false", "0", "nan"}


@lru_cache(maxsize=1)
def inventory() -> gpd.GeoDataFrame:
    frame = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    audit = pd.read_csv(PHYSICAL / "all_959_record_diagnosis.csv")[["record_id", "query_success"]]
    frame = frame.merge(audit, on="record_id", validate="one_to_one")
    frame["query_success"] = frame.query_success.map(_bool)
    frame["x_m"], frame["y_m"] = frame.geometry.x, frame.geometry.y
    assert len(frame) == 959 and int(frame.query_success.sum()) == 847
    return frame


@lru_cache(maxsize=1)
def surveyed_inventory() -> gpd.GeoDataFrame:
    frame = gpd.read_parquet(SURVEYED_V4 / "heldout_survey_inventory.parquet").to_crs(32618)
    frame = frame.copy()
    frame["record_id"] = frame.actual_id
    frame["source_excel_row"] = frame.source_row
    frame["Region"] = "Complete surveyed route"
    frame["Route"] = frame.route
    frame["Team"] = frame.inventory_sources
    frame["Culvert ID"] = frame.culvert_id
    frame["query_success"] = True
    frame["x_m"], frame["y_m"] = frame.geometry.x, frame.geometry.y
    assert len(frame) == 31
    return frame


@lru_cache(maxsize=1)
def surveyed_segments() -> gpd.GeoDataFrame:
    return gpd.read_parquet(SURVEYED_V4 / "surveyed_route_segments.parquet").to_crs(32618)


@lru_cache(maxsize=1)
def v7_supporting_links() -> gpd.GeoDataFrame:
    return gpd.read_parquet(V7 / "supporting_d8_links.parquet").to_crs(32618)


@lru_cache(maxsize=8)
def predictions(model: str) -> gpd.GeoDataFrame:
    if model == "v4" and (V4 / "ranked_sites.parquet").exists():
        source = {"label": "Experimental v4 physical reranker", "path": V4 / "ranked_sites.parquet", "version": "physical-ranker-v4"}
    else:
        source = MODELS.get(model)
    if source is None:
        raise ValueError("Choose a supported comparison model.")
    frame = gpd.read_parquet(source["path"]).copy()
    if frame.crs is None or frame.crs.to_epsg() != 32618:
        frame = frame.to_crs(32618)
    if "global_rank" not in frame:
        frame["global_rank"] = np.arange(1, len(frame) + 1)
    frame = frame.sort_values(["global_rank", "candidate_id"], kind="stable").reset_index(drop=True)
    frame["x_m"], frame["y_m"] = frame.geometry.x, frame.geometry.y
    return frame


def options() -> dict:
    models = [dict(id=key, label=value["label"], version=value["version"], sites=len(predictions(key))) for key, value in MODELS.items()]
    if (V4 / "ranked_sites.parquet").exists():
        models.append(dict(id="v4", label="Experimental v4 physical reranker", version="physical-ranker-v4", sites=len(predictions("v4"))))
    routes = sorted(
        {str(value) for value in inventory().loc[inventory().query_success, "Route"].dropna().unique()}
        | set(surveyed_inventory().Route.astype(str))
    )
    return {
        "models": models,
        "routes": routes,
        "actual_total": len(inventory()),
        "supported_search_records": int(inventory().query_success.sum()),
        "outside_supported_searches": int((~inventory().query_success).sum()),
        "areas": [
            *[{"id": key, "label": value[3]} for key, value in AREAS.items()],
            {"id": "surveyed", "label": "Complete surveyed NY214 + NY28 intervals"},
        ],
    }


def _scope(payload: dict, truth: gpd.GeoDataFrame):
    selection = str(payload.get("selection", "rock"))
    if selection in AREAS:
        latitude, longitude, radius, label = AREAS[selection]
        center = gpd.GeoSeries([Point(longitude, latitude)], crs=4326).to_crs(32618).iloc[0]
        return truth.loc[truth.geometry.distance(center).le(radius)].copy(), center.buffer(radius), label
    if selection in {"region6", "region8"}:
        label = "Region 6" if selection == "region6" else "Region 8"
        selected = truth.loc[truth.Region.astype(str).str.casefold().eq(label.casefold())].copy()
        area = selected.geometry.union_all().convex_hull.buffer(5000) if len(selected) else None
        return selected, area, label
    if selection == "route":
        route = str(payload.get("route", "")).strip()
        if not route:
            raise ValueError("Choose a route.")
        selected = truth.loc[truth.Route.astype(str).eq(route)].copy()
        area = selected.geometry.union_all().buffer(750) if len(selected) else None
        return selected, area, f"Route {route}"
    if selection == "coordinates":
        latitude = float(payload.get("latitude"))
        longitude = float(payload.get("longitude"))
        radius = float(payload.get("radius_m", 2500))
        if not np.isfinite([latitude, longitude, radius]).all() or not 100 <= radius <= 20000:
            raise ValueError("Use valid coordinates and a 100–20,000 m radius.")
        center = gpd.GeoSeries([Point(longitude, latitude)], crs=4326).to_crs(32618).iloc[0]
        return truth.loc[truth.geometry.distance(center).le(radius)].copy(), center.buffer(radius), "Coordinate search"
    if selection == "all":
        return truth.copy(), None, "All supported inventory locations"
    raise ValueError("Choose a supported location selection.")


def _within(frame: gpd.GeoDataFrame, area):
    if area is None:
        return frame.copy()
    if area.is_empty:
        return frame.iloc[:0].copy()
    return frame.iloc[frame.sindex.query(area, predicate="intersects")].copy()


def _surveyed_scope(payload: dict):
    selection = str(payload.get("selection", "surveyed"))
    truth = surveyed_inventory()
    if selection in {"surveyed", "all"}:
        return truth.copy(), None, "Complete surveyed NY214 + NY28 intervals"
    if selection == "route":
        route = str(payload.get("route", "")).strip()
        selected = truth.loc[truth.Route.astype(str).eq(route)].copy()
        segments = surveyed_segments().loc[surveyed_segments().route.eq(route)]
        if selected.empty or segments.empty:
            raise ValueError("Choose NY214 or NY28 for the v7 surveyed-route comparison.")
        return selected, segments.geometry.buffer(100).union_all(), f"Surveyed {route} intervals"
    raise ValueError("v7 is available only for the complete surveyed-route selection or NY214/NY28.")


def compare(payload: dict) -> dict:
    model = str(payload.get("model", "baseline_rf"))
    source = ({**MODELS, "v4": {"label": "Experimental v4 physical reranker", "version": "physical-ranker-v4"}}).get(model)
    if source is None or (model == "v4" and not (V4 / "ranked_sites.parquet").exists()):
        raise ValueError("Choose an available comparison model.")
    try:
        budget = int(payload.get("budget", 1000))
        tolerance = float(payload.get("tolerance_m", 20))
    except (TypeError, ValueError) as error:
        raise ValueError("Budget and tolerance must be numeric.") from error
    if budget not in {25, 50, 75, 100, 117, 119, 125, 142, 150, 200, 250, 500, 1000, 5000, 10000, 13065}:
        raise ValueError("Choose a supported inspection budget.")
    if tolerance not in {5, 10, 15, 20, 25, 50}:
        raise ValueError("Choose a supported matching tolerance.")

    if model == "v7":
        actual, area, label = _surveyed_scope(payload)
        actual_total, supported_total, outside_total = 31, 31, 0
    else:
        supported = inventory().loc[inventory().query_success].copy()
        actual, area, label = _scope(payload, supported)
        actual_total, supported_total, outside_total = 959, 847, 112
    scoped = _within(predictions(model), area).sort_values(["global_rank", "candidate_id"], kind="stable")
    shown = scoped.head(budget).copy().reset_index(drop=True)
    actual = actual.sort_values("source_excel_row", kind="stable").reset_index(drop=True)
    truth_xy = np.c_[actual.geometry.x, actual.geometry.y]
    matched = match_one_to_one(shown, truth_xy, tolerance)
    actual["match_status"] = "missed"
    actual["matched_prediction_id"] = None
    actual["match_distance_m"] = np.nan
    shown["match_status"] = "unverified"
    shown["matched_record_id"] = None
    shown["match_distance_m"] = np.nan
    for row in matched.itertuples():
        prediction_id = str(shown.iloc[row.site_row].get("physical_site_id", shown.iloc[row.site_row].candidate_id))
        actual.loc[row.real_row, ["match_status", "matched_prediction_id", "match_distance_m"]] = ["matched", prediction_id, row.distance_m]
        shown.loc[row.site_row, ["match_status", "matched_record_id", "match_distance_m"]] = ["matched", actual.iloc[row.real_row].record_id, row.distance_m]

    actual_columns = ["record_id", "source_excel_row", "Region", "Route", "Team", "Culvert ID", "query_success", "x_m", "y_m", "match_status", "matched_prediction_id", "match_distance_m"]
    prediction_columns = ["candidate_id", "physical_site_id", "road_id", "road_name", "route_id", "proposal_kind", "candidate_source", "support_sources", "physical_evidence", "member_physical_evidence", "local_profile_dip_20m", "negative_tpi_9x9_m", "negative_tpi_15x15_m", "upstream_area_at_link_m2", "crossing_angle_degrees", "global_rank", "existing_rf_score", "model_score", "score_RF", "score_HGB", "score_v3", "research_priority", "member_count", "x_m", "y_m", "match_status", "matched_record_id", "match_distance_m"]
    distances = matched.distance_m.to_numpy(float)
    bounds_frame = pd.concat([actual[["geometry"]], shown[["geometry"]]], ignore_index=True)
    bounds = None
    if len(bounds_frame):
        bounds_geo = gpd.GeoDataFrame(bounds_frame, geometry="geometry", crs=32618).to_crs(4326).total_bounds
        bounds = [float(value) for value in bounds_geo]
    support = gpd.GeoDataFrame(geometry=[], crs=32618)
    if model == "v7" and len(shown):
        keys = set(shown.get("drainage_path_identity", pd.Series(dtype=str)).dropna().astype(str))
        support = v7_supporting_links().loc[
            v7_supporting_links().link_key.astype(str).isin(keys)
        ].copy()
    return {
        "experimental": True,
        "selection": label,
        "model": {"id": model, "label": source["label"], "version": source["version"], "full_population": len(predictions(model))},
        "budget": budget,
        "budget_scope": "Top K by saved global rank after geographic selection",
        "tolerance_m": tolerance,
        "actual_total": actual_total,
        "supported_search_records": supported_total,
        "outside_supported_searches": outside_total,
        "active_actual": len(actual),
        "available_predictions_in_selection": len(scoped),
        "shown_predictions": len(shown),
        "matched": len(matched),
        "missed": len(actual) - len(matched),
        "recall": len(matched) / len(actual) if len(actual) else None,
        "matches_per_1000": 1000 * len(matched) / len(shown) if len(shown) else None,
        "median_distance_m": float(np.median(distances)) if len(distances) else None,
        "bounds": bounds,
        "actuals": geojson(actual, actual_columns),
        "predictions": geojson(shown, prediction_columns),
        "supporting_evidence": geojson(support, ["link_key", "upstream_area_m2"]),
        "notice": "Retrospective, development-exposed comparison. v7 inventory coordinates were used only after candidate sealing. Unmatched predictions are unverified, not false positives.",
    }
