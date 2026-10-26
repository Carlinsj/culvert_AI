"""Train route-held-out physical-crossing classifiers on confirmed survey extents.

The ordering is deliberate: official marker/road geometry, physical candidate
generation, feature extraction, and frozen-RF scoring all finish before either
culvert inventory is loaded.  Inventory coordinates are used only for labels
and evaluation after the candidate population is sealed.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from shapely.geometry import Point
from threadpoolctl import threadpool_limits

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file
from culvert_ai.surveyed_routes import (
    CRS,
    associate_inventory_to_segments,
    build_survey_segments,
    canonical_route,
    deduplicate_inventory,
    label_physical_sites,
    read_survey_manifest,
)
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.hydrology_sites import (
    THRESHOLDS,
    consolidate_events,
    detect_events,
    physical_measurements,
    search_candidates,
)
from staging.model_sprint import FEATURE_FAMILIES, group_extrema, multiscale_extrema


VERSION = "surveyed-route-supervision-v4-experimental"
REPORT = ROOT / "reports/experiments/surveyed_route_supervision_v4_20261020"
MODEL_DIR = ROOT / "models/experiments/surveyed_route_supervision_v4_20261020"
MANIFEST = ROOT / "configs/surveyed_route_segments.csv"
EXTENT_EXCLUSIONS = ROOT / "configs/surveyed_route_extent_exclusions.csv"
OFFICIAL_WORKBOOK = ROOT / "data/incoming_nysdot/Small Culverts_R8_2026.07.14.xlsx"
OFFICIAL_EXTRACT = ROOT / "data/processed/nysdot/nysdot_small_culverts_r8_2026_07_14.geoparquet"
SUMMER_EXTRACT = ROOT / "reports/experiments/summer_inventory_20261016/inventory_extract.json"
HYDRO = ROOT / "data/processed/hydrology/usgs_3dep_10m"
MARKER_URL = "https://gis.dot.ny.gov/hostingny/rest/services/Roadways/Roadway/FeatureServer/0/query"
ROADWAY_URL = "https://gis.dot.ny.gov/hostingny/rest/services/Roadways/Roadway/FeatureServer/3/query"
BUDGETS = (5, 10, 25, 50, 100, 250)
RANDOM_TRIALS = 500
RANDOM_SEED = 20261020


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def fetch_json(url: str, parameters: dict[str, object]) -> dict:
    request_url = url + "?" + urllib.parse.urlencode(parameters)
    with urllib.request.urlopen(request_url, timeout=60) as response:  # noqa: S310 - fixed official URLs
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"NYSDOT service error: {result['error']}")
    return result


def resolve_official_geometry(specs, report: Path):
    marker_where = "REFERENCE_MARKER_PANEL IN (" + ",".join(
        "'" + marker.replace("'", "''") + "'"
        for marker in sorted({value for spec in specs for value in (spec.start_marker, spec.end_marker)})
    ) + ")"
    marker_response = fetch_json(
        MARKER_URL,
        {
            "where": marker_where,
            "outFields": "OBJECTID,ROUTE_ID,MEASURE,REFERENCE_MARKER_PANEL,Panel,LOCERROR",
            "returnGeometry": "true",
            "outSR": 26918,
            "orderByFields": "REFERENCE_MARKER_PANEL",
            "f": "geojson",
        },
    )
    route_ids = sorted({spec.route_id for spec in specs})
    route_where = "Route_ID IN (" + ",".join(f"'{value}'" for value in route_ids) + ")"
    roadway_response = fetch_json(
        ROADWAY_URL,
        {
            "where": route_where,
            "outFields": "OBJECTID,Route_ID,From_Measure,To_Measure,Route_Display_Value,Route_Signing,Route_Number,County_Name,Roadway_Name,Begin_Description,End_Description",
            "returnGeometry": "true",
            "returnM": "true",
            "returnZ": "false",
            "outSR": 26918,
            "orderByFields": "Route_ID,From_Measure",
            "f": "json",
        },
    )
    save_json(report / "nysdot_reference_markers_snapshot.geojson", marker_response)
    save_json(report / "nysdot_measured_routes_snapshot.json", roadway_response)
    segments = build_survey_segments(specs, marker_response["features"], roadway_response["features"])
    segments.to_parquet(report / "surveyed_route_segments.parquet", index=False)
    return segments, marker_response, roadway_response


def sealed_candidates(segments: gpd.GeoDataFrame, report: Path) -> gpd.GeoDataFrame:
    segment_roads = segments.to_crs(32618).copy()
    streams = gpd.read_file(ROOT / "data/raw/streams.gpkg").to_crs(32618)
    networks = {
        key: gpd.read_parquet(HYDRO / f"drainage_{key}.parquet").to_crs(32618)
        for key in THRESHOLDS
    }
    events, rejected = detect_events(segment_roads, streams, networks)
    events["segment_id"] = events.road_id.astype(str)
    route_by_segment = segments.set_index("segment_id").route.to_dict()
    events["route"] = events.segment_id.map(route_by_segment)
    events.to_parquet(report / "physical_crossing_events.parquet", index=False)
    rejected.to_csv(report / "rejected_crossing_contacts.csv", index=False)
    event_sites = consolidate_events(events)
    event_sites["segment_id"] = event_sites.road_id.astype(str)
    event_sites["route"] = event_sites.segment_id.map(route_by_segment)
    probes = search_candidates(segment_roads, interval_m=10.0)
    probes["road_part_index"] = probes.route_part_index
    probes["road_chainage_m"] = probes.route_sample_distance_m
    probes["segment_id"] = probes.road_id.astype(str)
    probes["route"] = probes.segment_id.map(route_by_segment)
    probes.to_parquet(report / "internal_terrain_measurement_probes.parquet", index=False)
    all_roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618)
    all_roads["route_id"] = all_roads.id.astype(str)
    feature_roads = gpd.GeoDataFrame(
        pd.concat([all_roads, segment_roads], ignore_index=True, sort=False),
        geometry="geometry",
        crs=32618,
    )
    measurement_points = gpd.GeoDataFrame(
        pd.concat([events, probes], ignore_index=True, sort=False),
        geometry="geometry",
        crs=32618,
    )
    measured = physical_measurements(measurement_points, feature_roads, streams, HYDRO)
    measured["segment_id"] = measured.road_id.astype(str)
    measured["route"] = measured.segment_id.map(route_by_segment)
    extrema = multiscale_extrema(measured)
    terrain_sites = group_extrema(extrema)
    terrain_sites["segment_id"] = terrain_sites.road_id.astype(str)
    terrain_sites["route"] = terrain_sites.segment_id.map(route_by_segment)
    # An explicit mapped/DEM intersection represents the crossing when a
    # terrain extremum falls at the same road location.  This suppresses
    # duplicate recommendations without moving either source coordinate.
    keep_terrain = []
    for index, terrain in terrain_sites.iterrows():
        same_segment = event_sites.loc[event_sites.segment_id.eq(terrain.segment_id)]
        duplicate = any(
            terrain.geometry.distance(event.geometry) <= 20.0
            and abs(float(terrain.road_chainage_m) - float(event.road_chainage_m)) <= 20.0
            for event in same_segment.itertuples()
        )
        if not duplicate:
            keep_terrain.append(index)
    terrain_sites = terrain_sites.loc[keep_terrain].copy()
    selected_ids = set(event_sites.candidate_id) | set(terrain_sites.candidate_id)
    result = measured.loc[measured.candidate_id.isin(selected_ids)].copy()
    event_meta = event_sites.set_index("candidate_id")
    terrain_meta = terrain_sites.set_index("candidate_id")
    result["physical_site_id"] = result.candidate_id.map(
        pd.concat([event_meta.physical_site_id, terrain_meta.physical_site_id])
    )
    result["member_count"] = result.candidate_id.map(
        pd.concat([event_meta.member_count, terrain_meta.member_count])
    )
    result["support_sources"] = result.candidate_id.map(event_meta.support_sources)
    result.loc[result.candidate_id.isin(terrain_meta.index), "support_sources"] = result.loc[
        result.candidate_id.isin(terrain_meta.index), "candidate_id"
    ].map(terrain_meta.physical_evidence_sources)
    result["candidate_source"] = np.where(
        result.candidate_id.isin(terrain_meta.index),
        "multiscale_terrain_extremum",
        "explicit_water_road_intersection",
    )
    result.to_parquet(report / "sealed_physical_crossing_features.parquet", index=False)
    return result.reset_index(drop=True)


def frozen_rf_scores(sites: pd.DataFrame) -> np.ndarray:
    bundle = load_frozen_flow_c_bundle(ROOT / "models/flow_c_frozen_v1.joblib")
    columns = bundle["feature_columns"]
    values = sites.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    fill = pd.Series(bundle["fill_values"], dtype=float).reindex(columns).fillna(0.0)
    values = values.fillna(fill)
    with threadpool_limits(limits=2):
        return bundle["model"].predict_proba(values)[:, 1]


def load_inventory_records() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    # This immutable extract retains every source workbook column and avoids a
    # runtime Excel-engine dependency.  The workbook hash remains part of the
    # experiment's input manifest.
    official = gpd.read_parquet(OFFICIAL_EXTRACT)
    official_records = gpd.GeoDataFrame(
        {
            "record_id": "official_row_" + official.source_row_id.astype(str),
            "source": "official",
            "source_row": official.source_row_id,
            "culvert_id": official.culvert_id,
            "route": official.route.map(canonical_route),
            "reference_marker": official["Reference Marker"].fillna("").astype(str),
            "geometry": official.geometry,
        },
        geometry="geometry",
        crs=official.crs,
    )
    summer_payload = json.loads(SUMMER_EXTRACT.read_text())
    summer = pd.DataFrame(summer_payload["records"])
    summer_records = gpd.GeoDataFrame(
        {
            "record_id": summer.record_id,
            "source": "summer",
            "source_row": summer.source_excel_row,
            "culvert_id": summer["Culvert ID"],
            "route": summer.Route.map(canonical_route),
            "reference_marker": "",
            "geometry": gpd.points_from_xy(summer.Longitude, summer.Latitude),
        },
        geometry="geometry",
        crs=4326,
    ).to_crs(CRS)
    all_records = gpd.GeoDataFrame(
        pd.concat([official_records.to_crs(CRS), summer_records], ignore_index=True),
        geometry="geometry",
        crs=CRS,
    )
    route_counts = (
        all_records.groupby(["source", "route"], dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    return all_records, route_counts


def scoped_truth(
    records: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, report: Path
) -> gpd.GeoDataFrame:
    routes = set(segments.route)
    possible = records.loc[records.route.isin(routes)].copy()
    route_buffer = segments.geometry.buffer(500).union_all()
    possible = possible.loc[possible.geometry.intersects(route_buffer)].copy()
    deduped = deduplicate_inventory(possible)
    truth, excluded = associate_inventory_to_segments(deduped, segments)
    deduped.to_parquet(report / "deduplicated_inventory_near_survey_extents.parquet", index=False)
    truth.to_parquet(report / "heldout_survey_inventory.parquet", index=False)
    excluded.to_csv(report / "inventory_extent_exclusions.csv", index=False)
    return truth


def balanced_weights(target: pd.Series) -> np.ndarray:
    counts = target.astype(int).value_counts()
    return target.astype(int).map({label: len(target) / (2 * count) for label, count in counts.items()}).to_numpy(float)


def usable_features(training: pd.DataFrame, declared: tuple[str, ...]) -> list[str]:
    result = []
    for column in declared:
        if column not in training:
            continue
        values = pd.to_numeric(training[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() >= 2 and values.dropna().nunique() > 1:
            result.append(column)
    return result


def estimator(family: str, features: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        [("physical", SimpleImputer(strategy="median", add_indicator=True), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if family == "rf":
        model = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_SEED,
            n_jobs=2,
        )
    elif family == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=3,
            l2_regularization=8,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(family)
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def evaluate_queue(
    queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, route: str, system: str, feature_set: str
) -> list[dict]:
    actual = truth.loc[truth.route.eq(route)].reset_index(drop=True)
    ordered = queue.loc[queue.route.eq(route)].copy().reset_index(drop=True)
    truth_xy = np.c_[actual.geometry.x, actual.geometry.y]
    segment_lengths = queue.attrs.get("segment_lengths", {})
    length_km = (
        float(sum(segment_lengths.get(value, 0.0) for value in ordered.segment_id.unique()) / 1000)
        if len(ordered)
        else 0.0
    )
    rows = []
    checkpoints = [*BUDGETS, "full"]
    seen = set()
    for requested in checkpoints:
        count = len(ordered) if requested == "full" else min(int(requested), len(ordered))
        if (requested, count) in seen:
            continue
        seen.add((requested, count))
        selected = ordered.head(count)
        matched = match_one_to_one(selected, truth_xy, 20.0) if len(actual) else pd.DataFrame()
        rows.append(
            {
                "route": route,
                "system": system,
                "feature_set": feature_set,
                "budget": requested,
                "predicted_sites": count,
                "candidate_population": len(ordered),
                "actual_culverts": len(actual),
                "matched_within_20m": len(matched),
                "recall": len(matched) / len(actual) if len(actual) else None,
                "median_localization_error_m": float(matched.distance_m.median()) if len(matched) else None,
                "predictions_per_road_km": count / length_km if length_km else None,
            }
        )
    return rows


def minimum_budget(queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, route: str) -> int | None:
    actual = truth.loc[truth.route.eq(route)].reset_index(drop=True)
    if not len(actual):
        return None
    target = int(np.ceil(0.70 * len(actual)))
    xy = np.c_[actual.geometry.x, actual.geometry.y]
    ordered = queue.loc[queue.route.eq(route)].reset_index(drop=True)
    for count in range(1, len(ordered) + 1):
        if len(match_one_to_one(ordered.head(count), xy, 20.0)) >= target:
            return count
    return None


def sample_points(segments: gpd.GeoDataFrame, route: str, count: int, rng) -> gpd.GeoDataFrame:
    parts = segments.loc[segments.route.eq(route)].geometry.tolist()
    lengths = np.asarray([part.length for part in parts], float)
    if not len(parts) or lengths.sum() <= 0 or count <= 0:
        return gpd.GeoDataFrame(geometry=[], crs=segments.crs)
    choices = rng.choice(len(parts), size=count, p=lengths / lengths.sum())
    points = [parts[index].interpolate(float(rng.random()) * lengths[index]) for index in choices]
    return gpd.GeoDataFrame({"candidate_id": [f"random_{i}" for i in range(count)]}, geometry=points, crs=segments.crs)


def random_controls(segments, truth, routes, site_counts):
    rows = []
    for route in routes:
        actual = truth.loc[truth.route.eq(route)].reset_index(drop=True)
        xy = np.c_[actual.geometry.x, actual.geometry.y]
        population = int(site_counts[route])
        budgets = sorted({min(budget, population) for budget in BUDGETS} | {population})
        for budget in budgets:
            values = []
            for trial in range(RANDOM_TRIALS):
                sites = sample_points(
                    segments, route, budget, np.random.default_rng(RANDOM_SEED + 1000 * budget + trial)
                )
                values.append(len(match_one_to_one(sites, xy, 20.0)))
            rows.append(
                {
                    "route": route,
                    "budget": budget,
                    "trials": RANDOM_TRIALS,
                    "actual_culverts": len(actual),
                    "mean_matches": float(np.mean(values)),
                    "p025_matches": float(np.quantile(values, 0.025)),
                    "p975_matches": float(np.quantile(values, 0.975)),
                    "mean_recall": float(np.mean(values) / len(actual)) if len(actual) else None,
                }
            )
    return pd.DataFrame(rows)


def ranked(frame: gpd.GeoDataFrame, score_column: str) -> gpd.GeoDataFrame:
    result = frame.sort_values(
        ["route", score_column, "candidate_id"], ascending=[True, False, True], kind="stable"
    ).copy()
    result["route_rank"] = result.groupby("route").cumcount() + 1
    sizes = result.groupby("route").candidate_id.transform("size")
    result["route_priority"] = 1 - (result.route_rank - 1) / np.maximum(sizes - 1, 1)
    return result.reset_index(drop=True)


def train_and_evaluate(labeled, truth, segments, report, model_dir):
    routes = sorted(segments.route.unique())
    segment_lengths = segments.set_index("segment_id").length_m.to_dict()
    labeled.attrs["segment_lengths"] = segment_lengths
    route_rows = []
    fold_rows = []
    queues = {}
    baseline = ranked(labeled, "existing_rf_score")
    baseline.attrs["segment_lengths"] = segment_lengths
    queues[("existing_rf", "all_physical")] = baseline
    for route in routes:
        train = labeled.loc[
            ~labeled.route.eq(route) & labeled.supervision_label.isin(["positive", "negative"])
        ].copy()
        held = labeled.loc[labeled.route.eq(route)].copy()
        overlap = set(train.segment_id) & set(held.segment_id)
        if overlap:
            raise RuntimeError(f"Held-out segment overlap: {sorted(overlap)}")
        for family in ("rf", "hgb"):
            for feature_set, declared in FEATURE_FAMILIES.items():
                features = usable_features(train, declared)
                record = {
                    "heldout_route": route,
                    "family": family,
                    "feature_set": feature_set,
                    "training_routes": ",".join(sorted(train.route.unique())),
                    "training_rows": len(train),
                    "training_positives": int(train.target.fillna(0).sum()),
                    "training_negatives": int(train.target.eq(0).sum()),
                    "heldout_rows": len(held),
                    "feature_count": len(features),
                    "heldout_coordinates_used_for_generation_or_fit": False,
                    "status": "fit",
                }
                if train.target.nunique() < 2 or len(features) < 2:
                    record["status"] = "not_fit_insufficient_two_class_training_data"
                    fold_rows.append(record)
                    continue
                model = estimator(family, features)
                with threadpool_limits(limits=2):
                    model.fit(
                        train[features],
                        train.target.astype(int),
                        model__sample_weight=balanced_weights(train.target),
                    )
                    scores = model.predict_proba(held[features])[:, 1]
                model_path = model_dir / f"{family}_{feature_set}_heldout_{route}.joblib"
                joblib.dump(
                    {
                        "version": VERSION,
                        "experimental": True,
                        "heldout_route": route,
                        "training_routes": sorted(train.route.unique()),
                        "features": features,
                        "model": model,
                    },
                    model_path,
                    compress=3,
                )
                record["model_path"] = str(model_path.relative_to(ROOT))
                fold_rows.append(record)
                score_column = f"score_{family}_{feature_set}"
                labeled.loc[held.index, score_column] = scores
    for key in [
        (family, feature_set)
        for family in ("rf", "hgb")
        for feature_set in FEATURE_FAMILIES
    ]:
        score_column = f"score_{key[0]}_{key[1]}"
        if score_column in labeled and labeled[score_column].notna().all():
            queue = ranked(labeled, score_column)
            queue.attrs["segment_lengths"] = segment_lengths
            queues[key] = queue
    for (system, feature_set), queue in queues.items():
        for route in routes:
            route_rows.extend(evaluate_queue(queue, truth, route, system, feature_set))
        queue["system"] = system
        queue["feature_set"] = feature_set
    scored = pd.concat(
        [queue.assign(system=system, feature_set=feature_set) for (system, feature_set), queue in queues.items()],
        ignore_index=True,
    )
    gpd.GeoDataFrame(scored, geometry="geometry", crs=labeled.crs).to_parquet(
        report / "heldout_scored_physical_sites.parquet", index=False
    )
    pd.DataFrame(fold_rows).to_csv(report / "training_route_overlap_audit.csv", index=False)
    metrics = pd.DataFrame(route_rows)
    metrics.to_csv(report / "heldout_route_metrics.csv", index=False)
    minimum_rows = []
    for (system, feature_set), queue in queues.items():
        for route in routes:
            minimum_rows.append(
                {
                    "system": system,
                    "feature_set": feature_set,
                    "route": route,
                    "minimum_budget_for_70_percent": minimum_budget(queue, truth, route),
                }
            )
    pd.DataFrame(minimum_rows).to_csv(report / "minimum_budget_for_70_percent.csv", index=False)
    return metrics, pd.DataFrame(fold_rows), queues


def write_report(report, segments, truth, labeled, metrics, folds, route_counts, queues, controls):
    routes = sorted(segments.route.unique())
    full = metrics.loc[metrics.budget.astype(str).eq("full")]
    coverage = full.loc[full.system.eq("existing_rf"), ["route", "actual_culverts", "candidate_population", "matched_within_20m", "recall"]]
    labels = labeled.supervision_label.value_counts().to_dict()
    lines = [
        "# Complete surveyed-route physical-crossing experiment — 2026-10-20",
        "",
        "## Decision",
        "",
        f"This run creates genuine hard negatives only inside {len(segments)} route intervals whose endpoints are explicitly documented in daily field reports and independently resolved through NYSDOT reference-marker and measured-roadway services. It does not treat any other road as negative coverage.",
        "",
        "No experimental model is promoted. Only two distinct routes have documented endpoints in the repository, so leave-one-route-out evaluation has two folds and no genuinely untouched route remains for final validation.",
        "",
        "## Survey extent audit",
        "",
        "| Route | Documented segments | Length (km) | Recorded culverts | Physical sites | Generator matches | Generator recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route in routes:
        group = segments.loc[segments.route.eq(route)]
        row = coverage.loc[coverage.route.eq(route)].iloc[0]
        lines.append(
            f"| {route} | {len(group)} | {group.length_m.sum()/1000:.2f} | {int(row.actual_culverts)} | {int(row.candidate_population)} | {int(row.matched_within_20m)} | {row.recall:.1%} |"
        )
    total_actual = int(coverage.actual_culverts.sum())
    total_matches = int(coverage.matched_within_20m.sum())
    total_sites = int(coverage.candidate_population.sum())
    total_recall = total_matches / total_actual if total_actual else 0.0
    combined_target = int(np.ceil(0.70 * total_actual))
    ambiguous_actuals = int(truth.road_association_ambiguous.fillna(False).astype(bool).sum())
    lines += [
        "",
        "The recall denominator includes every deduplicated inventory culvert associated with each exact surveyed interval, including culverts for which the generator produced no matching site.",
        "",
        f"Across the two routes, the sealed generator can match at most {total_matches}/{total_actual} culverts ({total_recall:.1%}) using {total_sites} distinct physical sites. The 70% target requires {combined_target} matches. Therefore 70% is not reachable at any ranking budget with this candidate population." if total_matches < combined_target else f"Across the two routes, the sealed generator matches {total_matches}/{total_actual} culverts ({total_recall:.1%}) using {total_sites} distinct physical sites.",
        "",
        "### Route-level 70% feasibility",
        "",
        "| Route | Existing RF minimum | Experimental RF all-physical | Experimental HGB all-physical |",
        "|---|---:|---:|---:|",
    ]
    for route in routes:
        values = []
        for key in (("existing_rf", "all_physical"), ("rf", "all_physical"), ("hgb", "all_physical")):
            value = minimum_budget(queues[key], truth, route) if key in queues else None
            values.append(str(value) if value is not None else "Not reachable")
        lines.append(f"| {route} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "The experimental models do not improve the minimum budget over the frozen RF on NY214, and no ranking can overcome NY28's candidate-generation ceiling.",
        "",
        "## Supervision",
        "",
        f"The sealed physical population contains {len(labeled)} sites: {labels.get('positive', 0)} positives, {labels.get('negative', 0)} hard negatives, and {labels.get('uncertain', 0)} uncertain sites excluded from fitting. Positives use one-to-one matching within 20 m. Nonmatched sites within 40 m remain uncertain. Negatives are physical drainage crossings beyond 40 m from every recorded culvert on the same complete segment. {ambiguous_actuals} inventory culvert(s) have ambiguous road association; they remain in the recall denominator and protect nearby candidates from negative labels, but cannot create positive training labels.",
        "",
        "Candidate generation and feature extraction completed before either workbook-derived inventory was loaded. The same crossing generator and feature pipeline produced positives, negatives, and held-out candidates.",
        "",
        "## Held-out route evaluation",
        "",
        "`heldout_route_metrics.csv` reports each route, model, feature ablation, budget, site count, matches, recall, localization error, and predictions per road kilometre. `minimum_budget_for_70_percent.csv` reports whether each ranked queue reaches 70% on each held-out route. Random road sampling controls are in `random_road_sampling_controls.csv`.",
        "",
        "### Existing frozen RF versus random road sampling",
        "",
        "| Route | Budget | RF matches | RF recall | Random mean matches | Random mean recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline_rows = metrics.loc[metrics.system.eq("existing_rf")].drop_duplicates(
        ["route", "predicted_sites"], keep="first"
    )
    for route in routes:
        for budget in controls.loc[controls.route.eq(route), "budget"].sort_values().tolist():
            metric = baseline_rows.loc[
                baseline_rows.route.eq(route) & baseline_rows.predicted_sites.eq(budget)
            ]
            control = controls.loc[controls.route.eq(route) & controls.budget.eq(budget)]
            if metric.empty or control.empty:
                continue
            m, c = metric.iloc[0], control.iloc[0]
            lines.append(
                f"| {route} | {int(m.predicted_sites)} | {int(m.matched_within_20m)} | {m.recall:.1%} | {c.mean_matches:.2f} | {c.mean_recall:.1%} |"
            )
    lines += [
        "",
        "The existing frozen RF is scored on exactly the same physical sites for a controlled baseline. Experimental RF and histogram gradient boosting use fixed configurations, balanced sample weights, and GIS, hydrology, terrain, and all-physical ablations. A held-out route contributes neither labels nor coordinates to candidate generation, feature selection, fitting, or threshold selection.",
        "",
        "## Extent limitation",
        "",
        "The summer workbook has only latitude, longitude, region, route, team, and optional culvert ID. The official workbook has culvert reference markers but no survey-start or survey-end columns. Exact endpoints for other routes are not present in the repository, so their unmatched candidates remain unlabeled rather than negative. `inventory_routes_without_documented_extents.csv` lists those excluded route/source counts.",
        "",
        "The documented NY28 marker 336–338 interval is also excluded from negative labeling because its no-culvert field-report statement conflicts with later official inventory records inside the interval. The exclusion is preserved in `surveyed_route_extent_exclusions.csv`.",
        "",
        "## Additional field data required",
        "",
        "Freeze at least four additional geographically separated complete corridors before candidate generation. For each corridor, record the route identifier and direction, exact start/end reference markers or endpoint coordinates, survey date/team, every observed culvert with coordinate uncertainty, and verified presence/absence for every inspected physical candidate. Keep those corridors untouched through candidate design, feature selection, model tuning, and threshold selection.",
        "",
        "## Validation limitation",
        "",
        f"Only {len(routes)} complete route groups ({len(segments)} intervals) are usable. These routes have prior development exposure, and the run evaluates multiple predeclared ablations. Results are geographically separated diagnostics, not independent evidence of 70% recall. At least four additional complete corridors with explicit endpoint markers or endpoint coordinates must be frozen before candidate generation and kept untouched through model and threshold selection.",
        "",
        "Frozen models and production were not changed.",
    ]
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report, model_dir = args.output, args.model_dir
    if (report / "COMPLETE.json").exists():
        raise FileExistsError("Completed surveyed-route experiment is immutable")
    report.mkdir(parents=True, exist_ok=args.resume)
    model_dir.mkdir(parents=True, exist_ok=args.resume)
    specs = read_survey_manifest(MANIFEST)
    segments, markers, roadway = resolve_official_geometry(specs, report)
    sites = sealed_candidates(segments, report)
    sites["existing_rf_score"] = frozen_rf_scores(sites)
    sites.to_parquet(report / "sealed_physical_crossing_features.parquet", index=False)
    seal = {
        "version": VERSION,
        "candidate_count": len(sites),
        "candidate_ids_sha256": hashlib.sha256("\n".join(sorted(sites.candidate_id)).encode()).hexdigest(),
        "inventory_loaded_after_seal": True,
        "candidate_generation_uses_inventory_coordinates": False,
    }
    save_json(report / "candidate_seal.json", seal)

    records, route_counts = load_inventory_records()
    documented_routes = set(segments.route)
    route_counts.loc[~route_counts.route.isin(documented_routes)].to_csv(
        report / "inventory_routes_without_documented_extents.csv", index=False
    )
    pd.read_csv(EXTENT_EXCLUSIONS).to_csv(
        report / "surveyed_route_extent_exclusions.csv", index=False
    )
    truth = scoped_truth(records, segments, report)
    labeled, positive_matches = label_physical_sites(sites, truth)
    labeled.to_parquet(report / "supervised_physical_sites.parquet", index=False)
    labeled.drop(columns="geometry").to_csv(report / "supervised_physical_sites.csv", index=False)
    positive_matches.to_csv(report / "positive_candidate_matches.csv", index=False)
    metrics, folds, queues = train_and_evaluate(labeled, truth, segments, report, model_dir)
    site_counts = labeled.groupby("route").size().to_dict()
    controls = random_controls(segments, truth, sorted(documented_routes), site_counts)
    controls.to_csv(report / "random_road_sampling_controls.csv", index=False)
    provenance = labeled[
        ["candidate_id", "route", "segment_id", "supervision_label", "matched_actual_id"]
    ].merge(
        truth[
            [
                "actual_id",
                "inventory_sources",
                "source_record_count",
                "road_association_ambiguous",
            ]
        ],
        how="left",
        left_on="matched_actual_id",
        right_on="actual_id",
    )
    provenance.to_csv(report / "training_label_provenance.csv", index=False)
    source_audit = []
    for route, group in truth.groupby("route", sort=True):
        source_audit.append(
            {
                "route": route,
                "actual_culverts": len(group),
                "official_only": int(group.inventory_sources.eq("official").sum()),
                "summer_only": int(group.inventory_sources.eq("summer").sum()),
                "cross_source_deduplicated": int(group.inventory_sources.str.contains(",", regex=False).sum()),
                "ambiguous_road_association": int(group.road_association_ambiguous.fillna(False).astype(bool).sum()),
            }
        )
    pd.DataFrame(source_audit).to_csv(report / "inventory_source_overlap_audit.csv", index=False)
    marker_coordinates = {
        feature["properties"]["REFERENCE_MARKER_PANEL"]: feature["geometry"]["coordinates"]
        for feature in markers["features"]
    }
    segment_audit = segments.drop(columns="geometry").copy()
    segment_audit["start_x"] = [marker_coordinates[row.start_marker][0] for row in segments.itertuples()]
    segment_audit["start_y"] = [marker_coordinates[row.start_marker][1] for row in segments.itertuples()]
    segment_audit["end_x"] = [marker_coordinates[row.end_marker][0] for row in segments.itertuples()]
    segment_audit["end_y"] = [marker_coordinates[row.end_marker][1] for row in segments.itertuples()]
    segment_audit.to_csv(report / "survey_extent_audit.csv", index=False)
    write_report(report, segments, truth, labeled, metrics, folds, route_counts, queues, controls)
    input_hashes = {
        "survey_manifest": sha256_file(MANIFEST),
        "survey_extent_exclusions": sha256_file(EXTENT_EXCLUSIONS),
        "official_workbook": sha256_file(OFFICIAL_WORKBOOK),
        "official_extract": sha256_file(OFFICIAL_EXTRACT),
        "summer_extract": sha256_file(SUMMER_EXTRACT),
        "frozen_rf": sha256_file(ROOT / "models/flow_c_frozen_v1.joblib"),
        "reference_marker_snapshot": sha256_file(report / "nysdot_reference_markers_snapshot.geojson"),
        "measured_route_snapshot": sha256_file(report / "nysdot_measured_routes_snapshot.json"),
    }
    save_json(report / "input_hashes.json", input_hashes)
    complete = {
        "version": VERSION,
        "survey_segments": len(segments),
        "heldout_routes": sorted(documented_routes),
        "inventory_culverts_in_scope": len(truth),
        "physical_sites": len(labeled),
        "positive_sites": int(labeled.supervision_label.eq("positive").sum()),
        "hard_negatives": int(labeled.supervision_label.eq("negative").sum()),
        "uncertain_sites": int(labeled.supervision_label.eq("uncertain").sum()),
        "production_changed": False,
        "frozen_models_changed": False,
        "promotion_decision": "no_promotion_insufficient_independent_routes",
    }
    save_json(report / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
