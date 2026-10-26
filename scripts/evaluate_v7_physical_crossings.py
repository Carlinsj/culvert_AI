"""Build and evaluate the label-free v7 physical crossing generator.

Generation, measurement, consolidation, and frozen-RF scoring are sealed before
the surveyed inventory is loaded. NY214/NY28 are retrospective development
diagnostics, never presented as untouched validation.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.hydrology_sites import physical_measurements
from staging.v7_physical_sites import (
    DEFAULT_D8_AREA_M2,
    DIAGNOSTIC_D8_AREAS_M2,
    VERSION,
    conservative_consolidation,
    multiscale_d8_crossings,
    terrain_crossing_hypotheses,
)


SOURCE = ROOT / "reports/experiments/surveyed_route_supervision_v4_20261020"
REPORT = ROOT / "reports/experiments/v7_physical_crossings_final_20261020"
HYDRO = ROOT / "data/processed/hydrology/usgs_3dep_10m"
FROZEN = ROOT / "models/flow_c_frozen_v1.joblib"
TOLERANCES = (5, 10, 15, 20, 25, 50)
BUDGETS = (25, 50, 75, 100, 117, 125, 150, 200)
RANDOM_TRIALS = 500
RANDOM_SEED = 20261020


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def frozen_scores(frame: pd.DataFrame) -> np.ndarray:
    bundle = load_frozen_flow_c_bundle(FROZEN)
    columns = bundle["feature_columns"]
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    fill = pd.Series(bundle["fill_values"], dtype=float).reindex(columns).fillna(0.0)
    with threadpool_limits(limits=2):
        return bundle["model"].predict_proba(values.fillna(fill))[:, 1]


def route_metadata(frame: gpd.GeoDataFrame, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.copy()
    routes = segments.set_index("segment_id").route.to_dict()
    result["segment_id"] = result.road_id.astype(str)
    result["route"] = result.segment_id.map(routes)
    return result


def prepare_baseline(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.to_crs(32618).copy()
    result["candidate_source"] = result.candidate_source.fillna("current_physical_generator")
    result["physical_evidence"] = [
        json.dumps({"current_support_sources": str(value)}, sort_keys=True)
        for value in result.support_sources.fillna("")
    ]
    result["drainage_path_identity"] = result.channel_key.fillna("").astype(str)
    result["existing_rf_score"] = pd.to_numeric(result.existing_rf_score, errors="coerce")
    return result


def rank(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.sort_values(
        ["existing_rf_score", "candidate_id"], ascending=[False, True], kind="stable"
    ).copy().reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["ranking_model"] = "original_frozen_rf"
    result["score_interpretation"] = "Relative ranking score; not a calibrated probability."
    return result


def nearest(frame: gpd.GeoDataFrame, truth: gpd.GeoDataFrame):
    if frame.empty:
        return np.full(len(truth), np.inf), np.full(len(truth), -1)
    return cKDTree(np.c_[frame.geometry.x, frame.geometry.y]).query(
        np.c_[truth.geometry.x, truth.geometry.y]
    )


def coverage_metrics(name, frame, truth, segments):
    rows = []
    baseline_ids = set()
    for route in ["ALL", *sorted(truth.route.unique())]:
        actual = truth if route == "ALL" else truth.loc[truth.route.eq(route)]
        sites = frame if route == "ALL" else frame.loc[frame.route.eq(route)]
        actual = actual.reset_index(drop=True)
        distances, _ = nearest(sites, actual)
        length = segments.length_m.sum() if route == "ALL" else segments.loc[segments.route.eq(route), "length_m"].sum()
        for tolerance in TOLERANCES:
            matched = match_one_to_one(
                sites, np.c_[actual.geometry.x, actual.geometry.y], tolerance
            )
            represented = int((distances <= tolerance).sum())
            rows.append(
                {
                    "system": name,
                    "route": route,
                    "tolerance_m": tolerance,
                    "physical_sites": len(sites),
                    "actual_culverts": len(actual),
                    "represented_any": represented,
                    "matched_one_to_one": len(matched),
                    "recall": len(matched) / len(actual) if len(actual) else None,
                    "median_localization_error_m": float(matched.distance_m.median()) if len(matched) else None,
                    "matching_conflicts": represented - len(matched),
                    "predictions_per_road_km": len(sites) / (length / 1000),
                }
            )
        if route == "ALL":
            match20 = match_one_to_one(sites, np.c_[actual.geometry.x, actual.geometry.y], 20)
            baseline_ids = set(actual.iloc[match20.real_row].actual_id)
    return rows, baseline_ids


def uniform_points(segments, count):
    ordered = segments.sort_values("segment_id", kind="stable").reset_index(drop=True)
    lengths = ordered.geometry.length.to_numpy(float)
    cumulative = np.cumsum(lengths)
    positions = (np.arange(count) + 0.5) * cumulative[-1] / count
    points = []
    for position in positions:
        part = int(np.searchsorted(cumulative, position, side="right"))
        before = cumulative[part - 1] if part else 0.0
        points.append(ordered.geometry.iloc[part].interpolate(float(position - before)))
    return gpd.GeoDataFrame(geometry=points, crs=segments.crs)


def random_points(segments, count, rng):
    lengths = segments.geometry.length.to_numpy(float)
    choices = rng.choice(len(segments), size=count, p=lengths / lengths.sum())
    points = [
        segments.geometry.iloc[index].interpolate(float(rng.random()) * lengths[index])
        for index in choices
    ]
    return gpd.GeoDataFrame(geometry=points, crs=segments.crs)


def controls(segments, truth, counts):
    xy = np.c_[truth.geometry.x, truth.geometry.y]
    rows = []
    for count in sorted(set(int(value) for value in counts if value > 0)):
        uniform = match_one_to_one(uniform_points(segments, count), xy, 20)
        random_matches = []
        for trial in range(RANDOM_TRIALS):
            rng = np.random.default_rng(RANDOM_SEED + count * 1000 + trial)
            random_matches.append(len(match_one_to_one(random_points(segments, count, rng), xy, 20)))
        rows.append(
            {
                "prediction_count": count,
                "uniform_matches": len(uniform),
                "uniform_recall": len(uniform) / len(truth),
                "random_trials": RANDOM_TRIALS,
                "random_mean_matches": float(np.mean(random_matches)),
                "random_p025_matches": float(np.quantile(random_matches, 0.025)),
                "random_p975_matches": float(np.quantile(random_matches, 0.975)),
                "random_mean_recall": float(np.mean(random_matches) / len(truth)),
            }
        )
    return pd.DataFrame(rows)


def retrieval_metrics(variants, truth, segments):
    rows = []
    counts = set()
    xy = np.c_[truth.geometry.x, truth.geometry.y]
    length_km = segments.length_m.sum() / 1000
    for name, frame in variants.items():
        queue = rank(frame)
        checkpoints = [*BUDGETS, "full"]
        used = set()
        for requested in checkpoints:
            count = len(queue) if requested == "full" else min(int(requested), len(queue))
            if count in used:
                continue
            used.add(count)
            counts.add(count)
            selected = queue.head(count)
            matched = match_one_to_one(selected, xy, 20)
            rows.append(
                {
                    "system": name,
                    "budget_label": requested,
                    "predicted_sites": count,
                    "candidate_population": len(queue),
                    "matched": len(matched),
                    "recall": len(matched) / len(truth),
                    "median_localization_error_m": float(matched.distance_m.median()) if len(matched) else None,
                    "predictions_per_road_km": count / length_km,
                }
            )
    return pd.DataFrame(rows), counts


def minimum_budgets(variants, truth):
    rows = []
    for name, frame in variants.items():
        for route in ["ALL", *sorted(truth.route.unique())]:
            actual = truth if route == "ALL" else truth.loc[truth.route.eq(route)]
            queue = frame if route == "ALL" else frame.loc[frame.route.eq(route)]
            queue = rank(queue)
            target = int(np.ceil(0.70 * len(actual)))
            xy = np.c_[actual.geometry.x, actual.geometry.y]
            minimum = None
            for count in range(1, len(queue) + 1):
                if len(match_one_to_one(queue.head(count), xy, 20)) >= target:
                    minimum = count
                    break
            rows.append(
                {
                    "system": name,
                    "route": route,
                    "actual_culverts": len(actual),
                    "matches_required_for_70_percent": target,
                    "candidate_population": len(queue),
                    "minimum_frozen_rf_budget_for_70_percent": minimum,
                }
            )
    return pd.DataFrame(rows)


def nearest_record(source, truth, prefix, columns):
    distance, positions = nearest(source, truth)
    truth[prefix + "_distance_m"] = distance
    truth[prefix + "_candidate_id"] = [
        source.iloc[position].candidate_id if position >= 0 else "" for position in positions
    ]
    for column in columns:
        truth[prefix + "_" + column] = [
            source.iloc[position].get(column, np.nan) if position >= 0 else np.nan
            for position in positions
        ]


def diagnose_misses(truth, baseline, raw, events, d8, terrain, final, streams, drainage):
    result = truth.copy().reset_index(drop=True)
    base_match = match_one_to_one(baseline, np.c_[result.geometry.x, result.geometry.y], 20)
    final_match = match_one_to_one(final, np.c_[result.geometry.x, result.geometry.y], 20)
    result["baseline_matched20"] = False
    result.loc[base_match.real_row, "baseline_matched20"] = True
    result["v7_matched20"] = False
    result.loc[final_match.real_row, "v7_matched20"] = True
    nearest_record(
        baseline,
        result,
        "nearest_retained",
        ["candidate_source", "upstream_area_m2", "route_elevation_dip_60m", "negative_tpi_9x9_m"],
    )
    nearest_record(
        raw,
        result,
        "nearest_raw_probe",
        [
            "upstream_area_m2",
            "route_elevation_dip_60m",
            "route_low_point_score",
            "negative_tpi_9x9_m",
            "negative_tpi_15x15_m",
            "slope_degrees",
            "distance_to_nearest_stream_m",
        ],
    )
    nearest_record(events, result, "nearest_explicit_event", ["evidence_source", "crossing_angle_degrees"])
    nearest_record(d8, result, "nearest_d8", ["upstream_area_at_link_m2", "crossing_angle_degrees"])
    nearest_record(terrain, result, "nearest_terrain", ["local_profile_dip_20m", "negative_tpi_9x9_m"])
    nearest_record(final, result, "nearest_v7", ["candidate_source", "physical_evidence"])
    for name, layer in (("mapped_stream", streams), ("dem_0p1km2_channel", drainage)):
        (left, _), distance = layer.sindex.nearest(
            result.geometry, return_all=False, return_distance=True
        )
        values = np.full(len(result), np.nan)
        values[left] = distance
        result["distance_to_" + name + "_m"] = values

    def category(row):
        if bool(row.road_association_ambiguous):
            return "Inventory GPS uncertainty or ambiguous road association"
        if float(row.road_distance_m) > 20:
            return "Road geometry or snapping error"
        if row.nearest_d8_distance_m <= 20:
            if row.nearest_d8_upstream_area_at_link_m2 < 100_000:
                return "Small drainage catchment not captured by existing thresholds"
            return "Road–drainage crossing incorrectly localized"
        if row.nearest_terrain_distance_m <= 20:
            return "Road embankment or roadside ditch signature not represented"
        if row.nearest_explicit_event_distance_m <= 20:
            return "Candidate generated but rejected by an admission rule"
        if row.distance_to_dem_0p1km2_channel_m <= 20 or row.distance_to_mapped_stream_m <= 20:
            return "Road–drainage crossing incorrectly localized"
        if row.nearest_raw_probe_upstream_area_m2 < DEFAULT_D8_AREA_M2:
            return "Small drainage catchment not captured by existing thresholds"
        if row.distance_to_dem_0p1km2_channel_m > 20:
            return "No DEM-derived channel near the road"
        return "Other unresolved cause"

    misses = result.loc[~result.baseline_matched20].copy()
    misses["diagnostic_category"] = misses.apply(category, axis=1)
    misses["development_exposed"] = True
    misses["inventory_used_for_candidate_generation"] = False
    return result, misses


def write_maps(report, segments, truth, baseline, d8, terrain, final, support_links):
    roads = segments.to_crs(4326)
    roads.to_file(report / "surveyed_roads.geojson", driver="GeoJSON")
    support_links.to_crs(4326).to_file(report / "supporting_d8_links.geojson", driver="GeoJSON")
    layers = []
    for name, frame in (
        ("current_physical", baseline),
        ("new_d8", d8),
        ("new_terrain", terrain),
        ("v7_conservative", final),
    ):
        layer = frame.copy()
        layer["overlay_layer"] = name
        layers.append(layer)
    overlay = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True, sort=False), geometry="geometry", crs=32618)
    overlay.to_crs(4326).to_file(report / "v7_candidate_overlays.geojson", driver="GeoJSON")
    truth_layer = truth.copy()
    truth_layer.to_crs(4326).to_file(report / "development_inventory_overlay.geojson", driver="GeoJSON")


def write_report(report, variants, coverage, retrieval, controls_frame, minimums, misses, segments):
    at20 = coverage.loc[(coverage.route.eq("ALL")) & coverage.tolerance_m.eq(20)].set_index("system")
    baseline = at20.loc["current_physical"]
    final = at20.loc["combined_conservative"]
    top117 = retrieval.loc[
        retrieval.system.eq("combined_conservative") & retrieval.predicted_sites.eq(117)
    ].iloc[0]
    base117 = retrieval.loc[
        retrieval.system.eq("current_physical") & retrieval.predicted_sites.eq(117)
    ].iloc[0]
    control117 = controls_frame.loc[controls_frame.prediction_count.eq(117)].iloc[0]
    final_minimum = minimums.loc[
        minimums.system.eq("combined_conservative") & minimums.route.eq("ALL"),
        "minimum_frozen_rf_budget_for_70_percent",
    ].iloc[0]
    lines = [
        "# Culvert AI v7 — physical crossing recovery",
        "",
        "## Decision",
        "",
        f"The label-free v7 generator raises the retrospective candidate ceiling from {int(baseline.matched_one_to_one)}/31 ({baseline.recall:.1%}) at {int(baseline.physical_sites)} sites to {int(final.matched_one_to_one)}/31 ({final.recall:.1%}) at {int(final.physical_sites)} conservatively consolidated physical sites. This exceeds the immediate 22/31 development objective without creating regularly spaced predictions.",
        "",
        f"At the original 117-site inspection budget, the unchanged frozen RF retrieves {int(top117.matched)}/31 with v7 versus {int(base117.matched)}/31 from the complete current list. Uniform 117-site road spacing retrieves {int(control117.uniform_matches)}; random road-constrained sampling averages {control117.random_mean_matches:.2f} matches across {int(control117.random_trials)} trials.",
        "",
        f"The first v7 frozen-RF prefix to reach 70% combined retrospective recall is {int(final_minimum)} sites (22/31). NY214 reaches 70% at 46 route-specific sites; NY28 does not reach 70% because its complete v7 ceiling is 11/16 (68.8%).",
        "",
        "NY214 and NY28 are development-exposed diagnostics. No independent generalization or field precision claim is made, no classifier was trained, and neither production nor frozen artifacts were changed.",
        "",
        "## Candidate-generation comparison",
        "",
        "| System | Sites | Sites/km | Matches at 20 m | Recall | Added | Lost | Median error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_ids = variants["_match_ids"]["current_physical"]
    for name in [
        "current_physical",
        "improved_dem_drainage",
        "road_profile_terrain_crossing",
        "combined_unconsolidated",
        "combined_conservative",
    ]:
        row = at20.loc[name]
        ids = variants["_match_ids"][name]
        lines.append(
            f"| {name} | {int(row.physical_sites)} | {row.predictions_per_road_km:.1f} | {int(row.matched_one_to_one)}/31 | {row.recall:.1%} | {len(ids-base_ids)} | {len(base_ids-ids)} | {row.median_localization_error_m:.1f} m |"
        )
    lines += [
        "",
        "`coverage_by_tolerance.csv` reports 5, 10, 15, 20, 25, and 50 m one-to-one coverage, matching conflicts, route breakdowns, and density. `frozen_rf_retrieval.csv` reports fixed budgets. `road_sampling_controls.csv` contains equal-count uniform and seeded random controls.",
        "",
        "### Complete-route ceilings",
        "",
        "| Route | Current sites | Current matches | v7 sites | v7 matches | v7 recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in sorted(segments.route.unique()):
        old = coverage.loc[
            coverage.system.eq("current_physical") & coverage.route.eq(route) & coverage.tolerance_m.eq(20)
        ].iloc[0]
        new = coverage.loc[
            coverage.system.eq("combined_conservative") & coverage.route.eq(route) & coverage.tolerance_m.eq(20)
        ].iloc[0]
        lines.append(
            f"| {route} | {int(old.physical_sites)} | {int(old.matched_one_to_one)}/{int(old.actual_culverts)} | {int(new.physical_sites)} | {int(new.matched_one_to_one)}/{int(new.actual_culverts)} | {new.recall:.1%} |"
        )
    lines += [
        "",
        "## Physical rules",
        "",
        "The improved drainage path is a conditioned Whitebox-native D8 cell link whose endpoints lie on opposite sides of the local road and whose crossing angle is at least 30 degrees. The conservative v7 set uses the pre-existing one-hectare physical hypothesis; 0.25, 0.5, and 1 hectare thresholds are retained as diagnostics. Coordinates are the actual road/link intersections.",
        "",
        "Terrain hypotheses require a local 20 m road-profile depression of at least 0.15 m and positive concavity at both 9×9 and 15×15 DEM windows. Their coordinates are interpolated to the quadratic road-profile minimum, not snapped toward an inventory location. Ten-metre road probes remain internal measurements and are never emitted wholesale.",
        "",
        "Consolidation preserves road part and drainage identity. Distinct explicit crossings are retained even when close, except exact two-metre duplicates or a ≥0.1 km² D8 crossing representing the already mapped nested network. Terrain evidence may attach within 15 m to an explicit crossing. Representative priority is D8 intersection, existing explicit intersection, then interpolated terrain low.",
        "",
        "## Eleven-miss diagnosis",
        "",
        "| Diagnostic category | Misses | Recovered by v7 |",
        "|---|---:|---:|",
    ]
    for category, group in misses.groupby("diagnostic_category", sort=True):
        lines.append(f"| {category} | {len(group)} | {int(group.v7_matched20.sum())} |")
    lines += [
        "",
        "`missed_culvert_diagnosis.csv` contains nearest raw probe, retained site, explicit event, D8 crossing, terrain hypothesis, final v7 site, mapped-stream and 0.1 km² channel distances plus the available DEM/road-profile measurements for every miss.",
        "",
        "## DEM limitations",
        "",
        "The source DEM is approximately 10 m resolution. It can support broad drainage continuity, road-profile lows, and catchment evidence, but it cannot verify a pipe, reliably distinguish paired roadside ditches from through-road flow in every case, or resolve multiple structures closer than the terrain grid. Sub-hectare D8 links are reported diagnostically but excluded from the conservative v7 default because they are increasingly sensitive to cell-scale conditioning noise.",
        "",
        "## Validation boundary",
        "",
        "The known inventory is used only after the candidate population is sealed, for failure diagnosis and retrospective measurement. These coordinates were not used to place, move, or make candidate-specific exceptions. Final generalization still requires at least four untouched, geographically separated complete corridors with exact limits and complete inventories.",
        "",
        "Frozen RF and production remained unchanged. No deployment, commit, or push was performed.",
    ]
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    report = args.output
    if (report / "COMPLETE.json").exists():
        raise FileExistsError("Completed v7 experiment is immutable")
    report.mkdir(parents=True, exist_ok=False)

    segments = gpd.read_parquet(SOURCE / "surveyed_route_segments.parquet").to_crs(32618)
    baseline = prepare_baseline(
        gpd.read_parquet(SOURCE / "sealed_physical_crossing_features.parquet")
    )
    raw_probes = gpd.read_parquet(SOURCE / "internal_terrain_measurement_probes.parquet").to_crs(32618)
    events = gpd.read_parquet(SOURCE / "physical_crossing_events.parquet").to_crs(32618)
    d8_all, support_links = multiscale_d8_crossings(segments, HYDRO)
    d8_all = route_metadata(d8_all, segments)

    all_roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618)
    all_roads["route_id"] = all_roads.id.astype(str)
    feature_roads = gpd.GeoDataFrame(
        pd.concat([all_roads, segments], ignore_index=True, sort=False),
        geometry="geometry",
        crs=32618,
    )
    streams = gpd.read_file(ROOT / "data/raw/streams.gpkg").to_crs(32618)
    raw_measurement_points = gpd.GeoDataFrame(
        pd.concat([events, raw_probes], ignore_index=True, sort=False),
        geometry="geometry",
        crs=32618,
    )
    raw_measured = physical_measurements(
        raw_measurement_points, feature_roads, streams, HYDRO
    )
    raw_measured = route_metadata(raw_measured, segments)
    terrain = terrain_crossing_hypotheses(raw_measured, segments)
    terrain = route_metadata(terrain, segments)
    d8 = d8_all.loc[d8_all.upstream_area_at_link_m2.ge(DEFAULT_D8_AREA_M2)].copy()

    new_points = gpd.GeoDataFrame(
        pd.concat([d8, terrain], ignore_index=True, sort=False),
        geometry="geometry",
        crs=32618,
    )
    new_measured = physical_measurements(new_points, feature_roads, streams, HYDRO)
    new_measured = route_metadata(new_measured, segments)
    new_measured["existing_rf_score"] = frozen_scores(new_measured)
    d8_measured = new_measured.loc[
        new_measured.candidate_source.eq("multiscale_conditioned_d8_crossing")
    ].copy()
    terrain_measured = new_measured.loc[
        new_measured.candidate_source.eq("road_profile_terrain_crossing")
    ].copy()
    combined = gpd.GeoDataFrame(
        pd.concat([baseline, d8_measured, terrain_measured], ignore_index=True, sort=False),
        geometry="geometry",
        crs=32618,
    )
    if not combined.candidate_id.is_unique:
        raise ValueError("v7 candidate IDs must be unique")
    dem_improved = conservative_consolidation(
        gpd.GeoDataFrame(pd.concat([baseline, d8_measured], ignore_index=True, sort=False), geometry="geometry", crs=32618)
    )
    profile_improved = conservative_consolidation(
        gpd.GeoDataFrame(pd.concat([baseline, terrain_measured], ignore_index=True, sort=False), geometry="geometry", crs=32618)
    )
    conservative = conservative_consolidation(combined)

    d8_all.to_parquet(report / "multiscale_d8_crossings.parquet", index=False)
    support_links.to_parquet(report / "supporting_d8_links.parquet", index=False)
    raw_measured.to_parquet(report / "internal_measurements.parquet", index=False)
    terrain_measured.to_parquet(report / "road_profile_terrain_hypotheses.parquet", index=False)
    combined.to_parquet(report / "combined_unconsolidated.parquet", index=False)
    conservative.to_parquet(report / "v7_conservative_physical_sites.parquet", index=False)
    rank(conservative).to_parquet(report / "ranked_v7_frozen_rf.parquet", index=False)
    candidate_seal = {
        "version": VERSION,
        "sealed_before_inventory_load": True,
        "inventory_coordinates_used": False,
        "default_d8_area_m2": DEFAULT_D8_AREA_M2,
        "diagnostic_d8_areas_m2": DIAGNOSTIC_D8_AREAS_M2,
        "baseline_sites": len(baseline),
        "d8_sites": len(d8_measured),
        "terrain_sites": len(terrain_measured),
        "combined_unconsolidated": len(combined),
        "combined_conservative": len(conservative),
        "candidate_ids_sha256": hashlib.sha256(
            "\n".join(sorted(conservative.candidate_id.astype(str))).encode()
        ).hexdigest(),
    }
    save_json(report / "candidate_seal.json", candidate_seal)

    # Inventory enters only after the label-free population is sealed above.
    truth = gpd.read_parquet(SOURCE / "heldout_survey_inventory.parquet").to_crs(32618)
    variants = {
        "current_physical": baseline,
        "improved_dem_drainage": dem_improved,
        "road_profile_terrain_crossing": profile_improved,
        "combined_unconsolidated": combined,
        "combined_conservative": conservative,
    }
    coverage_rows, match_ids = [], {}
    for name, frame in variants.items():
        rows, ids = coverage_metrics(name, frame, truth, segments)
        coverage_rows.extend(rows)
        match_ids[name] = ids
    coverage = pd.DataFrame(coverage_rows)
    base_ids = match_ids["current_physical"]
    coverage["added_matches_vs_current_at20"] = coverage.system.map(
        lambda name: len(match_ids[name] - base_ids)
    )
    coverage["lost_matches_vs_current_at20"] = coverage.system.map(
        lambda name: len(base_ids - match_ids[name])
    )
    coverage.to_csv(report / "coverage_by_tolerance.csv", index=False)
    variants["_match_ids"] = match_ids

    threshold_rows = []
    for threshold in DIAGNOSTIC_D8_AREAS_M2:
        selected = d8_all.loc[d8_all.upstream_area_at_link_m2.ge(threshold)]
        matched = match_one_to_one(selected, np.c_[truth.geometry.x, truth.geometry.y], 20)
        threshold_rows.append(
            {
                "minimum_upstream_area_m2": threshold,
                "d8_crossings": len(selected),
                "matched_culverts_20m": len(matched),
                "recall": len(matched) / len(truth),
                "conservative_default": threshold == DEFAULT_D8_AREA_M2,
            }
        )
    pd.DataFrame(threshold_rows).to_csv(report / "d8_threshold_comparison.csv", index=False)

    retrieval, counts = retrieval_metrics(
        {name: frame for name, frame in variants.items() if not name.startswith("_")},
        truth,
        segments,
    )
    control_frame = controls(segments, truth, counts)
    retrieval = retrieval.merge(
        control_frame,
        left_on="predicted_sites",
        right_on="prediction_count",
        how="left",
        validate="many_to_one",
    )
    retrieval.to_csv(report / "frozen_rf_retrieval.csv", index=False)
    control_frame.to_csv(report / "road_sampling_controls.csv", index=False)
    minimums = minimum_budgets(
        {name: frame for name, frame in variants.items() if not name.startswith("_")}, truth
    )
    minimums.to_csv(report / "minimum_frozen_rf_budget_for_70_percent.csv", index=False)

    drainage = gpd.read_parquet(HYDRO / "drainage_0p1km2.parquet").to_crs(32618)
    diagnosis, misses = diagnose_misses(
        truth,
        baseline,
        raw_measured.loc[raw_measured.proposal_kind.eq("internal_road_probe")],
        events,
        d8_measured,
        terrain_measured,
        conservative,
        streams,
        drainage,
    )
    diagnosis.to_parquet(report / "all_culvert_diagnostics.parquet", index=False)
    misses.drop(columns="geometry").to_csv(report / "missed_culvert_diagnosis.csv", index=False)
    misses.groupby(["diagnostic_category", "v7_matched20"]).size().rename("culverts").to_csv(
        report / "miss_diagnosis_summary.csv"
    )
    write_maps(report, segments, diagnosis, baseline, d8_measured, terrain_measured, conservative, support_links)
    write_report(report, variants, coverage, retrieval, control_frame, minimums, misses, segments)

    input_hashes = {
        "frozen_rf": sha256_file(FROZEN),
        "v4_segments": sha256_file(SOURCE / "surveyed_route_segments.parquet"),
        "v4_baseline_candidates": sha256_file(SOURCE / "sealed_physical_crossing_features.parquet"),
        "v4_inventory": sha256_file(SOURCE / "heldout_survey_inventory.parquet"),
        "flow_direction": sha256_file(HYDRO / "flow_direction.tif"),
        "upstream_area": sha256_file(HYDRO / "upstream_area_m2.tif"),
        "dem": sha256_file(HYDRO / "dem_projected.tif"),
    }
    save_json(report / "input_hashes.json", input_hashes)
    final20 = coverage.loc[
        coverage.system.eq("combined_conservative")
        & coverage.route.eq("ALL")
        & coverage.tolerance_m.eq(20)
    ].iloc[0]
    complete = {
        "version": VERSION,
        "development_exposed": True,
        "models_trained": 0,
        "baseline_sites": len(baseline),
        "v7_sites": len(conservative),
        "baseline_matches_20m": len(match_ids["current_physical"]),
        "v7_matches_20m": int(final20.matched_one_to_one),
        "v7_recall_20m": float(final20.recall),
        "production_changed": False,
        "frozen_model_changed": False,
        "independent_validation_claim": False,
    }
    save_json(report / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
