"""Fixed-spec retrospective whole-corridor validation; never modify production."""
# ruff: noqa: E402 -- set thread limits and disable bytecode before research imports
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.candidates import generate_road_route_candidates  # noqa: E402
from culvert_ai.features import build_feature_table  # noqa: E402
from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file  # noqa: E402
from culvert_ai.inference import (  # noqa: E402
    RouteZConfig, add_dem_crossing_context, add_mapped_crossing_context,
    add_route_local_density_normalization,
)
from culvert_ai.model import _feature_fill_values, _prepare_features  # noqa: E402
from staging.ranking_policy import WARNING, rank_candidates  # noqa: E402

EXP = ROOT / "reports/experiments"
PRIOR = EXP / "site_challenger_20261015"
SOURCE = ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg"
MODEL = ROOT / "models/flow_c_frozen_v1.joblib"
HYDRO = ROOT / "data/processed/hydrology/usgs_3dep_10m"
WORKBOOK = ROOT / "data/incoming_nysdot/Small Culverts_R8_2026.07.14.xlsx"
MATCHES = EXP / "nysdot_ulster_supervised_v1_20260917/official_positive_matches.csv"
BUNDLED_PYTHON = Path("/Users/Carli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
GENERATED = {"route_interval_sample", "exact_road_stream_intersection", "nearest_road_stream_approach"}
KS = (10, 25, 50, 100, 250)
PURGE_M = 100.0
OUTSIDE = "<outside_recovered_corridors>"


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norm_route(value):
    value = str(value).strip().upper()
    if value in {"NAN", "NONE", "NULL", "", "<UNASSIGNED>"}:
        return "<unassigned>"
    for prefix in ("NY", "US"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def fold_for(route):
    """Outcome-independent assignment, fixed before any held-out predictions."""
    return int(hashlib.sha256(("culvert-heldout-v1:" + route).encode()).hexdigest()[:8], 16) % 3


def nearest_lines(points, roads):
    (left, right), distances = roads.sindex.nearest(points.geometry, return_all=True, return_distance=True)
    pairs = pd.DataFrame(dict(row=left, road_pos=right, distance_m=distances))
    pairs["road_id"] = roads.iloc[right].segment_id.to_numpy()
    pairs = pairs.sort_values(["row", "distance_m", "road_id"], kind="stable").drop_duplicates("row")
    return pairs.set_index("row").reindex(range(len(points)))


def match_one_to_one(sites, truth_xy, radius=20.0):
    """Deterministic maximum-cardinality matching, canonical input ordering."""
    if len(sites) == 0 or len(truth_xy) == 0:
        return pd.DataFrame(columns=["site_row", "real_row", "distance_m"])
    tree = cKDTree(truth_xy)
    rows, columns = [], []
    xy = np.c_[sites.geometry.x, sites.geometry.y]
    for row, point in enumerate(xy):
        for column in sorted(tree.query_ball_point(point, radius)):
            rows.append(row)
            columns.append(column)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)),
                       shape=(len(sites), len(truth_xy)))
    graph.sort_indices()
    matching = maximum_bipartite_matching(graph, perm_type="column")
    selected = np.flatnonzero(matching >= 0)
    actual = matching[selected]
    return pd.DataFrame(dict(site_row=selected, real_row=actual,
                             distance_m=np.linalg.norm(xy[selected] - truth_xy[actual], axis=1)))


def coverage(candidates, truth):
    if len(candidates) == 0:
        return np.full(len(truth), np.inf)
    return cKDTree(np.c_[candidates.geometry.x, candidates.geometry.y]).query(
        np.c_[truth.geometry.x, truth.geometry.y])[0]


def protected_hashes():
    old = json.loads((ROOT / "reports/staging/ranking_policy_correction_20261015/input_integrity.json").read_text())["after"]
    paths = {ROOT / p for p in old}
    paths.update([WORKBOOK, ROOT / "data/raw/roads.gpkg", ROOT / "data/raw/streams.gpkg", MATCHES,
                  ROOT / "data/raw/ulster_county_boundary.gpkg"])
    for folder in (PRIOR, ROOT / "reports/staging/ranking_policy_correction_20261015"):
        paths.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in str(p))
    paths.update(p for p in HYDRO.glob("*") if p.is_file())
    paths.add(EXP / "nysdot_ulster_hydrology_v5_20260917/experimental_hydrology_features.geoparquet")
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def inventory_audit(out, frame, roads):
    subprocess.run([str(BUNDLED_PYTHON), str(ROOT / "scripts/extract_inventory_for_validation.py"),
                    "--workbook", str(WORKBOOK), "--output", str(out / "inventory_extract.json")], check=True)
    extract = json.loads((out / "inventory_extract.json").read_text())
    raw = pd.DataFrame(extract["records"])
    raw["source_x"] = pd.to_numeric(raw.x, errors="coerce")
    raw["source_y"] = pd.to_numeric(raw.y, errors="coerce")
    assert extract["workbook_records"] == 3028 and len(raw) == 714
    assert np.isfinite(raw[["source_x", "source_y"]].to_numpy(float)).all()
    real = gpd.GeoDataFrame(raw, geometry=gpd.points_from_xy(raw.source_x, raw.source_y), crs=26918).to_crs(32618)
    real["declared_route"] = real.Route.map(norm_route)
    # Check the original sheet rows/coordinates, not Culvert ID, against old mappings.
    old = pd.read_csv(MATCHES).set_index("official_source_row_id").loc[real.source_excel_row]
    np.testing.assert_allclose(old.official_source_x, real.source_x, rtol=0, atol=1e-7)
    np.testing.assert_allclose(old.official_source_y, real.source_y, rtol=0, atol=1e-7)
    historic = pd.read_csv(EXP / "nysdot_ulster_real_inventory_benchmark_v1_20261006/source_extract/small_culverts_r8_source.csv")
    ulster = historic.loc[historic.County.astype(str).str.upper().eq("ULSTER")]
    assert len(ulster) == 714
    np.testing.assert_allclose(ulster[["x", "y"]].to_numpy(float), raw[["source_x", "source_y"]], rtol=0, atol=1e-7)
    nearest = nearest_lines(real, roads)
    real["nearest_segment_id"] = nearest.road_id.to_numpy()
    real["distance_to_recovered_road_m"] = nearest.distance_m.to_numpy()
    real["corridor"] = roads.iloc[nearest.road_pos.to_numpy(int)].corridor.to_numpy()
    real.loc[real.distance_to_recovered_road_m.gt(100), "corridor"] = OUTSIDE
    real["fold"] = real.corridor.map(lambda r: -1 if r == OUTSIDE else fold_for(r))
    boundary = gpd.read_file(ROOT / "data/raw/ulster_county_boundary.gpkg").to_crs(32618).geometry.union_all()
    real["inside_original_study_boundary"] = real.geometry.map(boundary.covers)
    rf = pd.read_csv(MODEL.with_suffix(".joblib.training_records.csv"))
    official_ids = set(rf.loc[rf.experiment_label_source.eq("nysdot_official_match_le_25m"), "candidate_id"])
    real["rf_mapping_candidate_id"] = old.candidate_id.to_numpy()
    real["rf_original_positive_distance_m"] = old.match_distance_m.to_numpy()
    real["rf_direct_inventory_label"] = old.accepted_official_positive.to_numpy() & old.candidate_id.isin(official_ids).to_numpy()
    field_ids = rf.loc[rf.experiment_label_source.eq("field_confirmed_culvert"), "candidate_id"]
    field = frame.loc[frame.candidate_id.isin(field_ids)]
    real["rf_field_positive_nearest_m"] = coverage(field, real)
    real["rf_field_positive_spatial_overlap_25m"] = real.rf_field_positive_nearest_m.le(25)
    hgb = pd.read_csv(PRIOR / "final_training_rows.csv")
    hgb_sites = set(hgb.loc[hgb.target.eq(1), "site_key"])
    real["hgb_direct_inventory_label"] = real.record_key.isin(hgb_sites)
    train_corridors = set(frame.loc[frame.candidate_id.isin(rf.candidate_id), "corridor"]) - {"<unassigned>"}
    real["road_represented_in_rf_training"] = real.corridor.isin(train_corridors)
    real["prior_performance_or_policy_development_use"] = True  # All 714 rows verified above.
    real["genuinely_untouched_test"] = False
    real["A_direct_inventory_positive_any_model"] = real.rf_direct_inventory_label | real.hgb_direct_inventory_label
    real["B_prior_model_or_policy_evaluation"] = True
    real["C_not_direct_positive_but_rf_training_corridor"] = ~real.A_direct_inventory_positive_any_model & real.road_represented_in_rf_training
    real["D_untouched_heldout_corridor"] = False
    real["E_uncertain_historical_use"] = False
    real["primary_provenance_class"] = np.where(real.A_direct_inventory_positive_any_model, "A_direct_positive_and_B_prior_development", "B_prior_development")
    real.drop(columns="geometry").to_csv(out / "inventory_training_overlap.csv", index=False)
    rf_geom = frame.set_index("candidate_id").loc[rf.candidate_id].copy().reset_index(drop=True)
    rf_detail = rf.copy()
    rf_detail["label_free_corridor"] = rf_geom.corridor.to_numpy()
    rf_detail["nearest_inventory_m"] = coverage(real, rf_geom)
    rf_detail["official_records_directly_linked"] = rf.candidate_id.map(
        pd.Series(real.rf_direct_inventory_label.to_numpy(), index=real.rf_mapping_candidate_id).groupby(level=0).sum()
    ).fillna(0).astype(int)
    rf_detail.to_csv(out / "rf_training_rows_spatial_audit.csv", index=False)
    hgb.assign(inventory_record_exists=hgb.site_key.isin(set(real.record_key))).to_csv(out / "hgb_training_row_audit.csv", index=False)
    ids = real["Small Culvert ID"].replace(["null", ""], np.nan)
    counts = dict(workbook_records=3028, ulster_records=714, valid_coordinates=714,
                  missing_culvert_id=int(ids.isna().sum()), repeated_nonmissing_id_groups=int(ids.dropna().value_counts().gt(1).sum()),
                  duplicate_xy_records=int(real.duplicated(["source_x", "source_y"], keep=False).sum()),
                  declared_named_routes=int(real.declared_route.nunique()-1),
                  rf_total_training_rows=len(rf), rf_official_positive_rows=len(official_ids),
                  rf_distinct_inventory_records_direct=int(real.rf_direct_inventory_label.sum()),
                  rf_field_positive_rows=len(field_ids), rf_field_positive_spatial_overlap_inventory=int(real.rf_field_positive_spatial_overlap_25m.sum()),
                  hgb_training_rows=len(hgb), hgb_positive_rows=int(hgb.target.sum()), hgb_inventory_records_direct=int(real.hgb_direct_inventory_label.sum()),
                  any_direct_inventory_positive=int(real.A_direct_inventory_positive_any_model.sum()),
                  prior_development_inventory=714, union_training_or_selection_inventory=714,
                  genuinely_untouched_inventory=0, genuinely_untouched_corridors=0,
                  corridor_assigned=int(real.fold.ge(0).sum()), outside_recovered_corridors=int(real.fold.lt(0).sum()),
                  outside_original_study_boundary=int((~real.inside_original_study_boundary).sum()),
                  original_label_radius_m=25, current_evaluation_radius_m=20,
                  source_crs="EPSG:26918", evaluation_crs="EPSG:32618",
                  source_crs_evidence="NYSDOT Environmental Viewer Directory, revised 2026-04-08, page 11, Small Culverts",
                  source_crs_url="https://www.dot.ny.gov/divisions/engineering/environmental-analysis/manuals-and-guidance/epm/repository/Environmental_Viewer_Directory.pdf",
                  spreadsheet_skill="Read-only bundled openpyxl extraction; only eight permitted identification/location fields exported; source unchanged.")
    save_json(out / "inventory_integrity.json", counts)
    return real, counts


def recover_features():
    """Recompute no labels: join preserved numeric GIS artifacts, remove route oracles."""
    frame = gpd.read_file(SOURCE)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    samples = frame.loc[frame.source.eq("route_interval_sample")].copy()
    routes = frame.matched_route.map(norm_route)
    distance, nearest = cKDTree(np.c_[samples.geometry.x, samples.geometry.y]).query(np.c_[frame.geometry.x, frame.geometry.y])
    missing = routes.eq("<unassigned>") & (distance <= 100)
    routes.loc[missing] = samples.iloc[nearest[missing]].matched_route.map(norm_route).to_numpy()
    frame["corridor"] = routes
    hydro = pd.read_parquet(EXP / "nysdot_ulster_hydrology_v5_20260917/experimental_hydrology_features.geoparquet").drop(columns="geometry")
    frame = frame.drop(columns=[c for c in hydro if c != "candidate_id" and c in frame]).merge(hydro, on="candidate_id", validate="one_to_one")
    v7 = load_module("heldout_fixed_v7_features", EXP / "nysdot_ulster_dem_hydrology_v7_completed_20260924/run_experiment.py")
    frame = v7._add_dem_features(frame)  # Pure artifact join/log transforms, not preparation/training.
    frame = add_route_local_density_normalization(frame, route_column="corridor")
    return frame, samples, v7.V2


def recover_roads(samples):
    roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618)
    roads["segment_id"] = roads.id.astype(str)
    mapping = samples.assign(segment_id=samples.road_id.astype(str)).groupby("segment_id").matched_route.agg(lambda x: sorted(set(x))[0])
    assert samples.groupby("road_id").matched_route.nunique().eq(1).all()
    selected = roads.loc[roads.segment_id.isin(mapping.index)].copy().sort_values("segment_id").reset_index(drop=True)
    assert len(selected) == len(mapping)
    selected["corridor"] = selected.segment_id.map(mapping).map(norm_route)
    selected["fold"] = selected.corridor.map(fold_for)
    return selected, roads


def current_features(roads, all_roads, out, columns):
    candidates = generate_road_route_candidates(roads, include_all_roads=True, route_id_column="corridor",
                                                interval_m=75, lateral_offsets_m=(0.0,))
    candidates["candidate_id"] = "eval75_" + candidates.candidate_id
    candidates["route_id"] = candidates.matched_route.astype(str)
    streams = gpd.read_file(ROOT / "data/raw/streams.gpkg").to_crs(32618)
    print(f"Computing existing 48 features for {len(candidates)} current 75 m candidates", flush=True)
    features = build_feature_table(candidates, roads=all_roads, streams=streams,
                                   dem_path=HYDRO / "dem_projected.tif",
                                   flow_accumulation_path=HYDRO / "flow_accumulation_cells.tif",
                                   drainage_area_path=HYDRO / "upstream_area_m2.tif",
                                   density_radius_m=75, density_radii_m=(100, 250, 500))
    features = add_mapped_crossing_context(features, all_roads, streams)
    crossings = {k: gpd.read_parquet(HYDRO / f"road_drainage_crossings_{k}.parquet")
                 for k in ("0p1km2", "0p5km2", "2km2", "10km2")}
    features = add_dem_crossing_context(features, crossings)
    features = add_route_local_density_normalization(features, route_column="route_id")
    assert set(columns).issubset(features.columns)
    features["corridor"] = features.route_id
    features["fold"] = features.corridor.map(fold_for)
    # No inspection/label metadata can enter the persisted inference matrix.
    features = features[["candidate_id", "road_id", "corridor", "fold", *columns, "geometry"]]
    features.to_parquet(out / "current75_features.parquet", index=False)
    return features


def safe_training_frame(frame, roads, heldout):
    """Exclude complete corridors AND a buffer around their full line geometry."""
    held_geometry = roads.loc[roads.corridor.isin(heldout)].geometry.union_all()
    safe = frame.loc[~frame.corridor.isin(heldout) & frame.corridor.ne("<unassigned>")].copy()
    safe["distance_to_heldout_road_m"] = shapely.distance(safe.geometry.array, held_geometry)
    return safe.loc[safe.distance_to_heldout_road_m.gt(PURGE_M)].copy().reset_index(drop=True)


def rf_training(safe, real_train, original_matches, sampler):
    """Same fixed hard-negative procedure, rerun using TRAINING labels only."""
    safe = safe.copy()
    accepted = original_matches.loc[original_matches.accepted_official_positive
                                    & original_matches.official_source_row_id.isin(real_train.source_excel_row)]
    official_ids = set(accepted.candidate_id)
    safe["training_label_class"] = "unlabeled_candidate"
    safe["experiment_label_source"] = "ordinary_unlabeled_candidate"
    field_pos = safe.source.eq("field_report_observed_culvert")
    negative_ids = set(pd.read_csv(MODEL.with_suffix(".joblib.training_records.csv")).query("training_label_class == 'verified_negative'").candidate_id)
    safe.loc[field_pos, "training_label_class"] = "verified_positive"
    safe.loc[field_pos, "experiment_label_source"] = "field_confirmed_culvert"
    safe.loc[safe.candidate_id.isin(negative_ids), "training_label_class"] = "verified_negative"
    safe.loc[safe.candidate_id.isin(negative_ids), "experiment_label_source"] = "field_confirmed_no_culvert"
    safe.loc[safe.candidate_id.isin(official_ids), "training_label_class"] = "verified_positive"
    safe.loc[safe.candidate_id.isin(official_ids), "experiment_label_source"] = "nysdot_official_match_le_25m"
    safe["experiment_route"] = safe.corridor
    count = int(safe.training_label_class.eq("verified_positive").sum())
    hard = sampler._hard_negative_sample(safe, sample_size=count)
    train = sampler._selected_hard(safe, hard).reset_index(drop=True)
    train["target"] = train.experiment_target.astype(int)
    train["label_source"] = train.experiment_label_source
    train["site_key"] = train.candidate_id
    train["weight"] = 1.0
    train["route"] = train.corridor
    return train


def hgb_training(safe, real_train):
    parts = []
    for route in sorted(real_train.corridor.unique()):
        part = safe.loc[safe.corridor.eq(route) & safe.source.isin(GENERATED)].copy()
        official = real_train.loc[real_train.corridor.eq(route)]
        if part.empty or official.empty:
            continue
        distance, nearest = cKDTree(np.c_[official.geometry.x, official.geometry.y]).query(np.c_[part.geometry.x, part.geometry.y])
        use = distance <= 20
        part = part.loc[use].copy()
        part["site_key"] = official.iloc[nearest[use]].record_key.to_numpy()
        part["target"] = 1
        part["label_source"] = "authoritative_inventory_positive_20m"
        parts.append(part)
    negative_ids = set(pd.read_csv(MODEL.with_suffix(".joblib.training_records.csv")).query("training_label_class == 'verified_negative'").candidate_id)
    neg = safe.loc[safe.candidate_id.isin(negative_ids)].copy()
    neg["site_key"] = "field_negative_" + neg.candidate_id
    neg["target"] = 0
    neg["label_source"] = "field_verified_negative"
    train = pd.concat([*parts, neg], ignore_index=True).drop_duplicates("candidate_id").reset_index(drop=True)
    train["weight"] = 1 / train.groupby(["target", "site_key"]).candidate_id.transform("size")
    train["route"] = train.corridor
    return train


def rank_sites(frame, score, policy, reference):
    scored = frame[["candidate_id", "corridor", "geometry"]].rename(columns={"corridor": "route_id"}).copy()
    scored["flow_c_score"] = score
    sites = rank_candidates(scored, policy=policy, route_z_config=reference)
    return sites.rename(columns={"flow_c_score": "model_score", "route_id": "corridor"})


def evaluate(sites, candidates, truth, context):
    xy = np.c_[truth.geometry.x, truth.geometry.y]
    rows = []
    full = match_one_to_one(sites, xy)
    raw_dist = coverage(candidates, truth)
    for k in (*KS, len(sites)):
        head = sites.head(k)
        matched = match_one_to_one(head, xy)
        counts = head.corridor.value_counts()
        rows.append(dict(**context, k=k, actual_denominator=len(head), official_culverts=len(truth),
                         candidates=len(candidates), consolidated_sites=len(sites),
                         raw_candidate_covered=int((raw_dist <= 20).sum()), full_list_matches=len(full),
                         matched=len(matched), inventory_matched_fraction=len(matched)/len(head) if len(head) else None,
                         inventory_recall=len(matched)/len(truth) if len(truth) else None,
                         inventory_unmatched_recommendations=len(head)-len(matched), routes_in_top_k=len(counts),
                         largest_route=str(counts.index[0]) if len(counts) else None,
                         largest_route_share=float(counts.iloc[0]/len(head)) if len(head) else None))
    return rows


def diagnostics(sites, candidates, truth, context, columns):
    xy = np.c_[truth.geometry.x, truth.geometry.y]
    full = match_one_to_one(sites, xy)
    top = match_one_to_one(sites.head(250), xy)
    result = truth[["record_key", "source_excel_row", "declared_route", "corridor", "nearest_segment_id",
                    "inside_original_study_boundary", "geometry"]].copy().reset_index(drop=True)
    result["raw_candidate_distance_m"] = coverage(candidates, result)
    result["representative_distance_m"] = coverage(sites, result)
    tree = cKDTree(np.c_[sites.geometry.x, sites.geometry.y])
    first = [min((int(sites.iloc[i].global_rank) for i in tree.query_ball_point(point, 20)), default=None) for point in xy]
    result["first_geometric_representative_rank"] = first
    result["full_one_to_one_matched"] = result.index.isin(full.real_row)
    result["top250_one_to_one_matched"] = result.index.isin(top.real_row)
    result["failure_category"] = np.select(
        [result.raw_candidate_distance_m.gt(20), result.representative_distance_m.gt(20),
         ~result.full_one_to_one_matched, ~result.top250_one_to_one_matched],
        ["A_no_candidate_20m", "B_no_surviving_representative_20m", "B_one_to_one_capacity_loss", "C_ranked_below_budget"],
        default="matched_at_250")
    for k in KS:
        assignment = match_one_to_one(sites.head(k), xy)
        result[f"matched_at_{k}"] = result.index.isin(assignment.real_row)
    for key, value in context.items():
        result[key] = value
    high = sites.head(250).copy()
    high["inventory_matched_at_250"] = high.index.isin(top.site_row)
    high["nearest_inventory_distance_m"] = coverage(truth, high)
    high["unmatched_reason"] = np.where(high.nearest_inventory_distance_m.gt(20), "no_inventory_record_within_20m", "one_to_one_capacity_or_duplicate")
    high = high.merge(pd.DataFrame(candidates[["candidate_id", *columns]]), on="candidate_id", validate="one_to_one")
    for key, value in context.items():
        high[key] = value
    return result.drop(columns="geometry"), high.drop(columns="geometry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXP / "real_inventory_heldout_roads_20261015")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output
    if (out / "COMPLETE.json").exists():
        raise FileExistsError("Completed baseline is immutable; use a new output directory.")
    out.mkdir(parents=True, exist_ok=args.resume)
    before_path = out / "protected_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
        assert before == protected_hashes()
    else:
        before = protected_hashes()
        save_json(before_path, before)
    bundle = load_frozen_flow_c_bundle(MODEL)
    columns = bundle["feature_columns"]
    hgb_bundle = joblib.load(PRIOR / "challenger.joblib")
    config = hgb_bundle["configuration"]
    helper = load_module("heldout_fixed_hgb_procedure", ROOT / "scripts/train_site_challenger.py")
    print("Recovering original features, road geometry and training provenance", flush=True)
    frame, samples, sampler = recover_features()
    roads, all_roads = recover_roads(samples)
    roads.to_parquet(out / "recovered_road_segments.parquet", index=False)
    if (out / "inventory_extract.json").exists() and args.resume:
        # Only our derivative extraction is reused; integrity was checked above.
        inventory_path = out / "inventory_extract.json"
        inventory_path.rename(out / "inventory_extract_previous_attempt.json")
    real, inventory_summary = inventory_audit(out, frame, roads)
    original_matches = pd.read_csv(MATCHES)
    plan = dict(classification="retrospective_development_validation_not_untouched_test",
                route_folds={r: fold_for(r) for r in sorted(roads.corridor.unique())},
                split_rule="sha256('culvert-heldout-v1:' + corridor), first 8 hex digits modulo 3",
                spatial_purge_m=PURGE_M, primary_candidates="current generator: 75m, midpoint fallback, zero lateral offset",
                secondary_candidates="all preserved generated 10m/crossing candidates, not only prior 51417 score pool",
                geometry_scope="all 121 complete road segments that supplied frozen interval candidates, independent of inventory proximity",
                feature_context="countywide existing roads/streams; existing 48-feature functions; unlabeled route normalization; no label-derived route overrides",
                rf_parameters=bundle["model_parameters"], rf_fit="training-only median fill; unchanged 800-tree estimator; no sample weights; regenerate fixed hard negatives from training roads only",
                rf_positive_labels="original accepted <=25m inventory mappings plus recorded field positives, training corridors only",
                rf_negative_labels="recorded verified negatives plus original fixed route-balanced hard-weak 1:1 selection, not verified absence",
                hgb_configuration=config, hgb_fit="existing fixed HGB_REGULAR and route/site/class weights; training-route <=20m positives; verified negatives only",
                hgb_policy="raw_rf adapter applied to HGB numeric scores; no RF estimator used for HGB",
                normalization_fallback="training-road predictions only, no frozen all-development OOF fallback",
                matching="deterministic maximum one-to-one within20m, geometry not CulvertID, independently at every K",
                official_assignment="nearest recovered full road segment <=100m, tie by segment ID; declared Route retained separately; outside records not reassigned to a distant road",
                scores_pooled_across_models=False, hyperparameter_search=False, controlled_improvement=False,
                production_modified=False, frozen_rf_sha256=sha256_file(MODEL), frozen_hgb_sha256=sha256_file(PRIOR / "challenger.joblib"))
    plan_path = out / "preregistered_method.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == json.loads(json.dumps(plan))
    else:
        save_json(plan_path, plan)
    real.drop(columns="geometry").to_csv(out / "inventory_split_manifest.csv", index=False)
    roads.drop(columns="geometry").to_csv(out / "segment_split_manifest.csv", index=False)
    generated = frame.loc[frame.source.isin(GENERATED) & frame.corridor.ne("<unassigned>")].copy()
    generated["fold"] = generated.corridor.map(fold_for)
    generated["road_id"] = generated.road_id.astype(str)
    generated[["candidate_id", "road_id", "corridor", "fold", *columns, "geometry"]].to_parquet(out / "historical10_features.parquet", index=False)
    current_path = out / "current75_features.parquet"
    current = gpd.read_parquet(current_path) if args.resume and current_path.exists() else current_features(roads, all_roads, out, columns)
    save_json(out / "candidate_generator_audit.json", dict(
        historical_source_interval_m=float(samples.groupby("road_id").route_sample_distance_m.diff().dropna().median()),
        current_bundle_and_code_interval_m=75, current_candidates=len(current),
        historical_full_generated_candidates=len(generated), earlier_saved_score_pool=51417,
        current_feature_hydrology_edge_risk_default=int(current.hydrology_edge_risk.max()),
        historical_hydrology_edge_risk_fraction=float(generated.hydrology_edge_risk.mean()),
        road_segments=len(roads), corridor_groups=int(roads.corridor.nunique()),
        note="Do not treat the historical 10m and deployed 75m populations as equivalent. Current hydrology edge-risk default is inherited unchanged from production code, not corrected here."))
    metrics, per_route, concentrations, misses, high_rows, score_distributions, leakages = [], [], [], [], [], [], []
    for fold in range(3):
        heldout = set(roads.loc[roads.fold.eq(fold), "corridor"])
        truth = real.loc[real.fold.eq(fold)].sort_values("record_key").reset_index(drop=True)
        train_real = real.loc[real.fold.ge(0) & real.fold.ne(fold)].copy()
        safe = safe_training_frame(frame, roads, heldout)
        rf_train = rf_training(safe, train_real, original_matches, sampler)
        hgb_train = hgb_training(safe, train_real)
        assert not (set(rf_train.corridor) | set(hgb_train.corridor)) & heldout
        assert not set(hgb_train.site_key) & set(truth.record_key)
        heldout_official_ids = set(original_matches.loc[original_matches.official_source_row_id.isin(truth.source_excel_row), "candidate_id"])
        # The spatial corridor buffer must also exclude old official training rows
        # even when legacy route labels would have put them in a different fold.
        assert not set(rf_train.candidate_id) & heldout_official_ids
        for name, train in (("RF", rf_train), ("HGB", hgb_train)):
            assert train.distance_to_heldout_road_m.gt(PURGE_M).all()
            train[["candidate_id", "corridor", "target", "label_source", "site_key", "weight", "distance_to_heldout_road_m"]].to_csv(out / f"training_manifest_{name}_fold{fold}.csv", index=False)
            leakages.append(dict(model=name, fold=fold, training_rows=len(train), positives=int(train.target.sum()),
                                 negative_rows=int(train.target.eq(0).sum()), heldout_corridors=sorted(heldout),
                                 min_training_distance_to_heldout_road_m=float(train.distance_to_heldout_road_m.min()),
                                 heldout_inventory_records=len(truth), heldout_labels_in_training=0))
        rf_path, hgb_path = out / f"evaluation_copy_rf_fold{fold}.joblib", out / f"evaluation_copy_hgb_fold{fold}.joblib"
        if args.resume and rf_path.exists():
            fitted_rf = joblib.load(rf_path)
        else:
            medians = _feature_fill_values(rf_train, columns)
            fitted_rf = dict(model=clone(bundle["model"]), medians=medians)
            print(f"Fitting evaluation RF copy {fold}: {len(rf_train)} rows; held out {sorted(heldout)}", flush=True)
            fitted_rf["model"].fit(_prepare_features(rf_train, columns, medians), rf_train.target)
            joblib.dump(fitted_rf, rf_path, compress=3)
        if args.resume and hgb_path.exists():
            fitted_hgb = joblib.load(hgb_path)
        else:
            print(f"Fitting fixed HGB evaluation copy {fold}: {len(hgb_train)} rows", flush=True)
            with threadpool_limits(limits=2):
                fitted_hgb = helper.fit_model(config, hgb_train, columns)
            joblib.dump(fitted_hgb, hgb_path, compress=3)
        train_scores = fitted_rf["model"].predict_proba(_prepare_features(safe, columns, fitted_rf["medians"]))[:, 1]
        reference = RouteZConfig(global_center=float(train_scores.mean()), global_scale=max(float(train_scores.std(ddof=0)), 1e-6))
        save_json(out / f"normalization_reference_fold{fold}.json", dict(center=reference.global_center, scale=reference.global_scale, source="training_corridor_predictions_only"))
        for cohort, population in (("current75", current), ("historical10", generated)):
            test = population.loc[population.fold.eq(fold)].sort_values("candidate_id").reset_index(drop=True)
            scores = {}
            scores["RF"] = fitted_rf["model"].predict_proba(_prepare_features(test, columns, fitted_rf["medians"]))[:, 1]
            with threadpool_limits(limits=2):
                scores["HGB"] = helper.score_model(fitted_hgb, test, columns)
            predictions = test[["candidate_id", "road_id", "corridor", "geometry"]].copy()
            for model, score in scores.items():
                predictions[model+"_score"] = score
            predictions.to_parquet(out / f"heldout_predictions_{cohort}_fold{fold}.parquet", index=False)
            for model, policy in (("RF", "raw_rf"), ("RF", "dynamic_route_z"), ("HGB", "raw_rf")):
                context = dict(cohort=cohort, fold=fold, model=model, policy=policy)
                sites = rank_sites(test, scores[model], policy, reference)
                sites.to_parquet(out / f"sites_{cohort}_{model}_{policy}_fold{fold}.parquet", index=False)
                metrics.extend(evaluate(sites, test, truth, context))
                missed, high = diagnostics(sites, test, truth, context, columns)
                misses.append(missed)
                high_rows.append(high)
                for route in sorted(heldout):
                    road_sites = sites.loc[sites.corridor.eq(route)].copy().reset_index(drop=True)
                    road_truth = truth.loc[truth.corridor.eq(route)].copy().reset_index(drop=True)
                    road_candidates = test.loc[test.corridor.eq(route)]
                    per_route.extend(evaluate(road_sites, road_candidates, road_truth, dict(**context, corridor=route)))
                    distribution = predictions.loc[predictions.corridor.eq(route), model+"_score"]
                    score_distributions.append(dict(**context, corridor=route, count=len(distribution), mean=float(distribution.mean()),
                                                    std=float(distribution.std(ddof=0)), median=float(distribution.median()),
                                                    minimum=float(distribution.min()), maximum=float(distribution.max())))
                    concentrations.append(dict(**context, corridor=route, candidates=len(road_candidates),
                                               known_inventory=len(road_truth), representatives=len(road_sites),
                                               first_global_rank=int(road_sites.global_rank.min()) if len(road_sites) else None,
                                               top100=int(road_sites.global_rank.le(100).sum()),
                                               top250=int(road_sites.global_rank.le(250).sum()),
                                               share250=float(road_sites.global_rank.le(250).sum()/min(250, len(sites))),
                                               neglected_top250=not bool(road_sites.global_rank.le(250).any())))
            print(f"Evaluated fold {fold} {cohort}: {len(test)} candidates / {len(truth)} official records", flush=True)
    pd.DataFrame(metrics).to_csv(out / "heldout_group_metrics.csv", index=False)
    pd.DataFrame(per_route).to_csv(out / "heldout_road_metrics.csv", index=False)
    pd.DataFrame(concentrations).to_csv(out / "route_concentration.csv", index=False)
    pd.concat(misses, ignore_index=True).to_csv(out / "inventory_failure_diagnostics.csv", index=False)
    pd.concat(high_rows, ignore_index=True).to_csv(out / "top250_site_diagnostics.csv", index=False)
    pd.DataFrame(score_distributions).to_csv(out / "route_score_distributions.csv", index=False)
    save_json(out / "leakage_audit.json", leakages)
    summaries = []
    table = pd.DataFrame(metrics)
    for key, part in table.loc[table.k.isin(KS)].groupby(["cohort", "model", "policy", "k"]):
        denominator = int(part.actual_denominator.sum())
        official = int(part.official_culverts.sum())
        matched = int(part.matched.sum())
        summaries.append(dict(cohort=key[0], model=key[1], policy=key[2], k_per_group=int(key[3]), groups=3,
                              total_budget=denominator, matched=matched, evaluated_inventory=official,
                              matched_fraction=matched/denominator, inventory_recall=matched/official,
                              macro_matched_fraction=float(part.inventory_matched_fraction.mean()),
                              pooled_global_top_k=False))
    pd.DataFrame(summaries).to_csv(out / "aggregate_fixed_budget_metrics.csv", index=False)
    # Per-segment coverage is label-free geometry assignment, with complete inventory accounting.
    segments = []
    for cohort, population in (("current75", current), ("historical10", generated)):
        all_distance = coverage(population, real)
        for road in roads.itertuples():
            truth_indices = real.index[(real.nearest_segment_id.eq(road.segment_id)) & real.fold.ge(0)]
            own_candidates = population.loc[population.road_id.astype(str).eq(road.segment_id)]
            segments.append(dict(cohort=cohort, segment_id=road.segment_id, corridor=road.corridor, fold=road.fold,
                                 length_m=float(road.geometry.length), candidates_on_segment=len(own_candidates),
                                 official_records=len(truth_indices), covered_by_any_cohort_candidate=int((all_distance[truth_indices] <= 20).sum()),
                                 outside_original_boundary=int((~real.loc[truth_indices].inside_original_study_boundary).sum())))
    pd.DataFrame(segments).to_csv(out / "per_segment_candidate_coverage.csv", index=False)
    after = protected_hashes()
    assert before == after, "Protected input/production/historical artifact changed."
    save_json(out / "input_integrity.json", dict(unchanged=True, files=len(before), before=before, after=after))
    for relative in ("scripts/extract_inventory_for_validation.py", "scripts/validate_inventory_heldout_roads.py"):
        shutil.copyfile(ROOT / relative, out / (Path(relative).name + ".txt"))
    save_json(out / "COMPLETE.json", dict(retrospective=True, independent_test=False, fits=6,
                                          hyperparameter_search=False, production_unchanged=True,
                                          evaluated_inventory=int(real.fold.ge(0).sum()),
                                          outside_corridor_inventory=int(real.fold.lt(0).sum()),
                                          inventory=inventory_summary, warning=WARNING))
    print(pd.DataFrame(summaries).query("k_per_group in [100,250]").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
