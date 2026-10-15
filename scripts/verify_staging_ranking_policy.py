"""Reproduce fixed-policy saved-score controls. No model is fitted or tuned."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file  # noqa: E402
from culvert_ai.inference import RouteZConfig, rank_flow_c_candidates  # noqa: E402
from staging.ranking_policy import (  # noqa: E402
    FROZEN_MODEL_SHA256, WARNING, RankingPolicy, export_sites,
    predict_from_features, rank_candidates,
)

EXP = ROOT / "reports/experiments"
PREVIOUS = EXP / "site_coverage_negative_weights_20261015"
SOURCE = ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg"
MODEL = ROOT / "models/flow_c_frozen_v1.joblib"
POOL = EXP / "nysdot_ulster_consensus_precision_20261013/candidate_consensus_features.parquet"
OOF = EXP / "nysdot_ulster_dem_hydrology_v8_20261005/hydro_c_plus_flow_c__whole_route__oof_predictions.parquet"
KS = (25, 50, 100, 250, 500, 1000)


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protected_hashes():
    old = json.loads((PREVIOUS / "input_integrity.json").read_text())["after"]
    paths = {ROOT / p for p in old}
    paths.update([POOL, OOF, SOURCE, MODEL])
    for folder in ("src", "web", "server", "api", "models"):
        paths.update(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in str(p))
    paths.update(p for p in PREVIOUS.rglob("*")
                 if p.is_file() and "__pycache__" not in str(p))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/staging/ranking_policy_correction_20261015")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    before = protected_hashes()
    assert sha256_file(MODEL) == FROZEN_MODEL_SHA256
    bundle = load_frozen_flow_c_bundle(MODEL)
    assert bundle["model"].n_estimators == 800 and len(bundle["feature_columns"]) == 48
    assert sha256_file(SOURCE) == bundle["training_data_provenance"]["feature_source_sha256"]
    assert sha256_file(OOF) == bundle["route_z_global"]["source_sha256"]
    config = RouteZConfig(global_center=bundle["route_z_global"]["center"],
                          global_scale=bundle["route_z_global"]["scale"])
    pool = pd.read_parquet(POOL, columns=["candidate_id", "route"])
    saved = pd.read_parquet(OOF).set_index("candidate_id").probability
    duplicate = pd.read_parquet(EXP / "nysdot_ulster_ranking_v18_20261006/out_of_fold_predictions/flow_c.parquet")
    assert saved.sort_index().equals(duplicate.set_index("candidate_id").outer_oof_score.sort_index())
    assert len(pool) == 51417 and set(pool.candidate_id) == set(saved.index)
    native = gpd.read_file(SOURCE, columns=["candidate_id", "geometry"])
    native = native.set_index("candidate_id", drop=False).loc[pool.candidate_id].reset_index(drop=True)
    assert native.crs.to_epsg() == 32618
    native["route_id"] = pool.route.to_numpy()
    native["flow_c_score"] = native.candidate_id.map(saved)
    # Import only the existing inventory reader and fixed one-to-one matcher.
    # Never call the research runner's preparation, training or experiment main.
    evaluator_path = EXP / "nysdot_ulster_ranking_v18_20261006/run_experiment.py"
    spec = importlib.util.spec_from_file_location("staging_fixed_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    real = evaluator.load_real()
    geometry = gpd.GeoSeries(gpd.points_from_xy(real.x, real.y), crs=26918).to_crs(32618)
    real["x"], real["y"] = geometry.x, geometry.y
    truth_xy = real[["x", "y"]].to_numpy(float)
    metrics, route_rows = [], []
    results = {}
    for policy in RankingPolicy:
        sites = rank_candidates(native, policy=policy, route_z_config=config)
        shuffled = rank_candidates(native.sample(frac=1, random_state=42), policy=policy,
                                   route_z_config=config)
        assert sites.candidate_id.tolist() == shuffled.candidate_id.tolist()
        assert set(sites.candidate_id).issubset(set(native.candidate_id))
        # Exact representative/rank parity with the preserved matched-control run.
        artifact = "RF_RAW" if policy is RankingPolicy.CHALLENGER else "RF_DYNAMIC_ROUTE_Z"
        reference = pd.read_parquet(PREVIOUS / f"representatives_{artifact}.parquet")
        assert sites.candidate_id.tolist() == reference.candidate_id.tolist()
        if policy is RankingPolicy.CURRENT:
            pd.testing.assert_frame_equal(
                sites[rank_flow_c_candidates(native, route_z_config=config).columns],
                rank_flow_c_candidates(native, route_z_config=config),
            )
        export_sites(sites, out / policy.value)
        check_csv = pd.read_csv(out / policy.value / "sites.csv")
        check_geo = gpd.read_file(out / policy.value / "sites.geojson")
        assert sites.candidate_id.tolist() == check_csv.candidate_id.tolist() == check_geo.candidate_id.tolist()
        assert check_csv.field_verification_notice.eq(WARNING).all()
        eval_sites = sites.rename(columns={"physical_site_id": "site_id"}).copy()
        eval_sites["x_m"], eval_sites["y_m"] = sites.geometry.x, sites.geometry.y
        full, _ = evaluator.match_sites(eval_sites, truth_xy)
        assert len(full) == 653 and len(real) == 714
        for k in (*KS, len(sites)):
            matches, _ = evaluator.match_sites(eval_sites, truth_xy, top_k=k)
            counts = sites.head(k).route_id.value_counts()
            metrics.append(dict(policy=policy.value, k=k, recommendations=min(k, len(sites)),
                                matched=len(matches), inventory_records=714,
                                precision=len(matches)/min(k, len(sites)),
                                largest_route=str(counts.index[0]),
                                largest_route_count=int(counts.iloc[0]), routes=len(counts)))
        top = {r["k"]: r["matched"] for r in metrics if r["policy"] == policy.value}
        assert (top[100], top[250]) == ((15, 33) if policy is RankingPolicy.CHALLENGER else (0, 5))
        for route in sorted(native.route_id.fillna("<unassigned>").astype(str).unique()):
            group = sites.loc[sites.route_id.eq(route)]
            row = dict(policy=policy.value, route=route,
                       input_candidates=int(native.route_id.fillna("<unassigned>").eq(route).sum()),
                       full_representatives=len(group),
                       first_global_rank=int(group.global_rank.min()) if len(group) else None,
                       recommendations_at_100=int(group.global_rank.le(100).sum()),
                       recommendations_at_250=int(group.global_rank.le(250).sum()),
                       share_at_250=float(group.global_rank.le(250).sum()/250))
            row["neglected_at_250"] = row["recommendations_at_250"] == 0
            route_rows.append(row)
        results[policy.value] = sites
    # Actual frozen-estimator inference on renamed arbitrary IDs: a SOFTWARE
    # compatibility check, not evidence of accuracy on independent unseen roads.
    fixture = pd.read_parquet(EXP / "site_challenger_20261015/development_features.parquet").head(48)
    geometry = native.set_index("candidate_id").loc[fixture.candidate_id].geometry.reset_index(drop=True)
    features = gpd.GeoDataFrame(fixture[["candidate_id", *bundle["feature_columns"]]].reset_index(drop=True),
                                geometry=geometry, crs=32618)
    features["route_id"] = [f"unseen-road-{i % 3}-arbitrary" for i in range(len(features))]
    features["candidate_id"] = [f"unseen-candidate-{i:04d}" for i in range(len(features))]
    features.to_parquet(out / "software_check_features.parquet", index=False)
    software = {}
    for policy in RankingPolicy:
        sites = predict_from_features(features, MODEL, policy=policy)
        assert set(sites.route_id) == set(features.route_id)
        assert set(sites.candidate_id).issubset(set(features.candidate_id))
        software[policy.value] = dict(candidates=len(features), representatives=len(sites),
                                      road_ids=sorted(set(sites.route_id)), passed=True)
    pd.testing.assert_frame_equal(results[RankingPolicy.CURRENT.value],
                                  rank_candidates(native, route_z_config=config))
    pd.DataFrame(metrics).to_csv(out / "top_k_comparison.csv", index=False)
    routes = pd.DataFrame(route_rows)
    routes.to_csv(out / "per_route_concentration.csv", index=False)
    real.to_csv(out / "native_inventory_evaluation_only.csv", index=False)
    # Preserve exact source changes alongside outputs for reproducibility.
    for relative in ("staging/ranking_policy.py", "staging/ranking-policy.json",
                     "scripts/stage_ranking_policy.py", "scripts/verify_staging_ranking_policy.py",
                     "tests/test_staging_ranking_policy.py"):
        # Text snapshots must not be collected as duplicate pytest modules.
        target = out / "code" / (relative + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    after = protected_hashes()
    assert before == after, "Protected production/research inputs changed."
    json_file(out / "input_integrity.json", dict(unchanged=True, before=before, after=after))
    json_file(out / "validation.json", dict(
        environment="staging", model_sha256=FROZEN_MODEL_SHA256,
        trees=800, features=48, candidates=51417, site_radius_m=20,
        score_source=str(OOF.relative_to(ROOT)), fit_calls=0,
        candidate_population_identical=True, native_crs="EPSG:32618",
        exact_saved_representatives_reproduced=True, ordering_shuffle_test=True,
        exports_round_trip=True, rollback_exact=True, unseen_id_software_checks=software,
        protected_files=len(before), production_unchanged=True, deployed=False,
        independent_held_out_roads_available=False,
        limitations=[
            "Saved historical RF OOF predictions, not a new evaluation of the deployed full-fit estimator.",
            "Ulster roads have been reused extensively; legacy features/folds include route-label overrides.",
            "Renaming road IDs verifies software compatibility, not geographic generalization.",
            "Software checks reuse a saved 48-field development matrix; they do not rerun GIS feature generation.",
            "No untouched independent held-out-road score/feature/truth set was identified in available artifacts.",
            "Scores are not calibrated probabilities; no 70% precision tier is validated.",
            "Global raw ordering can neglect roads; per-road counts are reported without quotas or retuning.",
        ], recommendation="Controlled staging comparison only; no production promotion.",
        field_verification_notice=WARNING,
    ))
    print(pd.DataFrame(metrics).query("k in [100, 250]").to_string(index=False))
    print(f"Validated; protected {len(before)} existing files. Outputs: {out}")


if __name__ == "__main__":
    main()
