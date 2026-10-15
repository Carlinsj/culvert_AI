"""Read-only independent checks of completed whole-road validation artifacts."""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.candidates import generate_road_route_candidates  # noqa: E402
from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file  # noqa: E402
from scripts.validate_inventory_heldout_roads import (  # noqa: E402
    MODEL, SOURCE, coverage, fold_for, match_one_to_one, rank_sites, save_json,
)
from culvert_ai.inference import RouteZConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "reports/experiments/real_inventory_heldout_roads_20261015")
    args = parser.parse_args()
    run = args.run
    out = run / "post_run_audit"
    out.mkdir(exist_ok=False)
    inventory = pd.read_csv(run / "inventory_split_manifest.csv")
    real = gpd.GeoDataFrame(inventory, geometry=gpd.points_from_xy(inventory.source_x, inventory.source_y), crs=26918).to_crs(32618)
    roads = gpd.read_parquet(run / "recovered_road_segments.parquet")
    frame = gpd.read_file(SOURCE, columns=["candidate_id", "source", "geometry"]).set_index("candidate_id")
    bundle = load_frozen_flow_c_bundle(MODEL)
    current = gpd.read_parquet(run / "current75_features.parquet")
    old = gpd.read_parquet(run / "historical10_features.parquet")
    regenerated = generate_road_route_candidates(roads, include_all_roads=True, route_id_column="corridor", interval_m=75, lateral_offsets_m=(0.,))
    assert current.candidate_id.tolist() == ("eval75_" + regenerated.candidate_id).tolist()
    assert current.geometry.equals(regenerated.geometry)
    assert current.corridor.tolist() == regenerated.matched_route.tolist()
    assert current.fold.tolist() == current.corridor.map(fold_for).tolist()
    checks, conflicts, group_totals, distributions = [], [], [], []
    for column in bundle["feature_columns"]:
        for cohort, table in (("current75", current), ("historical10", old)):
            values = pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            distributions.append(dict(feature=column, cohort=cohort, candidates=len(table),
                                      missing=int(values.isna().sum()), unique=int(values.nunique()),
                                      nonzero=int(values.fillna(0).ne(0).sum()), median=float(values.median()),
                                      q05=float(values.quantile(.05)), q95=float(values.quantile(.95))))
    for fold in range(3):
        truth = real.loc[real.fold.eq(fold)].sort_values("record_key").reset_index(drop=True)
        heldout = set(roads.loc[roads.fold.eq(fold), "corridor"])
        held_lines = roads.loc[roads.fold.eq(fold)].geometry.union_all()
        for model in ("RF", "HGB"):
            manifest = pd.read_csv(run / f"training_manifest_{model}_fold{fold}.csv")
            training = frame.loc[manifest.candidate_id].copy().reset_index(drop=True)
            distance = shapely.distance(training.geometry.array, held_lines)
            assert (distance > 100).all() and not set(manifest.corridor) & heldout
            label_distance = coverage(truth, training)
            assert label_distance.min() > 50  # >=25m training +20m evaluation, with margin.
            train_real = real.loc[real.fold.ge(0) & real.fold.ne(fold)].reset_index(drop=True)
            td, ti = cKDTree(np.c_[train_real.geometry.x, train_real.geometry.y]).query(np.c_[training.geometry.x, training.geometry.y])
            negative = manifest.target.eq(0).to_numpy() & (td <= 20)
            if negative.any():
                rows = manifest.loc[negative].copy()
                rows["fold"] = fold
                rows["model"] = model
                rows["nearest_training_inventory_record"] = train_real.iloc[ti[negative]].record_key.to_numpy()
                rows["distance_m"] = td[negative]
                conflicts.append(rows)
            checks.append(dict(fold=fold, model=model, training_rows=len(manifest),
                               min_heldout_road_distance_m=float(distance.min()),
                               min_heldout_inventory_distance_m=float(label_distance.min()),
                               overlapping_train_test_corridors=0,
                               training_label_sources=manifest.label_source.value_counts().to_dict(),
                               train_negative_inventory_conflicts_at_20m=int(negative.sum())))
            estimator = joblib.load(run / f"evaluation_copy_{model.lower()}_fold{fold}.joblib")["model"]
            assert estimator.n_features_in_ == 48
            if model == "RF":
                assert estimator.get_params() == bundle["model"].get_params()
            else:
                assert (estimator.max_iter, estimator.max_leaf_nodes, estimator.min_samples_leaf,
                        estimator.learning_rate, estimator.l2_regularization) == (200, 7, 40, .04, 10.)
        ref = json.loads((run / f"normalization_reference_fold{fold}.json").read_text())
        reference = RouteZConfig(global_center=ref["center"], global_scale=ref["scale"])
        for cohort, population in (("current75", current), ("historical10", old)):
            predictions = gpd.read_parquet(run / f"heldout_predictions_{cohort}_fold{fold}.parquet")
            expected = population.loc[population.fold.eq(fold)].sort_values("candidate_id").reset_index(drop=True)
            assert predictions.candidate_id.tolist() == expected.candidate_id.tolist()
            assert predictions.geometry.equals(expected.geometry)
            ids = predictions.candidate_id.tolist()
            assert not set(ids) & set(pd.read_csv(run / f"training_manifest_RF_fold{fold}.csv").candidate_id)
            for model, policy in (("RF", "raw_rf"), ("RF", "dynamic_route_z"), ("HGB", "raw_rf")):
                sites = gpd.read_parquet(run / f"sites_{cohort}_{model}_{policy}_fold{fold}.parquet")
                rebuilt = rank_sites(predictions, predictions[model+"_score"].to_numpy(), policy, reference)
                assert sites.candidate_id.tolist() == rebuilt.candidate_id.tolist()
                np.testing.assert_array_equal(sites.model_score, sites.candidate_id.map(predictions.set_index("candidate_id")[model+"_score"]))
                xy = np.c_[truth.geometry.x, truth.geometry.y]
                assignment = match_one_to_one(sites, xy)
                raw_distance = coverage(predictions, truth)
                group_totals.append(dict(cohort=cohort, fold=fold, model=model, policy=policy,
                                         official=len(truth), raw_covered=int((raw_distance <= 20).sum()),
                                         full_matches=len(assignment), representatives=len(sites),
                                         matched_250=len(match_one_to_one(sites.head(250), xy))))
    pd.DataFrame(checks).to_json(out / "spatial_and_training_audit.json", orient="records", indent=2)
    if conflicts:
        pd.concat(conflicts, ignore_index=True).to_csv(out / "training_label_conflicts.csv", index=False)
    pd.DataFrame(group_totals).to_csv(out / "independently_recomputed_group_totals.csv", index=False)
    pd.DataFrame(distributions).to_csv(out / "feature_distribution_audit.csv", index=False)
    before = json.loads((run / "protected_before.json").read_text())
    changed = [p for p, digest in before.items() if sha256_file(ROOT / p) != digest]
    assert not changed
    metrics = pd.DataFrame(group_totals)
    summary = metrics.groupby(["cohort", "model", "policy"])[["official", "raw_covered", "full_matches", "representatives", "matched_250"]].sum()
    summary.to_csv(out / "coverage_summary.csv")
    save_json(out / "audit_result.json", dict(passed=True, protected_files=len(before),
                                               current_candidates_reproduced=True, frozen_specs_reproduced=True,
                                               policy_score_vectors_identical=True, consolidation_reproduced=True,
                                               group_inventory_disjoint=True, min_train_holdout_inventory_distance_over_50m=True,
                                               inspection_features_used=False,
                                               classification="retrospective, not untouched final testing"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
