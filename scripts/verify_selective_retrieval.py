"""Read-only scientific replay and integrity checks; no fitting or source writes."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_selective_retrieval import OLD, SUMMER, HYDRO, save
from scripts.evaluate_hydrology_sites import protected
from scripts.verify_physical_discovery import independent_count
from staging import selective_sites as s
from staging import site_ranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out != OLD
    inputs = json.loads((out / "input_hashes.json").read_text())
    assert all(sha256_file(Path(path)) == digest for path, digest in inputs.items())
    protected_count = len(protected())
    before_models = out / "model_hashes_before_grouping_replay.json"
    if before_models.exists():
        model_hashes = json.loads(before_models.read_text())
        assert len(model_hashes) == 4
        assert all(sha256_file(Path(path)) == digest for path, digest in model_hashes.items())
    old_sources = json.loads((OLD / "code_manifest.json").read_text())
    # Documentation can evolve; baseline inference/UI/launcher source cannot.
    runtime = {name: digest for name, digest in old_sources.items() if not name.endswith(".md")}
    assert all(sha256_file(ROOT / name) == digest for name, digest in runtime.items())
    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    assignment = pd.read_csv(out / "inventory_fold_assignment.csv").set_index("record_id")
    truth = (
        truth.loc[truth.record_id.isin(assignment.index)]
        .sort_values("source_excel_row", kind="stable")
        .reset_index(drop=True)
    )
    assert len(truth) == 847 and assignment.index.is_unique
    table = pd.read_csv(out / "comparison.csv")
    verified = 0
    for name, rows in table.groupby("system"):
        queue = gpd.read_parquet(out / f"queue_{name}.parquet")
        assert queue.candidate_id.is_unique and queue.crs.to_epsg() == 32618
        for row in rows.itertuples():
            part = queue if row.budget == "all" else queue.head(int(row.budget))
            assert len(part) == row.stops
            assert independent_count(part, truth, row.tolerance_m) == row.matched
            verified += 1
    baseline = gpd.read_parquet(OLD / "variant_BASELINE.parquet")
    hybrid = gpd.read_parquet(OLD / "variant_HYBRID.parquet")
    for rule in s.RULES:
        fresh = s.anchored_groups(s.population(baseline, hybrid, rule))
        stored = gpd.read_parquet(out / f"provenance_anchored_{rule}.parquet")
        assert set(fresh.candidate_id) == set(stored.candidate_id)
        expected = s.order(fresh)
        saved = gpd.read_parquet(out / f"queue_SELECTIVE-GROUPED-{rule}-RF.parquet")
        assert expected.candidate_id.tolist() == saved.candidate_id.tolist()
        shuffled = s.anchored_groups(
            s.population(
                baseline.sample(frac=1, random_state=1), hybrid.sample(frac=1, random_state=2), rule
            )
        )
        assert s.order(shuffled).candidate_id.tolist() == expected.candidate_id.tolist()
        anchors = fresh.set_index("candidate_id").loc[baseline.candidate_id]
        np.testing.assert_array_equal(anchors.score_RF, baseline.score_RF)
        assert all(a.equals(b) for a, b in zip(anchors.geometry, baseline.geometry))
        # Missing drainage identity cannot cause a non-coincident merge.
        assert len(fresh) == len(s.population(baseline, hybrid, rule))
    roads = gpd.read_parquet(out / "road_fold_assignment.parquet")
    assert roads.groupby("id").research_fold.nunique().max() == 1
    recovered = roads.loc[roads.heldout_corridor.ne("")]
    assert recovered.groupby("heldout_corridor").research_fold.nunique().max() == 1
    site_manifest = pd.read_csv(out / "supervised_site_manifest.csv")
    measurements = gpd.read_parquet(HYDRO / "physical_measurements_v2.parquet")
    sites = measurements.set_index("candidate_id").loc[site_manifest.candidate_id].reset_index()
    for column in (
        "site_key",
        "target",
        "route",
        "label_source",
        "excluded_conflicting_label_within20m",
    ):
        sites[column] = site_manifest[column].to_numpy()
    sites = sites.loc[~sites.excluded_conflicting_label_within20m].reset_index(drop=True)
    labels = pd.read_csv(
        ROOT / "reports/experiments/site_challenger_20261015/final_training_rows.csv"
    )
    siblings = measurements.loc[measurements.proposal_kind.eq("training_measurement_only")].merge(
        labels, on="candidate_id", validate="one_to_one"
    )
    for fold in range(3):
        keep, distance, _ = site_ranker.training_exclusion(
            sites, siblings, roads.loc[roads.research_fold.eq(fold)]
        )
        train = pd.read_csv(out / f"training_fold{fold}.csv")
        assert set(train.candidate_id) == set(sites.loc[keep].candidate_id)
        assert distance[keep].min() > 1000
        assert train.site_key.is_unique
        assert set(train.target) == {0, 1}
        assert set(train.loc[train.target.eq(0), "label_source"]) == {"field_verified_negative"}
        bundle = joblib.load(out / f"experimental_site_ranker_fold{fold}.joblib")
        assert set(bundle["features"]).issubset(site_ranker.FEATURES)
    choice = json.loads((out / "all_development_choice.json").read_text())
    frame = s.anchored_groups(s.population(baseline, hybrid, choice["rule"]))
    lookup = frame.set_index("candidate_id", drop=False)
    queue = gpd.read_parquet(out / "queue_SITE-MODEL-CROSSFIT.parquet")
    measured = lookup.loc[queue.candidate_id].reset_index(drop=True)
    replay = np.empty(len(queue))
    for fold in range(3):
        use = queue.research_fold.eq(fold)
        bundle = joblib.load(out / f"experimental_site_ranker_fold{fold}.joblib")
        replay[use] = site_ranker.score(bundle, measured.loc[use])
    np.testing.assert_allclose(replay, queue.score_site_oof, rtol=0, atol=1e-12)
    # All 959 observations remain original: 112 coverage failures are not
    # silently turned into supported zero-detection observations.
    original = pd.read_csv(OLD / "all_959_record_diagnosis.csv")
    assert len(original) == 959 and (~original.query_success).sum() == 112
    assert (ROOT / "handoff.md").name not in set(old_sources)
    result = dict(
        independent_metric_rows_replayed=verified,
        queue_systems=table.system.nunique(),
        deterministic_admission_grouping_rules=3,
        whole_road_spatial_model_folds_verified=3,
        model_predictions_replayed=len(queue),
        guarded_inputs_unchanged=len(inputs),
        protected_artifacts_unchanged=protected_count,
        baseline_runtime_sources_unchanged=len(runtime),
        frozen_model_fits=0,
        verification_fits=0,
        production_changed=False,
        deployed=False,
        independent_validation=False,
        default_changed=False,
    )
    save(out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
