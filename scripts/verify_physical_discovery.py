"""Independent truth-to-site augmenting-path check of the saved research counts."""

# ruff: noqa: E402
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

from scripts.capture_summer_discovery import DEFAULT as OUT, PRIOR
from scripts.diagnose_summer_discovery import xy
from scripts.evaluate_hydrology_sites import protected


def independent_count(sites, truth, radius):
    if not len(sites) or not len(truth):
        return 0
    edges = cKDTree(xy(sites)).query_ball_point(xy(truth), radius)
    assigned = {}

    def visit(record, seen):
        for site in sorted(edges[record]):
            if site in seen:
                continue
            seen.add(site)
            if site not in assigned or visit(assigned[site], seen):
                assigned[site] = record
                return True
        return False

    for record in range(len(truth)):
        visit(record, set())
    return len(assigned)


def main():
    assert (OUT / "COMPUTATIONS_COMPLETE.json").exists()
    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
    audit = pd.read_csv(OUT / "all_959_record_diagnosis.csv").set_index("record_id")
    truth = truth.loc[truth.record_id.map(audit.query_success)].reset_index(drop=True)
    table = pd.read_csv(OUT / "candidate_and_site_coverage.csv")
    verified = 0
    for variant, rows in table.groupby("variant"):
        frame = gpd.read_parquet(OUT / f"variant_{variant}.parquet")
        assert frame.candidate_id.is_unique
        assert frame.crs.to_epsg() == 32618
        for r in rows.itertuples():
            assert independent_count(frame, truth, r.tolerance_m) == r.matched
            verified += 1
    # Original source rows and baseline partition, including unsupported rows.
    assert len(audit) == 959 and audit.index.is_unique
    assert int(audit.query_success.sum()) == 847
    assert int((audit.query_success & ~audit.full_list_matched20).sum()) == 500
    assert int((~audit.query_success).sum()) == 112
    assert audit.primary_failure_category.notna().all()
    assert (
        not audit.loc[audit.query_success & ~audit.full_list_matched20, "primary_failure_category"]
        .eq("Candidate retained but ranked too low")
        .any()
    )
    baseline = gpd.read_parquet(OUT / "variant_BASELINE.parquet")
    assert independent_count(baseline, truth, 20) == 347
    assert independent_count(baseline.loc[baseline.global_rank.le(25)], truth, 20) == 155
    from staging.discovery_sites import ordered

    ranked_rows = 0
    budgets = pd.read_csv(OUT / "fixed_budget_retrieval.csv")
    for (variant, family, policy, scope), rows in budgets.groupby(
        ["variant", "family", "policy", "budget_scope"]
    ):
        frame = gpd.read_parquet(OUT / f"variant_{variant}.parquet")
        queue = (
            ordered(frame.assign(model_score=frame[f"score_{family}"]), policy=policy)
            if scope == "global_fixed_sites"
            else None
        )
        for row in rows.itertuples():
            selected = (
                queue.iloc[: row.requested_budget]
                if queue is not None
                else frame.loc[frame[f"query_rank_{family}"].le(row.requested_budget)]
            )
            assert len(selected) == row.recommendations
            assert independent_count(selected, truth, 20) == row.matched
            ranked_rows += 1
    hashes = json.loads((OUT / "baseline_input_hashes.json").read_text()) | json.loads(
        (OUT / "experiment_input_hashes.json").read_text()
    )
    from culvert_ai.flow_c_bundle import sha256_file

    assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    guard = protected()
    result = dict(
        independent_stage_metric_rows=verified,
        independent_ranked_metric_rows=ranked_rows,
        baseline_full_matches=347,
        baseline_top25_matches=155,
        records=959,
        supported_misses=500,
        unsupported=112,
        protected_artifacts_unchanged=len(guard),
        additional_inputs_unchanged=len(hashes),
        fits=0,
        production_changed=False,
    )
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
