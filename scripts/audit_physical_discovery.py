"""Secondary evidence and explicit model-development exposure; no exclusions."""

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
from scripts.diagnose_summer_discovery import xy, nearest


def main():
    records = pd.read_csv(OUT / "all_959_record_diagnosis.csv")
    truth = (
        gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
        .set_index("record_id")
        .loc[records.record_id]
        .reset_index()
    )
    raw = gpd.read_parquet(OUT / "baseline_raw.parquet")
    neighbors = cKDTree(xy(raw)).query_ball_point(xy(truth), 20)
    evidence = []
    for record, use in zip(records.itertuples(), neighbors):
        group = raw.iloc[use]
        evidence.append(
            dict(
                record_id=record.record_id,
                raw_candidates_within20=len(group),
                any_profile_available=bool(group.road_profile_available.fillna(False).any()),
                any_dip_ge015=bool(group.route_elevation_dip_60m.ge(0.15).any()),
                any_positive_flow_concavity=bool(
                    (group.upstream_area_m2.gt(0) & group.negative_tpi_9x9_m.gt(0)).any()
                ),
                any_1ha_flow=bool(group.upstream_area_m2.ge(10000).any()),
                any_admitted=bool(group.evidence_admitted.fillna(False).any()),
                any_ml_local_peak=bool(group.ml_local_peak.fillna(False).any()),
                any_physical_crossing=bool(group.physical_crossing.fillna(False).any()),
            )
        )
    records = records.merge(pd.DataFrame(evidence), on="record_id", validate="one_to_one")
    measured = gpd.read_parquet(
        ROOT
        / "reports/experiments/hydrology_driven_sites_20261015/physical_measurements_v2.parquet"
    ).set_index("candidate_id")
    for family, path, target in [
        ("RF", ROOT / "models/flow_c_frozen_v1.joblib.training_records.csv", "experiment_target"),
        (
            "HGB",
            ROOT / "reports/experiments/site_challenger_20261015/final_training_rows.csv",
            "target",
        ),
    ]:
        manifest = pd.read_csv(path)
        positives = measured.loc[manifest.loc[manifest[target].eq(1), "candidate_id"]].reset_index()
        distances, positions = nearest(positives, truth)
        records[f"{family}_positive_training_nearest_m"] = distances
        records[f"{family}_positive_training_nearest_id"] = positives.iloc[
            positions
        ].candidate_id.to_numpy()
        records[f"{family}_positive_training_proximity25m"] = distances <= 25
    records["prior_inventory_evaluation_exposure"] = True
    records["candidate_design_exposure_this_diagnostic"] = True
    records["thresholds_fixed_before_challenger_matching"] = True
    records["exact_training_record_identity_verified"] = False
    records["untouched_independent_test"] = False
    records["gps_or_road_mismatch_is_hypothesis_not_correction"] = True
    records.to_csv(OUT / "all_959_record_diagnosis.csv", index=False)
    misses = records.loc[records.query_success & ~records.full_list_matched20]
    secondary = misses.groupby("failure_substage")[
        [
            "any_profile_available",
            "any_dip_ge015",
            "any_positive_flow_concavity",
            "any_1ha_flow",
            "any_ml_local_peak",
            "possible_unmapped_small_drainage",
            "possible_coordinate_road_mismatch",
        ]
    ].sum()
    secondary.to_csv(OUT / "failure_secondary_evidence_counts.csv")
    summary = dict(
        RF_within25m_training_positive=int(records.RF_positive_training_proximity25m.sum()),
        HGB_within25m_training_positive=int(records.HGB_positive_training_proximity25m.sum()),
        prior_evaluation_records=959,
        untouched_test_records=0,
        features_selected_or_tuned_on_this_inventory=False,
        new_threshold_search=False,
        records_deleted=0,
        coordinate_corrections=0,
    )
    (OUT / "provenance_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
