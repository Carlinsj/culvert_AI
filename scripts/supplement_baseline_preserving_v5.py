"""Seal the exact repaired-RF lineage replay and inference compatibility checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from culvert_ai.flow_c_bundle import sha256_file
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.baseline_reranker import load_bundle, rank
from staging.hydrology_sites import apply_feature_contract

REPORT = ROOT / "reports/experiments/baseline_preserving_v5_20261019"
MODEL_DIR = ROOT / "models/experiments/baseline_preserving_v5_20261019"
HYDROLOGY = ROOT / "reports/experiments/hydrology_driven_sites_20261015"
PHYSICAL = ROOT / "reports/experiments/physical_discovery_20261016"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"


def main() -> None:
    output = REPORT / "SUPPLEMENT_COMPLETE.json"
    if output.exists():
        raise RuntimeError("Supplement already completed")
    source_bundle = joblib.load(HYDROLOGY / "hydrology-sites-v2_RF_experimental.joblib")
    measured = gpd.read_parquet(HYDROLOGY / "physical_measurements_v2.parquet")
    context = measured.loc[measured.proposal_kind.eq("internal_road_probe")]
    manifest_path = ROOT / "models/flow_c_frozen_v1.joblib.training_records.csv"
    manifest = pd.read_csv(manifest_path)
    transformed = apply_feature_contract(measured, source_bundle["contract"], normalization_context=context)
    training = transformed.set_index("candidate_id", drop=False).loc[manifest.candidate_id].reset_index(drop=True)
    columns = source_bundle["feature_columns"]
    matrix = training[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(source_bundle["medians"])
    model = clone(source_bundle["model"])
    model.fit(matrix, manifest.experiment_target.astype(int))
    replay_bundle = dict(
        version="repaired-rf-original-lineage-v5-experimental",
        experimental=True,
        active_default=False,
        model=model,
        medians=source_bundle["medians"],
        feature_columns=columns,
        contract=source_bundle["contract"],
        training_manifest=str(manifest_path.relative_to(ROOT)),
        warning="Contains ordinary unlabeled candidates used historically as negatives and development-exposed official matches; lineage replay only.",
    )
    artifact = MODEL_DIR / "repaired_rf_original_lineage.joblib"
    joblib.dump(replay_bundle, artifact)

    candidates = gpd.read_parquet(PHYSICAL / "variant_BASELINE.parquet")
    candidate_matrix = candidates[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(source_bundle["medians"])
    replay_score = model.predict_proba(candidate_matrix)[:, 1]
    max_score_difference = float(np.max(np.abs(replay_score - candidates.score_RF.to_numpy(float))))
    queue = candidates.copy()
    queue["score_repaired_original_lineage"] = replay_score
    queue = queue.sort_values(["score_repaired_original_lineage", "candidate_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    queue["global_rank"] = np.arange(1, len(queue) + 1)
    queue["model_score"] = queue.score_repaired_original_lineage
    queue.to_parquet(REPORT / "queue_repaired_rf_original_lineage.parquet", index=False)

    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    audit = pd.read_csv(PHYSICAL / "all_959_record_diagnosis.csv").set_index("record_id")
    truth = truth.loc[truth.record_id.map(audit.query_success).astype(bool)].sort_values("source_excel_row").reset_index(drop=True)
    matched_1000 = len(match_one_to_one(queue.head(1000), np.c_[truth.geometry.x, truth.geometry.y], 20))

    saved = gpd.read_parquet(REPORT / "ranked_sites.parquet")
    replay_v5 = rank(candidates, load_bundle())
    inference = {
        "candidate_ids_equal": replay_v5.candidate_id.tolist() == saved.candidate_id.tolist(),
        "max_score_difference": float(np.max(np.abs(replay_v5.score_v5 - saved.score_v5))),
        "top1000_ids_equal": replay_v5.head(1000).candidate_id.tolist() == saved.head(1000).candidate_id.tolist(),
    }
    if (
        max_score_difference != 0
        or matched_1000 != 123
        or not inference["candidate_ids_equal"]
        or inference["max_score_difference"] > 1e-15
        or not inference["top1000_ids_equal"]
    ):
        raise RuntimeError("Lineage or v5 inference replay failed")
    label_counts = manifest.experiment_label_source.value_counts().to_dict()
    result = {
        "historic_training_rows": len(manifest),
        "historic_target_counts": {str(k): int(v) for k, v in manifest.experiment_target.value_counts().items()},
        "historic_label_sources": {str(k): int(v) for k, v in label_counts.items()},
        "ordinary_unlabeled_candidates_historically_used_as_negative": int(label_counts.get("ordinary_unlabeled_candidate", 0)),
        "official_matches": int(label_counts.get("nysdot_official_match_le_25m", 0)),
        "repaired_rf_max_score_difference": max_score_difference,
        "repaired_rf_matches_at_1000": matched_1000,
        "v5_inference_replay": inference,
        "artifact_sha256": sha256_file(artifact),
        "frozen_rf_sha256": sha256_file(ROOT / "models/flow_c_frozen_v1.joblib"),
    }
    (REPORT / "original_rf_training_lineage.json").write_text(json.dumps(result, indent=2) + "\n")
    output.write_text(json.dumps({"complete": True, **result}, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
