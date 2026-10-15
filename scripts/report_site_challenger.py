"""Finish the executed challenger experiment with an evidence-backed report."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from culvert_ai.flow_c_bundle import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/site_challenger_20261015"


def main():
    completed = json.loads((OUT / "COMPLETE.json").read_text())
    declaration = json.loads((OUT / "preregistered_config.json").read_text())
    selected = completed["final_selection"]
    results = pd.read_csv(OUT / "outer_oof_metrics.csv")
    baselines = pd.read_csv(OUT / "baseline_reproduction.csv")
    folds = pd.read_csv(OUT / "selected_procedure_route_metrics.csv")
    selections = pd.read_csv(OUT / "outer_fold_selections.csv")
    bundle = joblib.load(OUT / "challenger.joblib")
    final_name = f"{selected['config']}__{selected['policy']}"
    fixed = results.loc[results.system.eq(final_name)].iloc[0]
    nested = results.loc[results.system.eq("NESTED_SELECTION_PROCEDURE")].iloc[0]
    policy_stability = []
    for config in declaration["configurations"]:
        names = [f"{config['id']}__{policy}" for policy in declaration["policies"]]
        queues = {name: set(pd.read_csv(OUT / f"top250_{name}.csv").candidate_id) for name in names}
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                policy_stability.append(dict(configuration=config["id"], left=left, right=right,
                                             common_sites=len(queues[left] & queues[right]),
                                             jaccard=len(queues[left] & queues[right])/len(queues[left] | queues[right])))
    pd.DataFrame(policy_stability).to_csv(OUT / "within_model_policy_stability.csv", index=False)
    road_audit = json.loads((OUT / "road_group_audit.json").read_text())
    integrity = json.loads((OUT / "protected_integrity.json").read_text())
    assert integrity["unchanged"]
    assert integrity["before"] == integrity["after"]
    checks = json.loads((OUT / "leakage_checks.json").read_text())
    for check in checks:
        assert not set(check["training_routes"]) & set(check["validation_routes"])
        assert check["min_train_validation_distance_m"] > 40.
    oof = pd.read_parquet(OUT / "outer_oof_predictions.parquet")
    counts = oof.groupby("config").candidate_id.agg(["size", "nunique"])
    assert counts["size"].eq(counts["nunique"]).all()
    assert counts["size"].nunique() == 1
    label_manifest = pd.read_csv(OUT / "final_training_rows.csv")
    assert label_manifest.loc[label_manifest.target.eq(0), "label_source"].eq("field_verified_negative").all()
    negative_locations = pd.read_parquet(OUT / "verified_negative_features.parquet")
    inventory = pd.read_csv(ROOT / "reports/experiments/nysdot_ulster_real_inventory_benchmark_v1_20261006/source_extract/small_culverts_r8_source.csv")
    inventory = inventory.loc[inventory.County.astype(str).str.strip().str.upper().eq("ULSTER")]
    negative_inventory_distance = cKDTree(inventory[["x", "y"]]).query(negative_locations[["x_m", "y_m"]])[0]
    overlapping_negatives = int((negative_inventory_distance <= 20.).sum())

    # Package the actual source, fixed parameters, environment, and metadata.
    for name in ("train_site_challenger.py", "predict_site_challenger.py", "report_site_challenger.py"):
        shutil.copyfile(ROOT / "scripts" / name, OUT / name)
    import scipy
    import sklearn
    metadata = dict(
        configuration=bundle["configuration"], model_class=f"{bundle['model'].__class__.__module__}.{bundle['model'].__class__.__name__}",
        estimator_parameters=bundle["model"].get_params(deep=False),
        feature_columns=bundle["feature_columns"], normalization=bundle["normalization"],
        normalization_fallback=bundle["normalization_fallback"], radius_m=20.,
        consolidation="priority descending, raw model score descending, candidate_id ascending; global greedy 20m suppression",
        preprocessing=bundle["preprocessing"], feature_medians=bundle["medians"].to_dict(),
        pairwise_feature_scales=bundle.get("scales", pd.Series(dtype=float)).to_dict(),
        additional_input_provenance={str(path.relative_to(ROOT)): sha256_file(path) for path in (
            ROOT / road_audit["hydrology_feature_artifact"],
            ROOT / "reports/experiments/nysdot_ulster_dem_hydrology_v8_20261005/exact_feature_lists.json",
            ROOT / "reports/experiments/nysdot_ulster_dem_hydrology_v8_20261005/training_rows.csv",
            ROOT / "reports/experiments/nysdot_ulster_ranking_v18_20261006/experiment_config.json",
        )},
        training_rows=bundle["training_rows"], positives=int(label_manifest.target.sum()),
        verified_negatives=int(label_manifest.target.eq(0).sum()), unknown_negative_labels=0,
        verified_negative_locations_within_20m_of_inventory=overlapping_negatives,
        route_counts=label_manifest.groupby(["route", "target"]).size().rename("rows").reset_index().to_dict("records"),
        environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        frozen_artifact_is_independently_validated=False, deployed=False,
        field_message="Recommendations require field verification.",
    )
    (OUT / "challenger_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    historical_specifications = []
    historical_dir = ROOT / "reports/experiments/nysdot_ulster_ranking_v18_20261006/model_artifacts"
    for artifact in sorted(historical_dir.glob("pointwise_gb__authoritative_inventory__heldout_*.joblib")):
        old = joblib.load(artifact)
        historical_specifications.append(dict(
            artifact=str(artifact.relative_to(ROOT)), sha256=sha256_file(artifact),
            model_class=f"{old['model'].__class__.__module__}.{old['model'].__class__.__name__}",
            parameters=old["model"].get_params(deep=False),
            ordered_feature_columns=old["columns"], training_medians=old["median"].to_dict(),
        ))
    assert len(historical_specifications) == 11
    (OUT / "historical_model_specifications.json").write_text(
        json.dumps(historical_specifications, indent=2, default=str) + "\n")

    # End-to-end frozen inference on real existing feature rows and entirely
    # arbitrary renamed road IDs. No model fit occurs in this verification.
    spec = importlib.util.spec_from_file_location("frozen_challenger_predict", ROOT / "scripts/predict_site_challenger.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    dev = pd.read_parquet(OUT / "development_features.parquet").head(1000).copy()
    example = gpd.GeoDataFrame(dev, geometry=gpd.points_from_xy(dev.x_m, dev.y_m), crs="EPSG:26918").to_crs("EPSG:32618")
    example["route_id"] = "arbitrary-unseen-" + example.route.astype(str)
    first = runtime.predict(example, OUT / "challenger.joblib")
    second = runtime.predict(example, OUT / "challenger.joblib")
    assert first.candidate_id.tolist() == second.candidate_id.tolist()
    np.testing.assert_array_equal(first.raw_score, second.raw_score)
    xy = first[["x_m", "y_m"]].to_numpy(float)
    if len(xy) > 1:
        assert (cKDTree(xy).query(xy, k=2)[0][:, 1] > 20. - 1e-7).all()
    rejected = False
    try:
        runtime.predict(example.drop(columns=bundle["feature_columns"][0]), OUT / "challenger.joblib")
    except ValueError:
        rejected = True
    assert rejected
    (OUT / "inference_verification.json").write_text(json.dumps(dict(
        serialized_scores_bitwise_equal=True, repeated_ranks_identical=True,
        arbitrary_road_identifiers=True, physical_sites_separated_beyond_20m=True,
        missing_schema_field_rejected=True, input_rows=len(example), output_sites=len(first),
    ), indent=2) + "\n")
    all_metrics = pd.concat([baselines, results], ignore_index=True)
    lines = [
        "# Executed physical-site ranking challenger — 2026-10-15", "",
        f"Selected and frozen experimental challenger: **{selected['config']} + {selected['policy']}**. The all-development inner-validation objective selected this fixed specification; no outer outcome was used to choose it.", "",
        f"The nested selection procedure retrieved **{int(nested.confirmed_at_100)}/100** and **{int(nested.confirmed_at_250)}/250** distinct authoritative sites. The selected fixed specification's development OOF diagnostics are **{int(fixed.confirmed_at_100)}/100** and **{int(fixed.confirmed_at_250)}/250**. These are inventory-matched retrieval fractions; they are not verified field precision or an untouched final test.", "",
        f"The selected fixed specification {'exceeds' if fixed.confirmed_at_250 > 41 else 'does not exceed'} the historical 41/250 development benchmark. The nested selection procedure {'exceeds' if nested.confirmed_at_250 > 41 else 'does not exceed'} it. Fitting the final artifact on development labels does not independently validate the artifact.", "",
        "**No deployment improvement is established.** The frozen model is a reproducible research challenger, not a recommended replacement. The strongest post-hoc new-model matrix entry is 32/250 (HGB_SHALLOW + robust_route), below both historical POINTWISE 41/250 and saved RF raw ordering 33/250. That post-hoc entry was not selected as the final model because final configuration selection was restricted to inner-validation evidence.", "",
        "## Scope and compatibility", "",
        "Nine configurations actually trained: four HistGradientBoosting classifiers, two regularized GradientBoosting classifiers, two pairwise regularized logistic rankers, and one 800-tree RF control. External LightGBM/XGBoost/CatBoost packages were unavailable locally. No datasets were acquired. All new models use the exact same ordered 48 FLOW-C feature names. All controlled comparisons use the same candidate IDs, training labels, folds, spatial purges, and deterministic 20m consolidation.", "",
        "The historical POINTWISE model used **76 features**, including saved RF scores and local score summaries. It is a separate historical reference, not a 48-feature model improvement claim. Its exact per-outer-route compact/wide settings and ordered feature lists are recorded in historical_compatibility.json. PW-COMPACT used learning rate 0.05, 100 iterations, 15 leaves, L2=1; PW-WIDE used 0.05, 160 iterations, 31 leaves, L2=1. Both used authoritative positives, sampled presumed inventory negatives, and field negatives, with inner selection. The new experiment does not copy those presumed-negative labels.", "",
        "The legacy feature preparation could override road groups with authoritative inventory route labels. The interrupted first pass was stopped before selection. The completed experiment derives road groups from matched_route or the nearest existing road sample within 100m and recomputes only the existing four road-density normalization fields using the app's label-free transform. No new predictors or candidate coordinates were introduced.", "",
        f"Exactly {road_audit['changed_from_historical']} candidate road groups differ from the legacy assignment. Original archived baselines retain their archived definitions; the newly fitted RF_CONTROL is the valid controlled comparison under corrected road grouping and label policy.", "",
        "## Training labels and leakage controls", "",
        f"The final fit uses {metadata['positives']} candidate rows matched within 20m to authoritative positives, with one unit of pre-class-balancing weight per real site, plus {metadata['verified_negatives']} field-verified negatives. Unmatched candidates receive no negative target. Three unassigned field negatives are excluded. The field negatives remain concentrated on road 32 (97 of 126); route-balanced weighting is included in the declared search, but cannot replace representative absence evidence.", "",
        f"A descriptive post-fit data check finds {overlapping_negatives} supplied field-negative locations within the benchmark's 20m radius of an authoritative inventory record. Point-level inspection labels and radius-based inventory matches can conflict under this evaluation convention. Supplied labels were retained, with no post-outcome relabeling or configuration change; this ambiguity needs adjudication before independent field validation.", "",
        "Outer validation leaves one complete road out. Within each outer training set, two deterministic grouped-road splits generate inner OOF predictions. Every configuration and normalization choice is made using only those inner predictions and training-route inventory. Any training row within 40m of validation/outer candidate or negative geometry is purged, preventing a shared physical site at road junctions from entering both sides. Missing medians and pairwise scaling use training rows only. No target distance, inventory count, old density prediction, route ID, coordinate, or saved model score is a predictor.", "",
        f"All {len(checks)} recorded split checks pass disjoint-route and >40m minimum spatial separation assertions. Each configuration has {int(counts['size'].iloc[0])} unique outer-OOF candidates from the original 51,417-row pool. Label-free road assignment yields 12 outer groups, including one 32A candidate historically assigned to 32. The 32A group has no authoritative inventory record, so its route-specific site recall is undefined. The 11 original populated inventory roads remain present, and the archived baselines retain their original groups.", "",
        "The final fixed configuration is chosen in a separate two-fold all-development grouped evaluation. Primary objective is 0.5×(matched fraction@100 + matched fraction@250), with declared tie-breaks. The nested-selection OOF result estimates the selection procedure; the fixed-spec OOF matrix is diagnostic and must not be used to select a different winner after seeing outer outcomes.", "",
        "Outer folds sometimes select different normalization families. Raw scores and standardized scores therefore coexist in the pooled nested-selection queue, which creates an additional cross-fold score-scale limitation. Per-road nested retrieval remains interpretable, while the pooled procedure queue is a diagnostic rather than validation of one globally consistent score scale. The fixed-specification OOF rows each use one model family and one normalization across all roads; final specification selection still uses reused development evidence. No extra post-hoc normalization was introduced to improve the pooled result.", "",
        "## All executed configurations", "",
        "| ID | Family | Settings | Training weighting |", "|---|---|---|---|",
    ]
    for config in declaration["configurations"]:
        settings = ", ".join(f"{k}={v}" for k, v in config.items() if k not in ("id", "family", "weighting"))
        lines.append(f"| {config['id']} | {config['family']} | {settings} | {config['weighting']} |")
    lines += ["", "Site-balanced weights start at one per positive site and per verified-negative record; classes have equal total loss weight. Route-balanced configurations additionally equalize each class's route totals. RF_CONTROL retains the deployed estimator hyperparameters, with parallelism=2 and the common controlled labels/weights. Pairwise models compare positive-site representatives only against field-verified negatives; no unknown row enters a pair. Some pairwise fits reached the declared 2000-iteration optimizer limit, so their optimum is not guaranteed. No post-outcome retuning was performed.", "",
              "## Archived baseline reproduction and outer-OOF matrix", "",
              "| System/policy | Matched@25 | @50 | @100 | @250 | Matched fraction@250 | Site recall@250 | Roads@250 | Largest share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in all_metrics.itertuples():
        lines.append(f"| {row.system} | {row.confirmed_at_25} | {row.confirmed_at_50} | {row.confirmed_at_100} | {row.confirmed_at_250} | {row.inventory_matched_fraction_at_250:.2%} | {row.site_recall_at_250:.2%} | {row.routes_at_250} | {row.largest_route_share:.1%} |")
    lines += ["", "The archived RF dynamic ROUTE-Z result reproduces 5/250; historical POINTWISE reproduces 41/250. The older 650/714 figure describes post-ranking/post-consolidation FLOW-C representative recall from 15,650 sites, not classifier accuracy.", "",
              "## Attribution to model and policy", "",
              "Read vertically within a policy to compare new models under identical folds, labels, candidate geometry, and consolidation. Read across policies for one model to measure normalization effects. All comparisons are available; no policy is selected using the final outer matrix. Dynamic ROUTE-Z uses current mean/std rules with training-score fallback; robust normalization uses per-road median and IQR/1.349, with a training-only fallback for small/constant groups. Normalization is label-free and uses the available unlabeled road score distribution.", ""]
    controlled_rf = results.loc[results.system.eq(f"RF_CONTROL__{selected['policy']}")].iloc[0]
    selected_raw = results.loc[results.system.eq(f"{selected['config']}__raw")].iloc[0]
    lines += [f"At the final selected policy, the frozen specification's OOF diagnostic is {int(fixed.confirmed_at_250)}/250 versus {int(controlled_rf.confirmed_at_250)}/250 for newly trained RF_CONTROL: difference {int(fixed.confirmed_at_250-controlled_rf.confirmed_at_250):+d} matched sites under a common policy. The selected model with raw ordering retrieves {int(selected_raw.confirmed_at_250)}/250; its chosen policy changes this by {int(fixed.confirmed_at_250-selected_raw.confirmed_at_250):+d}. These are descriptive development comparisons, not an independent causal or significance claim.", "",
              "## Selected procedure per held-out road", "",
              "| Road | Inner-selected model | Policy | Matched@100 | Matched@250 | Site recall@250 |",
              "|---|---|---|---:|---:|---:|"]
    for row in folds.itertuples():
        selection = selections.loc[selections.scope.eq(f"outer_{row.route}")].iloc[0]
        recall = f"{row.site_recall_at_250:.2%}" if pd.notna(row.site_recall_at_250) else "N/A"
        lines.append(f"| {row.route} | {selection['config']} | {selection['policy']} | {row.confirmed_at_100} | {row.confirmed_at_250} | {recall} |")
    lines += ["", "Per-road @K metrics use the available recommendation count for short roads, and are not summed to obtain the global Top250. Exact counts, outer selections, Top250 identities, route shares, and candidate Jaccard stability across model/policy queues are saved as CSV files. within_model_policy_stability.csv directly compares each model's three policies using Top250 overlap and Jaccard. Variation across policies and routes is a stability diagnostic; this run does not establish repeatability in a new county.", "",
              "## Frozen artifact and reproducibility", "",
              "Model: `challenger.joblib`. Ordered schema: `feature_schema.json`. Full estimator parameters, medians, normalization, provenance and environment: `challenger_metadata.json`. Training manifests, OOF scores, input hashes, code snapshots, and artifact hashes accompany the model. The production RF bundle and the application remain byte-identical to their pre-run hashes.", "",
              "Train/reproduce: `scripts/python.sh scripts/train_site_challenger.py`. Completed outputs are protected against overwrite; per-road checkpoints support resuming interruption. Inference: `scripts/python.sh scripts/predict_site_challenger.py --features /path/features.gpkg --model reports/experiments/site_challenger_20261015/challenger.joblib --output /path/experimental_sites.gpkg --route-column route_id`. Input must contain all 48 numeric features, candidate_id, label-free road groups, and EPSG:32618 geometry. Consolidation uses the preserved EPSG:26918 benchmark metric frame at exactly 20m, then returns input geometry.", "",
              "Saved/reloaded scores are bitwise identical, arbitrary road-ID renaming preserves normalization, repeated inference preserves site order, missing fields are rejected, and no two emitted representatives are within 20m. These checks establish deterministic software behavior; the sample feature rows were existing development inputs and do not constitute new-road performance validation.", "",
              "## Deployment gate and remaining limitations", "",
              "This is an experimental challenger and has not been deployed. The Ulster roads have been reused throughout prior research; nested grouping limits leakage within this run but does not make them an untouched final test. Historical POINTWISE's score-derived RF predictors were not fully nested within its outer folds, so 41/250 remains a historical development reference with provenance limitations. Cross-model comparisons to that historical reference also change feature count, label policy, route assignment, and preprocessing; the controlled nine-configuration matrix isolates model/policy effects more closely.", "",
              "Before deployment, lock the exact artifact and preprocessing, evaluate it once on independently collected roads/region using the same input pipeline, and obtain field-reviewed outcomes that include representative negative inspections. Check the full road-to-feature-to-score pipeline on those unseen roads. Inventory nonmatches remain unknown. No 70% field precision claim is supported by this development experiment.", "",
              "**Recommendations require field verification.**"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = json.loads((OUT / "manifest.json").read_text())
    manifest["files"] = {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(final_model=str(OUT / "challenger.joblib"), selected=selected,
                          fixed_specification_oof=fixed.to_dict(), nested_procedure=nested.to_dict(),
                          leakage_checks=len(checks), frozen_inference_verified=True), indent=2))


if __name__ == "__main__":
    main()
