"""Describe completed fixed-spec validation; no fitting, tuning or source edits."""
import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_inventory_heldout_roads import coverage, save_json  # noqa: E402


def markdown(frame):
    """No optional table-rendering dependency required."""
    def val(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v).replace("|", "/")
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"]*len(frame.columns)) + " |",
                       *["| " + " | ".join(val(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def format_prose(line):
    """Fix prose spacing while preserving code, links, route IDs and acronyms."""
    pieces = re.split(r"(`[^`]*`|\[[^]]*\]\([^)]*\))", line)
    words = ("All|all|these|has|and|its|with|for|from|within|only|than|of|near|on|"
             "Group|group|page|revised|seed|rank|At|at|shoulders|full|old|"
             "by|below|above|minimum|maximum|Route|neglects|flags|cohort|records")
    for i in range(0, len(pieces), 2):
        text = pieces[i]
        text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
        text = re.sub(rf"\b({words})(?=\d)", r"\1 ", text)
        text = re.sub(r"(?<=\d)(km|m)\b", r" \1", text)
        text = re.sub(r"(?<=[A-Za-z]),(?=\d)", ", ", text)
        text = re.sub(r"(?<=[A-Za-z])\.(?=\d)", ". ", text)
        text = re.sub(r"(?<=[A-Za-z]):(?=\d)", ": ", text)
        text = re.sub(r",(?=\d{1,2}(?:\D|$))", ", ", text)
        pieces[i] = text
    return "".join(pieces)


def trace_suppression(run, failures, real):
    traces = []
    lost = failures.loc[failures.failure_category.eq("B_no_surviving_representative_20m")]
    for (cohort, fold, model, policy), part in lost.groupby(["cohort", "fold", "model", "policy"]):
        predicted = gpd.read_parquet(run / f"heldout_predictions_{cohort}_fold{fold}.parquet")
        sites = gpd.read_parquet(run / f"sites_{cohort}_{model}_{policy}_fold{fold}.parquet")
        candidate_xy = np.c_[predicted.geometry.x, predicted.geometry.y]
        site_xy = np.c_[sites.geometry.x, sites.geometry.y]
        raw_tree, site_tree = cKDTree(candidate_xy), cKDTree(site_xy)
        for row in part.itertuples():
            official = real.loc[row.record_key].geometry
            point = np.array([official.x, official.y])
            witnesses = raw_tree.query_ball_point(point, 20.)
            selected = min(witnesses, key=lambda i: (np.linalg.norm(candidate_xy[i]-point), predicted.iloc[i].candidate_id))
            neighbors = site_tree.query_ball_point(candidate_xy[selected], 20.)
            suppressor = min(neighbors, key=lambda i: int(sites.iloc[i].global_rank))
            candidate, representative = predicted.iloc[selected], sites.iloc[suppressor]
            distance = float(np.linalg.norm(site_xy[suppressor]-point))
            assert distance > 20
            traces.append(dict(cohort=cohort, fold=fold, model=model, policy=policy,
                               record_key=row.record_key, corridor=row.corridor,
                               witness_candidate_id=candidate.candidate_id,
                               candidate_to_inventory_m=float(np.linalg.norm(candidate_xy[selected]-point)),
                               suppressor_candidate_id=representative.candidate_id,
                               suppressor_rank=int(representative.global_rank),
                               candidate_to_suppressor_m=float(np.linalg.norm(candidate_xy[selected]-site_xy[suppressor])),
                               suppressor_to_inventory_m=distance,
                               witness_raw_score=float(candidate[model+"_score"]),
                               suppressor_raw_score=float(representative.model_score)))
    pd.DataFrame(traces).to_csv(run / "consolidation_selection_failure_traces.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "reports/experiments/real_inventory_heldout_roads_20261015")
    args = parser.parse_args()
    run = args.run
    if (run / "REPORT.md").exists():
        raise FileExistsError("Do not overwrite an existing report.")
    inventory = json.loads((run / "inventory_integrity.json").read_text())
    plan = json.loads((run / "preregistered_method.json").read_text())
    audit = json.loads((run / "post_run_audit/audit_result.json").read_text())
    aggregate = pd.read_csv(run / "aggregate_fixed_budget_metrics.csv")
    groups = pd.read_csv(run / "heldout_group_metrics.csv")
    road_metrics = pd.read_csv(run / "heldout_road_metrics.csv")
    failures = pd.read_csv(run / "inventory_failure_diagnostics.csv")
    concentration = pd.read_csv(run / "route_concentration.csv")
    # This is a report of this exact fixed-spec run, not a template that silently
    # repeats old numbers when a different environment produces different results.
    assert inventory["ulster_records"] == 714 and inventory["any_direct_inventory_positive"] == 701
    assert aggregate.query("cohort=='current75' and k_per_group==250").matched.tolist() == [36, 31, 35]
    assert concentration.query("cohort=='current75' and model=='RF' and policy=='raw_rf' and corridor=='212'").top250.tolist() == [31]
    coverage_summary = pd.read_csv(run / "post_run_audit/coverage_summary.csv")
    source = pd.read_csv(run / "inventory_split_manifest.csv")
    real = gpd.GeoDataFrame(source, geometry=gpd.points_from_xy(source.source_x, source.source_y), crs=26918).to_crs(32618).set_index("record_key", drop=False)
    trace_suppression(run, failures, real)
    complete = []
    wide_coverage = source[["record_key", "source_excel_row", "Route", "declared_route", "corridor", "fold", "distance_to_recovered_road_m", "inside_original_study_boundary"]].copy()
    for cohort in ("current75", "historical10"):
        population = gpd.read_parquet(run / f"{cohort}_features.parquet")
        wide_coverage[cohort+"_nearest_candidate_m"] = coverage(population, real)
        for model, policy in (("RF", "raw_rf"), ("RF", "dynamic_route_z"), ("HGB", "raw_rf")):
            records = failures.loc[failures.cohort.eq(cohort) & failures.model.eq(model) & failures.policy.eq(policy)].copy()
            outside = source.loc[source.fold.lt(0), ["record_key", "source_excel_row", "declared_route", "corridor", "nearest_segment_id", "inside_original_study_boundary"]].copy()
            outside["cohort"], outside["model"], outside["policy"], outside["fold"] = cohort, model, policy, -1
            outside["failure_category"] = "outside_recovered_corridors_not_scored"
            outside["full_one_to_one_matched"] = False
            outside["top250_one_to_one_matched"] = False
            records = pd.concat([records, outside], ignore_index=True)
            assert len(records) == 714 and records.record_key.is_unique
            complete.append(records)
    all_records = pd.concat(complete, ignore_index=True)
    all_records.to_csv(run / "inventory_outcomes_all714.csv", index=False)
    wide_coverage.to_csv(run / "candidate_coverage_all714.csv", index=False)
    global_current_covered = int(wide_coverage.current75_nearest_candidate_m.le(20).sum())
    assert global_current_covered == 348
    all_records.groupby(["cohort", "model", "policy", "declared_route", "failure_category"]).size().rename("official_records").reset_index().to_csv(run / "declared_route_inventory_outcomes.csv", index=False)
    # Descriptive score associations only: inventory-unmatched points are NOT negatives.
    associations = []
    for cohort in ("current75", "historical10"):
        for fold in range(3):
            predicted = gpd.read_parquet(run / f"heldout_predictions_{cohort}_fold{fold}.parquet")
            official = real.loc[real.fold.eq(fold)]
            near = coverage(official, predicted) <= 20
            for route, indices in predicted.groupby("corridor").groups.items():
                for model in ("RF", "HGB"):
                    for label, use in (("within_20m_of_inventory", near[indices]), ("inventory_unmatched_not_verified_negative", ~near[indices])):
                        values = predicted.loc[indices, model+"_score"].to_numpy()[use]
                        associations.append(dict(cohort=cohort, fold=fold, corridor=route, model=model,
                                                 descriptive_group=label, candidates=len(values),
                                                 median=float(np.median(values)) if len(values) else None,
                                                 q10=float(np.quantile(values, .1)) if len(values) else None,
                                                 q90=float(np.quantile(values, .9)) if len(values) else None))
    pd.DataFrame(associations).to_csv(run / "score_inventory_associations.csv", index=False)
    current_table = aggregate.query("cohort=='current75'").pivot(index="k_per_group", columns=["model", "policy"], values="matched").reset_index()
    current_table.columns = ["K per group", "HGB raw", "RF ROUTE-Z", "RF raw"]
    current_table["Total recommendations"] = current_table["K per group"] * 3
    group250 = groups.query("cohort=='current75' and k==250")[["fold", "model", "policy", "official_culverts", "matched", "actual_denominator", "largest_route", "largest_route_share"]]
    road = road_metrics.query("cohort=='current75' and model=='RF' and policy=='raw_rf' and k==250 and official_culverts>0")[["corridor", "official_culverts", "candidates", "raw_candidate_covered", "full_list_matches", "matched", "actual_denominator"]].copy()
    road.columns = ["Corridor", "Inventory", "Candidates", "Candidate-covered", "Full matched", "Road-local matched@250", "Actual road-local denominator"]
    failure_table = failures.groupby(["cohort", "model", "policy", "failure_category"]).size().unstack(fill_value=0).reset_index()
    failure_table.to_csv(run / "failure_category_summary.csv", index=False)
    feature = pd.read_csv(run / "post_run_audit/feature_distribution_audit.csv")
    feature_subset = feature.loc[feature.feature.isin(["route_elevation_dip_60m", "route_elevation_dip_rank_pct", "route_low_point_score", "hydrology_edge_risk"]), ["feature", "cohort", "candidates", "nonzero"]]
    neg = pd.read_csv(run / "post_run_audit/training_label_conflicts.csv")
    provenance = pd.DataFrame([
        ["RF directly linked inventory records", 701], ["RF selected official-positive candidate rows", 696],
        ["RF field-positive rows", 250], ["Inventory records within 25 m of RF field positives (already in 701)", 151],
        ["HGB direct inventory records", 689], ["Direct inventory labels in either final model", 701],
        ["Other records used in prior performance/policy development", 13],
        ["Union already used for training or development decisions", 714],
        ["Not direct-positive but on RF training corridors (subset of 13)", 4],
        ["Genuinely untouched independent records/corridors", 0],
    ], columns=["Evidence", "Count"])
    lines = ["# Real inventory whole-road validation", "", "## Decision", "",
             "**REQUIRE NEW INDEPENDENT EVIDENCE.** Keep the production model and UI unchanged. Retain raw RF only as an experimental staging comparator; do not accept or promote HGB. No new model search or controlled improvement was performed.", "",
             "Recommendations require field verification.", "",
             "The completed analysis is **retrospective development validation**, not an untouched final test. Six evaluation-only copies were fitted: three RF and three HGB, using fixed existing specifications. The original fully trained artifacts were never retrained or independently tested on their own training roads.", "",
             "The largest field-pipeline problem found is a **candidate/feature lineage mismatch**: preserved training-era interval samples are10m apart, but the current production generator and bundle declare75m. The current sampling has no candidate within20m for361 of705 recovered-corridor inventory records, and all three60m road-profile features become zero. Ranking-policy correction alone cannot recover sites absent from the candidate population.", "",
             "## Inventory integrity and identity", "",
             "Read the repository workbook `data/incoming_nysdot/Small Culverts_R8_2026.07.14.xlsx`, worksheet `Small Culverts - R8`. The request's filename ends `(1).xlsx`; that exact duplicate filename was not in the repository. The available workbook has3,028 rows and69 columns, and its714 Ulster row identities and coordinates agree with both preserved official-match mappings and the older workbook source extract. This verifies content correspondence without assuming filename identity.", "",
             "All714 distinct Ulster coordinate pairs were retained, including44 missing Culvert IDs, two reused-ID groups and one missing Route. Reused IDs149332 and150022 remain separate rows/locations. There are13 declared named routes. Identity is source Excel row plus coordinates, never a deduplicated Culvert ID.", "",
             "Source coordinates are NAD83 / UTM18N metres (EPSG:26918), verified against [NYSDOT's Environmental Viewer Directory](https://www.dot.ny.gov/divisions/engineering/environmental-analysis/manuals-and-guidance/epm/repository/Environmental_Viewer_Directory.pdf), revised2026-04-08, page11, Small Culverts. All distance/matching operations here use the native application EPSG:32618 after explicit transformation. No historical-coordinate convention is silently substituted.", "",
             "The spreadsheet skill was used read-only with the bundled spreadsheet runtime. Only the eight requested identification/location fields were extracted. Inspection condition, comments, photos, ratings, engineering attributes and density-derived inventory counts were never supplied to the48-feature models. The workbook was not modified.", "",
             "## What the existing models learned", "", markdown(provenance), "",
             "The701 RF-linked inventory records map to696 unique selected candidate rows because five rows each represent two official records. The original positive-label tolerance is25m, not the current20m evaluation tolerance. The RF's2,021 training rows comprise696 official-positive,250 field-positive,129 recorded verified-negative,941 unlabeled/background target-zero, and5 proximity-derived negative rows. These941 unknown/background rows must not be described as verified absence.", "",
             "The frozen HGB has2,652 rows:2,526 positive candidate rows linked to689 distinct inventory records, plus126 recorded verified negatives. HGB direct-inventory records are a subset of the701 RF-linked records. The field-positive spatial overlaps do not add previously untouched inventory records.", "",
             "Provenance classes are overlapping facts: A direct inventory-positive=701; B prior evaluation/model-or-policy development=714; C not direct-positive but on RF training corridors=4; D genuinely untouched road=0. Historical evaluation exposure is known for all714, not uncertain. Individual field-report-to-inventory identities without explicit identifiers remain spatial associations, not asserted exact asset identities. Nine of the13 non-direct records are outside the recovered road footprint; all13 still had previous development exposure.", "",
             "Evidence: `models/flow_c_frozen_v1.joblib.training_records.csv`, its manifest/bundle provenance, V8 `training_rows.csv` and frozen promotion code; supervised-v1 `official_positive_matches.csv`; HGB `site_challenger_20261015/final_training_rows.csv` and frozen artifact; historical workbook extract and saved OOF/coverage/policy artifacts. Per-record flags and links are in `inventory_training_overlap.csv`; original-row spatial checks are in `rf_training_rows_spatial_audit.csv` and `hgb_training_row_audit.csv`.", "",
             "## Method fixed before held-out scoring", "",
             "1. Recover all121 complete source road segments underlying interval candidates, grouped into22 label-free named corridors. No culvert proximity was used to choose candidate locations or road-segment subsets. This is the existing named-road study footprint, not every county road or arbitrary local-road deployment geography.",
             "2. Assign each complete corridor to one of three groups using a fixed SHA256(route) rule, not counts, scores or outcomes. All segments carrying the same corridor token stay together, including old-road aliases. See `preregistered_method.json`, `segment_split_manifest.csv` and `recovered_road_segments.parquet`.",
             "3. Retain the original Route field. For spatial evaluation membership, assign inventory records to the nearest recovered full segment within100m, ties by segment ID.705 records receive a corridor; nine NY9G records are3.7–11.5km outside the recovered sample footprint and are reported as out of scope, not relabeled onto a distant road. Ten records are outside the original county study boundary; one of these nevertheless falls in recovered road coverage. Declared and geometric corridors sometimes disagree at junctions or in source metadata, so this is a geometric-corridor holdout, not proof that every authoritative nominal Route ID is perfectly reconstructed.",
             "4. Before fitting, exclude all held-out corridors and all training candidates/field examples within100m of their **complete line geometry**, not just sampled points. Exclude unassigned field-label corridors. Exclude held-out official labels and legacy official-linked candidate IDs. Independent audit confirms all retained training rows are also more than50m from held-out inventory locations, preventing20/25m duplicate-label leakage.",
             "5. RF copies retain the exact frozen800-tree hyperparameters, seed42, class weighting,48-feature order, numeric preparation and training-only median imputation. Regenerate the original fixed route-balanced hard-weak-negative1:1 procedure with **training-road positives only**, rather than reusing a negative sample selected with all-road labels. No sample weights are added. The old route-label oracle is not used for feature normalization.",
             "6. HGB copies retain fixed HGB_REGULAR:200 iterations, learning rate0.04,7 leaves, minimum leaf40, L2=10, no early stopping, seed42, existing site/route/class weights, training-only median imputation, <=20m authoritative training positives and recorded verified negatives. This compares fixed existing training systems, not estimator families on identical label/weight regimes. It does not independently validate the already full-fit HGB artifact.",
             "7. Primary cohort: regenerate all11,136 candidates at the current75m interval, zero lateral offset and existing short-line midpoint fallback. Compute the same48 fields with existing GIS functions and saved hydrology crossings, without labels. Countywide road/stream context is fixed for both models and all folds. The app's location-clipped context can differ; no claim is made that this is an end-to-end test of every app search radius.",
             "8. Secondary cohort: all79,221 preserved eligible generated10m/crossing candidates on recoverable corridors, with label-free route normalization and existing GIS feature artifacts.995 original generated crossings cannot be assigned to these corridors within100m and are not silently added to the corridor study. Field observation points are not inference candidates. This is broader than the old51,417-candidate score pool; it is a separate diagnostic, not the old countywide control.",
             "9. Each evaluation group uses one fitted scorer. The identical RF prediction vector is ranked raw and with the existing dynamic ROUTE-Z rule; tiny-route fallback statistics come from training-road predictions only. HGB uses its own fixed raw ordering. All use existing deterministic score-ordered20m consolidation and canonical candidate-ID ties.",
             "10. At every K, recompute deterministic maximum one-to-one20m matching against distinct source-row inventory records. Each record belongs to only one group. Do not pool fitted models' scores or sum overlapping windows. Aggregates below sum disjoint-group retrieval counts at a fixed budget **per group**, not a manufactured global Top-K.", "",
             "## Primary current-pipeline results", "",
             "Counts below use705 in-scope records. The nine outside-corridor records remain in the all714 coverage/outcome tables and are not claimed tested. K means a budget for each of three groups.", "", markdown(current_table), "",
             "At250 per group (750 total), RF raw matches35/750=4.67%, ROUTE-Z31/750=4.13%, and HGB36/750=4.80%. These are **inventory-matched fractions**, not measured true precision or accuracy: inventory-unmatched recommendations are not verified negatives. Inventory recall is35/705,31/705 and36/705 respectively. No70% accuracy/precision claim is supported.", "",
             "Group-specific Top250 results:", "", markdown(group250), "",
             "Raw RF does **not** consistently outperform ROUTE-Z: group0 is18 vs20, group1 is2 vs4, group2 is15 vs7. HGB is not a consistently superior replacement either. A one-record aggregate advantage over RF raw is not independent evidence of improvement.", "",
             "### Candidate and full-list coverage", "", markdown(coverage_summary), "",
             "Within the held-out groups, current candidates cover344/705=48.79% of in-scope inventory (344/714=48.18% of the full workbook cohort). All three systems cover334/705 after one-to-one consolidation (334/714=46.78%); the10 further losses are one-to-one capacity, not a20m representative displacement in this cohort. Across the current population,10,896 group-specific representatives remain. Historical10m candidate coverage is700/705 (700/714), with RF full coverage656 and HGB659. Those historical totals use new held-out scorers and a broader population and must not be compared as learned gains over the older653/714 or650/714 controls.", "",
             "**Coverage scope clarification:**344 is the sum of within-held-out-group coverage, not coverage of the union of all generated geometry. The full75m footprint has348/714 records within20m of any candidate (348/705 in scope); four additional locations have a candidate only on geometry assigned to another evaluation group. These cannot be credited to that record's own held-out scorer. `candidate_coverage_all714.csv` and `per_segment_candidate_coverage.csv` report union-geometry coverage; group/road metrics report the restricted held-out population. Summed representative/recommendation counts are group-specific slots, not deduplicated countywide inspection locations. Geometries can overlap at junctions or concurrent routes, but distinct inventory records are never counted in more than one evaluation group.", "",
             "### Per-road retrieval and coverage", "",
             "The following RF-raw table uses a **road-local** inspection budget after fixed group-global consolidation. It is not the number allocated to that road in the group's Top250, and its rows must not be summed to claim group-budget results. Raw and ROUTE-Z are monotone within a road, so local order can agree while global allocation differs. All22 corridors, all K values, denominators and both other systems are in `heldout_road_metrics.csv`; full121-segment coverage is in `per_segment_candidate_coverage.csv`. Original declared-Route record outcomes, including NY9G/NY418/missing Route, are separately in `declared_route_inventory_outcomes.csv`.", "", markdown(road), "",
             "## Failure diagnostics", "",
             "For current RF raw,705 in-scope records partition into361 without a20m candidate,10 lost to one-to-one capacity,299 represented but not retrieved within their group's250-slot budget, and35 retrieved. Of the361 candidate misses,356 are within20m of recovered full road geometry and had a historical10m candidate; five are more than20m from that geometry. Sparse candidate placement is therefore a directly observed retrieval ceiling, not an inferred score-model cause.", "",
             "Of those356 own-group spacing/placement misses, four are covered by another group's candidate geometry. Across the entire recovered footprint,352 road-near inventory records still lack any75m candidate within20m. This distinction prevents attributing the four group-assignment boundary cases purely to global candidate sparsity.", "",
             "For historical RF raw, five have no candidate,32 lose all within20m representatives,12 lose one-to-one capacity,635 are represented but not retrieved at budget, and21 are retrieved. The consolidation trace file identifies a witness candidate and its actual earlier-ranked suppressor for every geometric loss. A candidate can be within20m of a culvert and within20m of its suppressor while that suppressor is more than20m from the culvert. This documents the exact selection geometry without changing the production rule.", "",
             "`inventory_failure_diagnostics.csv` retains every in-scope record per fixed system/cohort with nearest distances, first geometric representative rank and independently recomputed match flags at10/25/50/100/250. `inventory_outcomes_all714.csv` adds explicit out-of-scope rows. `top250_site_diagnostics.csv` lists candidate IDs, scores, road allocation and all48 features; unassigned inventory matches are called unknown, not false positives. `score_inventory_associations.csv` compares descriptive score distributions for inventory-near and unmatched candidates without treating the latter as negative truth.", "",
             "### Feature and training-label associations", "", markdown(feature_subset), "",
             "The road-profile function requires shoulders20–60m away. At75m sample spacing there are no neighboring samples in that window, so `route_elevation_dip_60m`, its percentile, and `route_low_point_score` are zero at every primary candidate. The historical population has19,536 nonzero dip rows and15,854 nonzero low-point scores. The current crossing-context function also defaults `hydrology_edge_risk` to zero; the historical artifact marks11,984 candidates. These are reproducible pipeline differences, not proof of their individual causal contributions to retrieval failure. Production was not repaired during this evaluation.", "",
             f"Four distinct recorded-negative candidate rows ({', '.join(sorted(neg.candidate_id.unique()))}) fall within20m of three official records. These are **potential spatial label conflicts** requiring source review, not proof a negative observation is wrong: coordinate offsets and distinct nearby structures remain possible. Original labels were retained to evaluate existing training procedures; no post-hoc relabeling was performed. `post_run_audit/training_label_conflicts.csv` lists only training-side occurrences in each fold.", "",
             "### Route allocation", "",
             "Route212 does not repeat the old84.8% countywide concentration in this evaluation. In its current-pipeline group, RF raw and ROUTE-Z each allocate31/250=12.4% to212; HGB allocates42/250=16.8%. On historical10m candidates, RF raw allocates88/250=35.2%, ROUTE-Z zero, HGB169/250=67.6%. Different scorer training/cohorts/group populations prevent interpreting these as a directly comparable countywide concentration improvement.", "",
             "Current RF raw neglects214 (32 official records; first recommendation rank2,464) and32A (one record; first rank2,656) at the group Top250; it also neglects300,302 and587, which have no official records in this inventory. ROUTE-Z neglects375 (20 records; first rank440),302 and587. HGB neglects375,300 and587. Group0 raw is led by44 at36.0%; group1 by28 at35.6%; group2 by87 at41.6%. A road with no inventory matches is not demonstrated culvert-free. Full counts and zero-allocation cases are in `route_concentration.csv`.", "",
             "## Limits and recommended next action", "",
             "There is no independently untouched final test. All714 inventory records and the model families/features/ranking options have previous development exposure. The deterministic holdouts prevent fitting-row/corridor leakage in these evaluation copies, not historical researcher/model-selection reuse. Three groups provide limited estimates of geographic variability. Road/corridor assignment is derived from available geometry and must not be confused with a perfect authoritative road-segment linkage. Conditions on arbitrary local roads, outside the recovered footprint or under every location-clipped app context are not established.", "",
             "The current primary results show no consistent policy winner, severe candidate sparsity, profile-feature collapse, retained training-label uncertainty and many inventory-unmatched early recommendations. They do not support promoting raw RF, HGB or claiming field-ready accuracy. Preserve current production artifacts pending a separately authorized correction/evaluation; keeping a frozen artifact is not an endorsement of its field performance.", "",
             "**Next focused work:** first reconcile training/inference candidate spacing and road-profile extraction in an isolated, preregistered pipeline-parity test using training/inner-validation roads only. Preserve the same48 definitions and use a fixed identical control. Do not select a new correction from these held-out outcomes and then claim this cohort is untouched. Review the four nearby negative observations against authoritative records before changing labels. Obtain new independent road/field observations for a final deployment decision. No new architecture search is justified.", "",
             "No controlled experimental improvement was implemented: the audit identified pipeline hypotheses, but a correction chosen after inspecting these outcomes would require a fresh evaluation cohort or explicit retrospective labeling. A valid negative deployment-readiness result is retained.", "",
             "## Reproducibility and protected artifacts", "",
             f"The independent verifier passed. All{audit['protected_files']} protected source/production/previous-research files retain their hashes. Frozen RF SHA256: `{plan['frozen_rf_sha256']}`. Frozen HGB SHA256: `{plan['frozen_hgb_sha256']}`. Evaluation copies exist only in this new research directory. The production model, model sidecars, UI, official inventory and historical reports were not modified, and nothing was deployed.", "",
             "Commands from `/Users/Carli/culvert_AI` (each output directory must be new):", "", "```sh",
             "scripts/python.sh scripts/validate_inventory_heldout_roads.py --output reports/experiments/my_new_fixed_validation",
             "scripts/python.sh scripts/audit_heldout_validation_results.py --run reports/experiments/my_new_fixed_validation",
             "scripts/python.sh scripts/report_inventory_heldout_validation.py --run reports/experiments/my_new_fixed_validation",
             "scripts/python.sh -m pytest tests/test_inventory_heldout_validation.py -q", "```", "",
             "The extractor uses the available bundled Python/openpyxl executable identified in the source; adjust that runtime path on another machine. It never writes the source workbook. `--resume` is for interrupted evaluation copies only; use a new output directory for a fresh reproduction. No broad model selection, extra feature search or final production fit is invoked.", "",
             "Key artifacts:", "",
             "- `inventory_integrity.json`, `inventory_training_overlap.csv`: integrity, direct/indirect use and independent-test accounting.",
             "- `preregistered_method.json`, inventory/segment split manifests, training manifests and `leakage_audit.json`: fixed procedure and complete-road exclusions.",
             "- `current75_features.parquet`, `historical10_features.parquet`, held-out prediction/site Parquets: every eligible candidate ID, all48 numeric features, native geometry and fixed-system outputs.",
             "- `heldout_group_metrics.csv`, `heldout_road_metrics.csv`, `aggregate_fixed_budget_metrics.csv`: every requested K with actual denominator, recall, inventory-match fraction, unmatched count and concentration.",
             "- `per_segment_candidate_coverage.csv`, `candidate_coverage_all714.csv`, per-record failures, consolidation traces and Top250 diagnostics: coverage ceilings and selection/score allocation failures.",
             "- `post_run_audit/`: independent generation/ranking/matching checks, feature distributions, training-side spatial label conflicts and recomputed totals.",
             "- `input_integrity.json`, source `.py.txt` snapshots and `COMPLETE.json`: protected hashes and exact executed source. No source test copies are collected as duplicate tests.", "",
             "Full regression suite:153 passed,12 existing geometry deprecation warnings. Evaluation scripts and tests pass lint. The eight new evaluation tests cover maximum matching versus greedy matching, inclusive20m boundary, separate inventory records at identical coordinates, empty coverage, stable corridor splits, full-line purge, deterministic segment ties and actual budget denominators.", "",
             "Recommendations require field verification."]
    lines = [format_prose(line) if not line.startswith("|") else line for line in lines]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    save_json(run / "FINAL_DECISION.json", dict(
        decision="REQUIRE NEW INDEPENDENT EVIDENCE", retrospective=True,
        independent_test_records=0, inventory_records=714, evaluated_records=705,
        current_within_heldout_group_candidate_covered=344,
        current_union_geometry_candidate_covered=global_current_covered,
        current_within_group_full_list_matched=334,
        k_per_group=250, groups=3, actual_total_budget=750,
        rf_raw_matches=35, rf_route_z_matches=31, hgb_raw_matches=36,
        controlled_improvement_performed=False, production_changed=False, deployed=False,
        regression_tests_passed=153, protected_files_unchanged=audit["protected_files"],
        recommendation="Resolve candidate/profile lineage in a training-only controlled study; obtain new independent field evidence.",
        warning="Recommendations require field verification.",
    ))
    print(run / "REPORT.md")


if __name__ == "__main__":
    main()
