"""Build and evaluate the final compact physical-site research queue.

Inventory coordinates are used only after the deterministic queue is sealed.
They never affect admission, grouping, representative selection, or ranking.
"""

# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd

from culvert_ai.flow_c_bundle import sha256_file
from scripts.validate_anti_saturation import (
    deterministic_uniform,
    edges_for,
    prefix_state,
)
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.final_physical_sites import (
    DEM_RESOLUTION_M,
    EXACT_DUPLICATE_M,
    VERSION,
    compact_cross_generator_sites,
    eligible_physical_union,
)
from staging.model_sprint import build_candidate_sites


REPORT = ROOT / "reports/experiments/final_physical_optimization_20261019"
MODEL = ROOT / "models/experiments/compact_physical_sites_v6_20261019"
PHYSICAL = ROOT / "reports/experiments/physical_discovery_20261016"
SELECTIVE = ROOT / "reports/experiments/selective_admission_20261016"
SPRINT = ROOT / "reports/experiments/model_improvement_sprint_20261017"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
V5_REPORT = ROOT / "reports/experiments/baseline_preserving_v5_20261019"
V5_MODEL = ROOT / "models/experiments/baseline_preserving_v5_20261019"
BUDGETS = (250, 500, 1000, 2000, 5000)
UI_BUDGETS = (100, 250, 500, 1000, 5000, 10000, 13065)
FRONTIER_STEP = 5000
TARGET_RECALL = 0.70
RANDOM_TRIALS = 1000
RANDOM_SEED = 20261019


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def minimum_budget(queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, target: int) -> int | None:
    """Return the first prefix whose exact bipartite matching reaches target."""
    edges = edges_for(queue, truth, 20.0)
    right: dict[int, int] = {}

    def augment(site: int, seen: set[int]) -> bool:
        for actual in sorted(edges[site]):
            if actual in seen:
                continue
            seen.add(actual)
            if actual not in right or augment(right[actual], seen):
                right[actual] = site
                return True
        return False

    for position in range(len(queue)):
        augment(position, set())
        if len(right) >= target:
            return position + 1
    return None


def evaluate_queue(name: str, queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame):
    rows = []
    checkpoints = sorted(set((*UI_BUDGETS, *BUDGETS, len(queue))))
    for budget in checkpoints:
        selected = queue.head(budget)
        matched = match_one_to_one(selected, np.c_[truth.geometry.x, truth.geometry.y], 20.0)
        rows.append(
            {
                "system": name,
                "budget": budget,
                "budget_label": "full" if budget == len(queue) else str(budget),
                "predicted_sites": len(selected),
                "candidate_population": len(queue),
                "matched": len(matched),
                "recall": len(matched) / len(truth),
                "matches_per_1000": 1000 * len(matched) / max(len(selected), 1),
                "median_localization_error_m": (
                    float(matched.distance_m.median()) if len(matched) else None
                ),
            }
        )
    return rows


def frontier(name: str, queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, extra: list[int]):
    checkpoints = sorted(
        set(
            [*BUDGETS, *range(FRONTIER_STEP, len(queue) + 1, FRONTIER_STEP), *extra, len(queue)]
        )
    )
    checkpoints = [value for value in checkpoints if 0 < value <= len(queue)]
    states = prefix_state(edges_for(queue, truth, 20.0), np.arange(len(queue)), checkpoints)
    return [
        {
            "system": name,
            "budget": budget,
            "matched": states[budget]["matches"],
            "recall": states[budget]["matches"] / len(truth),
            "candidate_population": len(queue),
        }
        for budget in checkpoints
    ]


def random_controls(pool: gpd.GeoDataFrame, truth: gpd.GeoDataFrame):
    pool_edges = edges_for(pool, truth, 20.0)
    counts = {budget: [] for budget in BUDGETS}
    for replicate in range(RANDOM_TRIALS):
        order = np.random.default_rng(RANDOM_SEED + replicate).choice(
            len(pool), size=max(BUDGETS), replace=False
        )
        states = prefix_state(pool_edges, order, BUDGETS)
        for budget in BUDGETS:
            counts[budget].append(states[budget]["matches"])
    rows = []
    for budget in BUDGETS:
        values = np.asarray(counts[budget])
        uniform = deterministic_uniform(pool, budget)
        uniform_matches = match_one_to_one(
            uniform, np.c_[truth.geometry.x, truth.geometry.y], 20.0
        )
        rows.append(
            {
                "budget": budget,
                "uniform_matches": len(uniform_matches),
                "random_trials": RANDOM_TRIALS,
                "random_mean_matches": float(values.mean()),
                "random_p025_matches": float(np.quantile(values, 0.025)),
                "random_p975_matches": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def route_region_metrics(name: str, queue: gpd.GeoDataFrame, truth: gpd.GeoDataFrame):
    rows = []
    for budget in (1000, 5000, len(queue)):
        selected = queue.head(budget)
        matched = match_one_to_one(selected, np.c_[truth.geometry.x, truth.geometry.y], 20.0)
        assigned = truth.iloc[matched.real_row.to_numpy()] if len(matched) else truth.iloc[:0]
        for column, stratum in (("Route", "route"), ("Region", "region")):
            for value, group in truth.groupby(column, dropna=False):
                rows.append(
                    {
                        "system": name,
                        "budget": budget,
                        "stratum": stratum,
                        "value": str(value),
                        "inventory": len(group),
                        "matched": int(assigned[column].astype(str).eq(str(value)).sum()),
                    }
                )
    return rows


def main() -> None:
    if (REPORT / "COMPLETE.json").exists():
        raise RuntimeError("Completed v6 experiment exists; use a new version instead of overwriting")
    REPORT.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    inputs = {
        "raw_candidates": PHYSICAL / "baseline_raw.parquet",
        "hybrid_sites": PHYSICAL / "variant_HYBRID-GROUPED.parquet",
        "baseline_sites": PHYSICAL / "baseline_sites.parquet",
        "baseline_queue": SELECTIVE / "queue_BASELINE-RF.parquet",
        "v3_queue": SPRINT / "ranked_physical_sites.parquet",
        "inventory": SUMMER / "inventory_coverage.parquet",
        "query_audit": PHYSICAL / "all_959_record_diagnosis.csv",
        "road_anchor_pool": PHYSICAL / "variant_sampling_10m.parquet",
        "heldout_validation": V5_MODEL / "validation_metrics.csv",
        "label_audit": V5_REPORT / "training_data_audit.json",
    }
    save_json(REPORT / "input_hashes.json", {key: sha256_file(path) for key, path in inputs.items()})

    raw = gpd.read_parquet(inputs["raw_candidates"])
    hybrid = gpd.read_parquet(inputs["hybrid_sites"])
    baseline_sites = gpd.read_parquet(inputs["baseline_sites"])
    candidates, _ = build_candidate_sites(hybrid, raw, baseline_sites)
    saved_v3 = gpd.read_parquet(inputs["v3_queue"])[
        ["candidate_id", "score_v3", "existing_score", "global_rank"]
    ]
    candidates = candidates.drop(
        columns=[column for column in saved_v3.columns if column != "candidate_id" and column in candidates]
    ).merge(saved_v3, on="candidate_id", validate="one_to_one")
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=raw.crs)

    baseline = gpd.read_parquet(inputs["baseline_queue"]).sort_values("global_rank", kind="stable")
    baseline_ids = set(baseline.candidate_id.astype(str))
    baseline_ranks = dict(zip(baseline.candidate_id.astype(str), baseline.global_rank.astype(int)))
    eligible = eligible_physical_union(candidates, baseline_ids)
    compact = compact_cross_generator_sites(eligible, baseline_ranks)
    compact_member_ids = set()
    for value in compact.member_candidate_ids:
        compact_member_ids.update(json.loads(value))
    if not baseline_ids.issubset(compact_member_ids):
        raise RuntimeError("Compact policy dropped a frozen-baseline candidate")
    if compact.geometry.isna().any() or compact.physical_site_id.duplicated().any():
        raise RuntimeError("Compact physical-site identity/geometry invariant failed")

    export_columns = [
        "physical_site_id", "candidate_id", "road_id", "road_name", "route_id",
        "road_part_index", "road_chainage_m", "proposal_kind", "candidate_source",
        "eligibility_reason", "physical_evidence_sources", "physical_evidence_count",
        "source_site_count", "source_family_count", "member_count", "member_candidate_ids",
        "existing_score", "score_RF", "score_HGB", "score_v3", "model_score",
        "global_rank", "flow_accumulation_cells", "upstream_area_m2",
        "distance_to_nearest_stream_m", "distance_to_nearest_mapped_crossing_m",
        "distance_to_dem_road_drainage_crossing_0p1km2_m",
        "route_elevation_dip_60m", "negative_tpi_9x9_m", "slope_degrees",
        "crossing_angle_degrees", "grouping_policy", "representative_selection",
        "model_version", "geometry",
    ]
    export_columns = [column for column in export_columns if column in compact]
    compact_export = compact[export_columns].copy()
    compact_export.to_parquet(REPORT / "ranked_compact_physical_sites.parquet", index=False)
    tabular = compact_export.copy()
    tabular["longitude"] = tabular.geometry.to_crs(4326).x
    tabular["latitude"] = tabular.geometry.to_crs(4326).y
    tabular.drop(columns="geometry").to_csv(REPORT / "ranked_compact_physical_sites.csv", index=False)
    compact_export.head(10000).to_crs(4326).to_file(
        REPORT / "ranked_compact_physical_sites_top10000.geojson", driver="GeoJSON"
    )

    truth_all = gpd.read_parquet(inputs["inventory"])
    audit = pd.read_csv(inputs["query_audit"])[["record_id", "query_success"]]
    truth_all = truth_all.merge(audit, on="record_id", validate="one_to_one")
    truth = truth_all.loc[truth_all.query_success].sort_values("source_excel_row").reset_index(drop=True)
    if len(truth_all) != 959 or len(truth) != 847:
        raise RuntimeError("Inventory scope changed")

    v3 = gpd.read_parquet(inputs["v3_queue"]).sort_values("global_rank", kind="stable")
    queues = {"original_rf": baseline, "v3_raw": v3, "v6_compact_physical": compact_export}
    target_matches = int(np.ceil(TARGET_RECALL * len(truth)))
    minimums = {
        name: minimum_budget(queue, truth, target_matches) for name, queue in queues.items()
    }
    metric_rows = []
    frontier_rows = []
    route_rows = []
    for name, queue in queues.items():
        metric_rows.extend(evaluate_queue(name, queue, truth))
        extra = [minimums[name]] if minimums[name] is not None else []
        frontier_rows.extend(frontier(name, queue, truth, extra))
        route_rows.extend(route_region_metrics(name, queue, truth))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT / "fixed_budget_metrics.csv", index=False)
    pd.DataFrame(frontier_rows).to_csv(REPORT / "coverage_frontier.csv", index=False)
    pd.DataFrame(route_rows).to_csv(REPORT / "route_region_metrics.csv", index=False)

    controls = random_controls(gpd.read_parquet(inputs["road_anchor_pool"]), truth)
    controls.to_csv(REPORT / "random_uniform_controls.csv", index=False)
    pd.read_csv(inputs["heldout_validation"]).to_csv(
        REPORT / "heldout_route_spatial_validation.csv", index=False
    )
    label_audit = json.loads(inputs["label_audit"].read_text())
    save_json(REPORT / "training_label_audit.json", label_audit)

    accounting = {
        "v3_hypotheses": len(v3),
        "pre_group_physical_union": len(eligible),
        "compact_physical_sites": len(compact_export),
        "removed_from_v3": len(v3) - len(compact_export),
        "reduction_fraction": (len(v3) - len(compact_export)) / len(v3),
        "multi_source_groups": int(compact.source_site_count.gt(1).sum()),
        "maximum_source_sites_per_group": int(compact.source_site_count.max()),
        "baseline_candidates_preserved_as_members": len(baseline_ids),
        "known_coordinates_used_for_admission_grouping_or_ranking": False,
    }
    save_json(REPORT / "candidate_accounting.json", accounting)
    policy = {
        "version": VERSION,
        "active_default": False,
        "experimental": True,
        "admission": "frozen baseline OR multiscale terrain extremum OR explicit mapped/DEM crossing",
        "grouping": compact.grouping_policy.iloc[0],
        "dem_resolution_m": DEM_RESOLUTION_M,
        "same_generator_duplicate_m": EXACT_DUPLICATE_M,
        "ranking": "baseline groups in frozen RF order, then maximum frozen-RF score; stable IDs break ties",
        "representative": compact.representative_selection.iloc[0],
        "evaluation_coordinates_used": False,
    }
    save_json(MODEL / "policy.json", policy)
    save_json(MODEL / "dataset_provenance.json", {
        "inputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for key, path in inputs.items()},
        "summer_inventory_used_for_queue_construction": False,
        "summer_inventory_used_for_retrospective_evaluation": True,
        "production_or_frozen_artifacts_modified": False,
    })

    def value(system: str, budget: int, column: str):
        return metrics.loc[(metrics.system.eq(system)) & metrics.budget.eq(budget), column].iloc[0]

    compact_full = int(value("v6_compact_physical", len(compact_export), "matched"))
    v3_full = int(value("v3_raw", len(v3), "matched"))
    table = [
        "| System | Sites | @250 | @500 | @1,000 | @2,000 | @5,000 | Full coverage | Minimum for 70% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"original_rf": "Original RF", "v3_raw": "v3 raw", "v6_compact_physical": "v6 compact physical"}
    for name, queue in queues.items():
        matches = [int(value(name, budget, "matched")) for budget in BUDGETS]
        full = int(value(name, len(queue), "matched"))
        minimum = "Not achievable" if minimums[name] is None else f"{minimums[name]:,}"
        table.append(
            f"| {labels[name]} | {len(queue):,} | {matches[0]} | {matches[1]} | {matches[2]} | {matches[3]} | {matches[4]} | {full} ({100*full/len(truth):.2f}%) | {minimum} |"
        )
    control_lines = []
    for row in controls.itertuples():
        control_lines.append(
            f"| {row.budget:,} | {row.uniform_matches} | {row.random_mean_matches:.1f} | {row.random_p025_matches:.0f}–{row.random_p975_matches:.0f} |"
        )
    report = f"""# Final physical-site discovery optimization — 2026-10-19

## Decision

**A practical 70% recall system is not supported by the available data.** The strongest defensible compact queue reaches the retrospective 70% threshold at **{minimums['v6_compact_physical']:,} recommendations** ({target_matches}/847), which is not an operational inspection budget. It contains **{len(compact_export):,}** physical sites and covers **{compact_full}/847 ({100*compact_full/len(truth):.2f}%)** when exhausted. The original RF remains the default because it is still strongest at practical budgets.

The 847-record inventory is development-exposed. These figures are retrospective retrieval, not independent recall.

## Coverage limit versus ranked recall

{chr(10).join(table)}

The original 13,065-site population cannot reach 70%; its coverage ceiling is 347/847. v3 can cover {v3_full}/847, but its saved ranking needs **{minimums['v3_raw']:,}** sites to reach 70%. The compact queue reduces v3 by **{accounting['removed_from_v3']:,} sites ({100*accounting['reduction_fraction']:.1f}%)** and reaches the same threshold {minimums['v3_raw']-minimums['v6_compact_physical']:,} inspections earlier, but 42k remains excessive.

`coverage_frontier.csv` separates full candidate coverage from prefix-ranked recall across the complete queue.

## Physical consolidation policy

The sealed policy keeps every frozen-baseline candidate as a group member, all independently detected multiscale terrain extrema, and explicit mapped/DEM crossing proposals. It excludes uncorroborated added internal road probes. Same-road proposals from different generators may consolidate only within one 10 m DEM cell; same-generator proposals require <=2 m coincidence. Complete-span chainage and Euclidean bounds prevent single-link chains and hairpin merges. Different non-empty channel identities never merge, preserving distinct nearby crossings. Representatives are existing coordinates—no point is created or moved.

This creates {accounting['multi_source_groups']:,} multi-source groups from {len(eligible):,} admitted proposals. Inventory coordinates were loaded only after this queue was sealed.

## Ranking and supervised-model limit

Baseline-containing groups retain frozen RF order; remaining sites use maximum frozen-RF score with stable tie breaking. This preserves 123 matches at 1,000. The verified-label-only v5 reranker remains rejected at 113/1,000. Its best conservative physical residual has mixed-route AUC 0.416. Only 3/14 training routes contain both classes, 11 are single-class, and 500/847 evaluation records lie within 25 m of a training positive. No genuinely untouched field inventory is available.

The selected physical model's three whole-route folds have ROC AUC **0.527, 0.711 and 0.329** (pooled OOF 0.691), while its three 5 km spatial folds have **0.684, 0.757 and 0.791**. After the conservative 10% baseline blend, within-route AUC on the only mixed routes is **0.439 on 212, 0.431 on 32 and 0.377 on 9W**. The fold spread and below-chance within-route results are why the learned reranker is not used in v6.

At 1,000 sites, v6's median localization error is **8.85 m**; at full coverage it is **11.24 m**. Region 6 retrieves 4/40 and Region 8 retrieves 119/807 at 1,000. Route-level counts for every budgeted system are in `route_region_metrics.csv`; these are retrospective strata, not untouched-route results.

## Random and uniform controls

| Budget | Uniform matches | Random mean | Random descriptive 95% interval |
|---:|---:|---:|---:|
{chr(10).join(control_lines)}

All controls use the fixed label-free 10 m road-anchor pool. Intervals describe seeded variability, not statistical significance.

## Labels required before another learned reranker

- On positive-only routes **208, 209, 213, 214, 28, 375, 42 and 52**, record at least 20 genuinely inspected non-culvert drainage opportunities per route (160 new verified negatives), sampled through continuous road-mile surveys rather than hand-picked points.
- Survey negative-only routes **299, 32A and 55** end to end and record every encountered culvert plus every inspected non-culvert crossing; do not impose a synthetic positive quota.
- Add at least 16 verified negatives on route **212** and 18 on **9W** so each mixed route has at least 20 negatives; retain route 32's existing mixed labels.
- Keep at least four newly completed corridors wholly untouched until candidate generation, grouping, feature extraction, model choice and thresholds are frozen. Every inspected location needs coordinates, route/chainage, culvert presence/absence, confidence, survey date and source identifier.

These are minimum balancing targets, not a guarantee of 70% recall. A credible independent claim requires the untouched corridors, including all culverts found during complete surveys.

## Reproducibility and safety

The ranked Parquet/CSV/GeoJSON, full frontier, route/region results, controls, label audit, grouped validation, policy and hashed provenance are saved with this report. The queue is experimental and selectable in staging; the original RF remains first/default. No production path, frozen model, deployment, commit or push was changed.
"""
    (REPORT / "REPORT.md").write_text(report)
    complete = {
        "version": VERSION,
        "sites": len(compact_export),
        "full_matches": compact_full,
        "full_recall": compact_full / len(truth),
        "minimum_budget_for_70_percent": minimums["v6_compact_physical"],
        "target_matches": target_matches,
        "baseline_top1000": int(value("original_rf", 1000, "matched")),
        "v6_top1000": int(value("v6_compact_physical", 1000, "matched")),
        "promotion_decision": "retain_original_rf_default",
        "production_changed": False,
        "frozen_models_changed": False,
    }
    save_json(REPORT / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
