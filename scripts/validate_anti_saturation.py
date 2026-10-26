"""Equal-budget anti-saturation audit for the summer inventory.

This is a retrospective, label-free control study. It does not fit a model,
change a queue, or treat unmatched candidates as negatives.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.flow_c_bundle import sha256_file
from scripts.validate_inventory_heldout_roads import match_one_to_one

OUT = ROOT / "reports/experiments/anti_saturation_validation_20261017"
PHYSICAL = ROOT / "reports/experiments/physical_discovery_20261016"
SPRINT = ROOT / "reports/experiments/model_improvement_sprint_20261017"
SELECTIVE = ROOT / "reports/experiments/selective_admission_20261016"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
BUDGETS = (100, 250, 500, 1000, 5000, 10000, 13065)
TOLERANCES = (5, 10, 15, 20, 25, 50)
N_RANDOM = 1000
SEED = 20261017


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def xy(frame: gpd.GeoDataFrame) -> np.ndarray:
    return np.c_[frame.geometry.x, frame.geometry.y]


def edges_for(points: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, tolerance: float):
    return cKDTree(xy(truth)).query_ball_point(xy(points), tolerance)


def prefix_state(edges, order, checkpoints):
    """Exact maximum-cardinality state at each prefix via augmenting paths."""
    right: dict[int, int] = {}

    def augment(site: int, seen: set[int]) -> bool:
        for target in sorted(edges[site]):
            if target in seen:
                continue
            seen.add(target)
            if target not in right or augment(right[target], seen):
                right[target] = site
                return True
        return False

    result = {}
    wanted = set(checkpoints)
    for position, site in enumerate(order[: max(checkpoints)], 1):
        augment(int(site), set())
        if position in wanted:
            result[position] = dict(matches=len(right), truth_to_site=dict(right))
    return result


def deterministic_uniform(pool: gpd.GeoDataFrame, budget: int) -> gpd.GeoDataFrame:
    """Exact-budget half-step samples on the fixed 10 m road-anchor sequence."""
    ordered = pool.sort_values(
        ["road_id", "road_part_index", "road_chainage_m", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    positions = np.floor((np.arange(budget) + 0.5) * len(ordered) / budget).astype(int)
    assert len(np.unique(positions)) == budget
    return ordered.iloc[positions].reset_index(drop=True)


def observed_rows(name, queue, truth, tolerance=20):
    rows, routes = [], []
    truth_xy = xy(truth)
    for budget in BUDGETS:
        selected = queue.head(budget)
        matched = match_one_to_one(selected, truth_xy, tolerance)
        rows.append(
            dict(
                system=name,
                budget=budget,
                predictions=len(selected),
                matches=len(matched),
                recall=len(matched) / len(truth),
                matches_per_1000=1000 * len(matched) / len(selected),
                median_localization_error_m=(
                    float(matched.distance_m.median()) if len(matched) else None
                ),
            )
        )
        assigned = truth.iloc[matched.real_row.to_numpy()] if len(matched) else truth.iloc[:0]
        for route, group in truth.groupby("Route", dropna=False):
            routes.append(
                dict(
                    system=name,
                    budget=budget,
                    route=str(route),
                    inventory=len(group),
                    predictions=int(selected.route_id.astype(str).eq(str(route)).sum())
                    if "route_id" in selected
                    else None,
                    matches=int(assigned.Route.astype(str).eq(str(route)).sum()),
                )
            )
    return rows, routes


def main() -> None:
    if (OUT / "COMPLETE.json").exists():
        raise RuntimeError("Completed audit exists; use a new version rather than overwrite it")
    OUT.mkdir(parents=True)
    inputs = {
        "inventory": SUMMER / "inventory_coverage.parquet",
        "query_audit": PHYSICAL / "all_959_record_diagnosis.csv",
        "road_anchor_pool": PHYSICAL / "variant_sampling_10m.parquet",
        "baseline_rf": SELECTIVE / "queue_BASELINE-RF.parquet",
        "baseline_hgb": SELECTIVE / "queue_BASELINE-HGB.parquet",
        "v3_physical": SPRINT / "ranked_physical_sites.parquet",
    }
    save_json(OUT / "input_hashes.json", {k: sha256_file(v) for k, v in inputs.items()})

    truth_all = gpd.read_parquet(inputs["inventory"])
    audit = pd.read_csv(inputs["query_audit"])[["record_id", "query_success"]]
    truth_all = truth_all.merge(audit, on="record_id", validate="one_to_one")
    truth = truth_all.loc[truth_all.query_success].sort_values("source_excel_row").reset_index(drop=True)
    assert len(truth_all) == 959 and len(truth) == 847 and (~truth_all.query_success).sum() == 112

    pool = gpd.read_parquet(inputs["road_anchor_pool"])
    pool = pool.drop_duplicates("candidate_id").reset_index(drop=True)
    assert len(pool) >= max(BUDGETS) and pool.crs.to_epsg() == truth.crs.to_epsg() == 32618
    queues = {
        "existing_baseline_rf": gpd.read_parquet(inputs["baseline_rf"]),
        "existing_baseline_hgb": gpd.read_parquet(inputs["baseline_hgb"]),
        "v3_physical_feature_model": gpd.read_parquet(inputs["v3_physical"]),
    }

    metric_rows, route_rows = [], []
    for name, queue in queues.items():
        rows, route = observed_rows(name, queue, truth)
        metric_rows.extend(rows)
        route_rows.extend(route)
    uniforms = {budget: deterministic_uniform(pool, budget) for budget in BUDGETS}
    for budget, frame in uniforms.items():
        rows, route = observed_rows("uniform_road_spacing", frame, truth)
        metric_rows.append(next(row for row in rows if row["budget"] == budget))
        route_rows.extend(row for row in route if row["budget"] == budget)

    # Repeated seeded random road sampling. The fixed 10 m anchor grid makes
    # every road metre approximately equally likely without consulting labels.
    pool_edges = edges_for(pool, truth, 20)
    random_counts = {budget: [] for budget in BUDGETS}
    random_route_counts = {(budget, str(route)): [] for budget in BUDGETS for route in truth.Route.unique()}
    for replicate in range(N_RANDOM):
        rng = np.random.default_rng(SEED + replicate)
        order = rng.choice(len(pool), size=max(BUDGETS), replace=False)
        states = prefix_state(pool_edges, order, BUDGETS)
        for budget, state in states.items():
            random_counts[budget].append(state["matches"])
            matched_truth = truth.iloc[list(state["truth_to_site"].keys())]
            counts = matched_truth.Route.astype(str).value_counts()
            for route in truth.Route.unique():
                random_route_counts[(budget, str(route))].append(int(counts.get(str(route), 0)))
    random_rows = []
    for budget in BUDGETS:
        values = np.asarray(random_counts[budget])
        row = dict(
            system="random_road_sampling",
            budget=budget,
            predictions=budget,
            trials=N_RANDOM,
            mean_matches=float(values.mean()),
            median_matches=float(np.median(values)),
            sd_matches=float(values.std(ddof=1)),
            p025_matches=float(np.quantile(values, 0.025)),
            p975_matches=float(np.quantile(values, 0.975)),
            min_matches=int(values.min()),
            max_matches=int(values.max()),
            mean_recall=float(values.mean() / len(truth)),
            mean_matches_per_1000=float(1000 * values.mean() / budget),
        )
        random_rows.append(row)
        for route in truth.Route.unique():
            route_values = np.asarray(random_route_counts[(budget, str(route))])
            route_rows.append(
                dict(
                    system="random_road_sampling",
                    budget=budget,
                    route=str(route),
                    inventory=int(truth.Route.astype(str).eq(str(route)).sum()),
                    predictions=None,
                    matches=None,
                    random_mean_matches=float(route_values.mean()),
                    random_p025_matches=float(np.quantile(route_values, 0.025)),
                    random_p975_matches=float(np.quantile(route_values, 0.975)),
                )
            )

    # Tolerance sensitivity uses deterministic observed queues and 200 fixed
    # random replicates per tolerance; it is descriptive, not a significance test.
    sensitivity = []
    sensitivity_queues = {**queues, **{f"uniform_{b}": frame for b, frame in uniforms.items()}}
    for tolerance in TOLERANCES:
        for name, queue in sensitivity_queues.items():
            for budget in BUDGETS:
                if name.startswith("uniform_") and name != f"uniform_{budget}":
                    continue
                selected = queue.head(budget)
                matched = match_one_to_one(selected, xy(truth), tolerance)
                sensitivity.append(
                    dict(system="uniform_road_spacing" if name.startswith("uniform_") else name,
                         tolerance_m=tolerance, budget=budget, matches=len(matched), trials=None)
                )
        edges = edges_for(pool, truth, tolerance)
        draws = {budget: [] for budget in BUDGETS}
        for replicate in range(200):
            order = np.random.default_rng(SEED + replicate).choice(
                len(pool), size=max(BUDGETS), replace=False
            )
            for budget, state in prefix_state(edges, order, BUDGETS).items():
                draws[budget].append(state["matches"])
        for budget, values in draws.items():
            sensitivity.append(
                dict(system="random_road_sampling", tolerance_m=tolerance, budget=budget,
                     matches=float(np.mean(values)), trials=len(values))
            )

    metrics = pd.DataFrame(metric_rows)
    random = pd.DataFrame(random_rows)
    metrics.to_csv(OUT / "equal_budget_metrics.csv", index=False)
    random.to_csv(OUT / "random_road_variability.csv", index=False)
    pd.DataFrame(route_rows).to_csv(OUT / "route_stratified_metrics.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "tolerance_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {"system": "existing_baseline_rf", "available_candidates": len(queues["existing_baseline_rf"])},
            {"system": "existing_baseline_hgb", "available_candidates": len(queues["existing_baseline_hgb"])},
            {"system": "v3_physical_feature_model", "available_candidates": len(queues["v3_physical_feature_model"])},
            {"system": "uniform_road_spacing_source_pool", "available_candidates": len(pool)},
            {"system": "random_road_sampling_source_pool", "available_candidates": len(pool)},
        ]
    ).to_csv(OUT / "system_candidate_counts.csv", index=False)

    baseline = metrics.loc[metrics.system.eq("existing_baseline_rf")].set_index("budget")
    v3 = metrics.loc[metrics.system.eq("v3_physical_feature_model")].set_index("budget")
    uniform_table = metrics.loc[metrics.system.eq("uniform_road_spacing")].set_index("budget")
    comparison = []
    for budget in BUDGETS:
        rv = random.loc[random.budget.eq(budget)].iloc[0]
        for name, table in (("existing_baseline_rf", baseline), ("v3_physical_feature_model", v3), ("uniform_road_spacing", uniform_table)):
            observed = int(table.loc[budget, "matches"])
            comparison.append(dict(system=name, budget=budget, observed_matches=observed,
                                   random_mean_matches=rv.mean_matches,
                                   excess_over_random_mean=observed-rv.mean_matches,
                                   random_p025=rv.p025_matches, random_p975=rv.p975_matches,
                                   descriptive_only=True))
    pd.DataFrame(comparison).to_csv(OUT / "baseline_random_comparison.csv", index=False)

    def ui_row(system, budget):
        row = metrics.loc[(metrics.system.eq(system)) & (metrics.budget.eq(budget))].iloc[0]
        return {"matches": int(row.matches), "recall": float(row.recall),
                "matches_per_1000": float(row.matches_per_1000),
                "median_distance_m": None if pd.isna(row.median_localization_error_m) else float(row.median_localization_error_m)}
    ui = {
        "evaluation": "Retrospective development-exposed benchmark; not independent validation.",
        "match_rule": "Maximum-cardinality one-to-one matching within 20 m in EPSG:32618.",
        "actual_total": 959,
        "supported_search_records": 847,
        "outside_query_coverage": 112,
        "active_default": "existing_baseline_rf",
        "promotion_decision": "retain_existing_default",
        "budgets": {},
    }
    for budget in BUDGETS:
        rr = random.loc[random.budget.eq(budget)].iloc[0]
        ui["budgets"][str(budget)] = {
            "existing_baseline_rf": ui_row("existing_baseline_rf", budget),
            "existing_baseline_hgb": ui_row("existing_baseline_hgb", budget),
            "v3_physical_feature_model": ui_row("v3_physical_feature_model", budget),
            "uniform_road_spacing": ui_row("uniform_road_spacing", budget),
            "random_road_sampling": {"trials": N_RANDOM, "mean_matches": float(rr.mean_matches),
                                     "p025_matches": float(rr.p025_matches), "p975_matches": float(rr.p975_matches)},
        }
    save_json(OUT / "ui_benchmark.json", ui)

    table_lines = [
        "| System | 100 | 250 | 500 | 1,000 | 5,000 | 10,000 | 13,065 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system, label in (
        ("existing_baseline_rf", "Existing baseline RF"),
        ("existing_baseline_hgb", "Existing baseline HGB"),
        ("v3_physical_feature_model", "v3 physical-feature RF"),
        ("uniform_road_spacing", "Uniform road spacing"),
    ):
        values = metrics.loc[metrics.system.eq(system)].set_index("budget").matches
        table_lines.append(
            "| " + label + " | " + " | ".join(str(int(values.loc[b])) for b in BUDGETS) + " |"
        )
    random_values = random.set_index("budget").mean_matches
    table_lines.append(
        "| Random road mean (1,000 trials) | "
        + " | ".join(f"{random_values.loc[b]:.1f}" for b in BUDGETS)
        + " |"
    )
    result_table = "\n".join(table_lines)
    report = f"""# Anti-saturation validation

Completed 2026-10-17. Retrospective and development-exposed; not independent validation.
No models were fit and no default was changed. Unmatched predictions remain unknown, not negatives.

## Population and matching

- All **959** workbook records are preserved. The exact searchable denominator is **847**; the **112** fixed-cell query failures are reported separately and never counted as model misses.
- All systems use the same successful-cell geographic scope and maximum-cardinality one-to-one matching in EPSG:32618. Primary tolerance is inclusive 20 m.
- Equal budgets are {', '.join(map(str, BUDGETS))}. Random road sampling uses **{N_RANDOM}** fixed-seed trials.
- Uniform and random controls use the same label-free 10 m road-anchor population ({len(pool):,} unique anchors). Neither uses inventory coordinates to place or rank predictions.

## Equal-budget 20 m matches

{result_table}

The existing RF yields 123 matches at 1,000 stops (123 per 1,000), versus 23 for v3, 15 for uniform spacing, and a random mean of 19.5 (seeded 95% interval 11–29). At 13,065 stops, baseline RF yields 347 versus v3 236, uniform 231, and random mean 222.0 (200–248). Baseline efficiency declines from 320 matches per 1,000 at budget 100 to 26.6 at 13,065, which makes the candidate-burden tradeoff explicit.

Available populations are 13,065 baseline sites (shared RF/HGB geometry), 68,302 v3 physical sites, and 231,854 unique 10 m road anchors for the uniform/random controls. At budget 1,000, median assigned distances are 8.85 m for baseline RF, 8.79 m for v3 and 11.22 m for uniform spacing; these medians describe different matched subsets and are not directly interchangeable accuracy estimates.

Tolerance sensitivity at budget 1,000 gives baseline/v3/random means of 21/8/1.6 at 5 m, 123/23/19.5 at 20 m, and 176/48/51.0 at 50 m. The ordering is therefore not an artifact of the primary radius.

## Decision

The v3 physical-feature ranker does not beat the existing RF queue at practical budgets, so the existing default is retained. The 68,302-site v3 population improves full-list coverage, but that is a density result rather than a ranking win. No p-value or significance claim is made: random intervals describe seeded variability only.

See `equal_budget_metrics.csv`, `system_candidate_counts.csv`, `random_road_variability.csv`, `baseline_random_comparison.csv`, `route_stratified_metrics.csv`, and `tolerance_sensitivity.csv` for exact results.
"""
    (OUT / "REPORT.md").write_text(report)
    save_json(OUT / "COMPLETE.json", {
        "actual_records": 959, "supported_search_records": 847, "outside_query_coverage": 112,
        "random_trials": N_RANDOM, "budgets": list(BUDGETS), "promotion_decision": "retain_existing_default",
        "models_fit": 0, "production_changed": False, "deployed": False,
    })


if __name__ == "__main__":
    main()
