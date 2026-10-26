"""Reconcile queues, source provenance and proper existing-model road holdouts.

No fitting. Run after evaluate_selective_retrieval.py and before sealing.
"""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.evaluate_selective_retrieval import OLD, SUMMER, KS, matches, match_ids, save, evaluate
from scripts.diagnose_summer_discovery import xy, nearest
from staging import selective_sites as s
from staging.feature_sites import ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert (out / "COMPUTATIONS_COMPLETE.json").exists() and not (out / "COMPLETE.json").exists()
    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    fold_assignment = pd.read_csv(out / "inventory_fold_assignment.csv").set_index("record_id")
    truth = (
        truth.loc[truth.record_id.isin(fold_assignment.index)]
        .sort_values("source_excel_row", kind="stable")
        .reset_index(drop=True)
    )
    truth["research_fold"] = truth.record_id.map(fold_assignment.research_fold)
    baseline = gpd.read_parquet(OLD / "variant_BASELINE.parquet")
    hybrid = gpd.read_parquet(OLD / "variant_HYBRID.parquet")
    populations = {}
    group_rows = []
    for rule in s.RULES:
        population = s.population(baseline, hybrid, rule)
        grouped = s.anchored_groups(population)
        # All member evidence is stored once for this grouping, not per policy.
        grouped.to_parquet(out / f"provenance_anchored_{rule}.parquet", index=False)
        populations[rule] = grouped
        for name, frame in [("before", population), ("after", grouped)]:
            group_rows.append(
                dict(
                    rule=rule,
                    stage=name,
                    hypotheses=len(frame),
                    matched20=len(matches(frame, truth)),
                    baseline_representatives_preserved=set(baseline.candidate_id).issubset(
                        frame.candidate_id
                    ),
                )
            )
    pd.DataFrame(group_rows).to_csv(out / "grouping_accounting.csv", index=False)
    choice = json.loads((out / "all_development_choice.json").read_text())
    # Report physically informed nonzero weights, not just the best setting
    # (which may select zero physical bonus). These are declared grid members.
    extra_metrics, extra_routes, extra_records = [], [], []
    for weight in (0.1, 0.25):
        frame = populations[choice["rule"]]
        evaluate(
            f"FIXED-PHYSICAL-{weight}",
            s.order(frame, weight=weight, penalty=0.1),
            truth,
            match_ids(s.order(baseline), truth),
            len(frame),
            out,
            extra_metrics,
            extra_routes,
            extra_records,
        )
    nested = gpd.read_parquet(out / "queue_NESTED-SELECTIVE.parquet")
    evaluate(
        "NESTED-SITE-MODEL",
        s.order(nested, score="score_site_oof"),
        truth,
        match_ids(s.order(baseline), truth),
        len(nested),
        out,
        extra_metrics,
        extra_routes,
        extra_records,
        scope="crossfit_retrospective",
    )
    for existing, rows in [
        ("comparison.csv", pd.DataFrame(extra_metrics)),
        ("region_route_results.csv", pd.DataFrame(extra_routes)),
        ("record_results.csv", pd.DataFrame(extra_records)),
    ]:
        old = pd.read_csv(out / existing)
        old = old.loc[~old.system.isin(rows.system.unique())]
        pd.concat([old, rows], ignore_index=True).to_csv(out / existing, index=False)
    metrics = pd.read_csv(out / "comparison.csv")
    budget_rows = []
    concentrations = []
    paired = []
    base_queue = s.order(baseline)
    base_nearest = nearest(base_queue, truth)[0]
    all_queues = {
        name: gpd.read_parquet(out / f"queue_{name}.parquet") for name in metrics.system.unique()
    }
    for name, queue in all_queues.items():
        for budget in (*KS, "all"):
            part = queue if budget == "all" else queue.head(budget)
            bpart = base_queue if budget == "all" else base_queue.head(budget)
            ids, bids = match_ids(part, truth), match_ids(bpart, truth)
            use = (
                metrics.system.eq(name)
                & metrics.budget.astype(str).eq(str(budget))
                & metrics.tolerance_m.eq(20)
            )
            assert use.sum() == 1
            metrics.loc[use, "added_vs_baseline20"] = len(ids - bids)
            metrics.loc[use, "lost_vs_baseline20"] = len(bids - ids)
            for i, row in truth.iterrows():
                budget_rows.append(
                    dict(
                        system=name,
                        budget=budget,
                        record_id=row.record_id,
                        source_excel_row=row.source_excel_row,
                        Region=row.Region,
                        Route=row.Route,
                        matched20=row.record_id in ids,
                        baseline_matched_at_same_budget20=row.record_id in bids,
                        change="added"
                        if row.record_id in ids - bids
                        else "lost"
                        if row.record_id in bids - ids
                        else "retained"
                        if row.record_id in ids
                        else "unmatched",
                        independent=False,
                    )
                )
            if budget != "all":
                pairs = cKDTree(xy(part)).query_pairs(40)
                pairs = [
                    (i, j)
                    for i, j in pairs
                    if str(part.iloc[i].road_id) == str(part.iloc[j].road_id)
                    and part.iloc[i].road_part_index == part.iloc[j].road_part_index
                    and abs(part.iloc[i].road_chainage_m - part.iloc[j].road_chainage_m) <= 40
                ]
                counts = part.road_name.fillna(part.road_id).value_counts()
                full_names = set(queue.road_name.fillna(queue.road_id))
                concentrations.append(
                    dict(
                        system=name,
                        budget=budget,
                        stops=len(part),
                        road_names=len(counts),
                        unrepresented_available_road_names=len(full_names - set(counts.index)),
                        max_road_name_share=float(counts.max() / len(part)),
                        near40m_same_road_pairs=len(pairs),
                        near40m_sites=len({i for pair in pairs for i in pair}),
                        note="Names can span segments; proximity is not proof of duplicate culverts",
                    )
                )
        nd = nearest(queue, truth)[0]
        for i, row in truth.iterrows():
            paired.append(
                dict(
                    system=name,
                    record_id=row.record_id,
                    baseline_nearest_m=base_nearest[i],
                    challenger_nearest_m=nd[i],
                    change_m=nd[i] - base_nearest[i],
                    within20m_both=bool(nd[i] <= 20 and base_nearest[i] <= 20),
                )
            )
    metrics.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(out / "budget_record_changes.csv", index=False)
    pd.DataFrame(concentrations).to_csv(out / "all_queue_concentration.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_nearest_localization.csv", index=False)
    # Proper existing-model control: reselect baseline representatives from the
    # excluded-road score, instead of reusing exposed final-RF-selected geometry.
    recovered = gpd.read_parquet(
        ROOT
        / "reports/experiments/real_inventory_heldout_roads_20261015/recovered_road_segments.parquet"
    )
    (left, _), distances = recovered.sindex.nearest(
        truth.geometry, return_all=False, return_distance=True
    )
    eligible = np.zeros(len(truth), bool)
    eligible[left] = distances <= 100
    actual = truth.loc[eligible].reset_index(drop=True)
    assert len(actual) == 845
    choice = json.loads((out / "all_development_choice.json").read_text())
    road_results = []
    road_routes = []
    for family in ("RF", "HGB"):
        frame = hybrid.loc[hybrid[f"score_{family}_roadheldout"].notna()].copy()
        frame[f"score_{family}"] = frame[f"score_{family}_roadheldout"]
        frame["model_score"] = frame[f"score_{family}"]
        base = ranked(frame.loc[frame.evidence_admitted].copy(), "raw_rf")
        selected = s.anchored_groups(s.population(base, frame, choice["rule"]))
        # Baseline raw score counts must match the sealed proper road control.
        expected = {"RF": (329, 128), "HGB": (323, 110)}[family]
        assert (len(matches(base, actual)), len(matches(base.head(1000), actual))) == expected
        queues = {
            "BASELINE": s.order(base, score=f"score_{family}"),
            "SELECTIVE": s.order(selected, score=f"score_{family}"),
            "SELECTIVE-DIVERSITY": s.order(
                selected,
                score=f"score_{family}",
                weight=choice["weight"],
                penalty=choice["penalty"],
            ),
        }
        for name, queue in queues.items():
            for k in (*KS, "all"):
                part = queue if k == "all" else queue.head(k)
                assigned = matches(part, actual)
                ids = set(actual.iloc[assigned.real_row].record_id)
                road_results.append(
                    dict(
                        family=family,
                        system=name,
                        budget=k,
                        stops=len(part),
                        sites=len(queue),
                        records=len(actual),
                        matched=len(assigned),
                        excluded_own_corridor_from_model_training=True,
                        representatives_reselected_with_excluded_road_scores=True,
                        policy_selected_on_exposed_development_data=True,
                        independent=False,
                    )
                )
                for (region, route), sub in actual.groupby(["Region", "Route"]):
                    road_routes.append(
                        dict(
                            family=family,
                            system=name,
                            budget=k,
                            Region=region,
                            Route=route,
                            records=len(sub),
                            matched=int(sub.record_id.isin(ids).sum()),
                        )
                    )
    pd.DataFrame(road_results).to_csv(
        out / "proper_existing_model_road_comparison.csv", index=False
    )
    pd.DataFrame(road_routes).to_csv(out / "proper_existing_model_road_routes.csv", index=False)
    # Compact old queue files generated before the evaluation storage refinement.
    keep = {
        "geometry",
        "candidate_id",
        "physical_site_id",
        "road_id",
        "road_name",
        "route_id",
        "road_part_index",
        "road_chainage_m",
        "proposal_kind",
        "baseline_anchor",
        "selective_rule",
        "research_fold",
        "score_RF",
        "score_HGB",
        "score_site_oof",
        "score_site_descriptive",
        "research_priority",
        "global_rank",
        "chosen_rule",
        "chosen_weight",
        "chosen_penalty",
        "member_count",
    }
    for name, frame in all_queues.items():
        frame[[c for c in frame if c in keep]].to_parquet(
            out / f"queue_{name}.parquet", index=False
        )
    save(
        out / "SUPPLEMENT_COMPLETE.json",
        dict(
            fits=0,
            proper_model_road_rows=len(road_results),
            all_queue_budget_rows=len(concentrations),
            grouping_rows=len(group_rows),
            independent_counts_checked=True,
            default_changed=False,
        ),
    )
    print(pd.DataFrame(road_results).query("budget==1000").to_string(index=False))


if __name__ == "__main__":
    main()
