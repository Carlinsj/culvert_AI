"""Read-only score audit: no estimator fitting, feature building, or new ranking.

Evaluate the exact archived ROUTE-Z ranking behind the reported 41/250 result.
The provenance check is essential: that archived score normalized POINTWISE-GB,
not the RF FLOW-C score normalized by the current production implementation.
"""

from __future__ import annotations

import importlib.util
import json
import hashlib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/precision_policy_20261014"
EXP = ROOT / "reports/experiments"
POOL = EXP / "nysdot_ulster_consensus_precision_20261013/candidate_consensus_features.parquet"
SCORE = EXP / "nysdot_ulster_score_comparability_v19_20261006/out_of_fold_scores/route-z.parquet"
FLOW = EXP / "nysdot_ulster_ranking_v18_20261006/out_of_fold_predictions/flow_c.parquet"
POINT = EXP / "nysdot_ulster_ranking_v18_20261006/out_of_fold_predictions/pointwise_gb__authoritative_inventory.parquet"
EVALUATOR = EXP / "nysdot_ulster_ranking_v18_20261006/run_experiment.py"
INVENTORY = EXP / "nysdot_ulster_real_inventory_benchmark_v1_20261006/source_extract/small_culverts_r8_source.csv"
KS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250)
TARGETS = (0.80, 0.75, 0.70, 0.60, 0.50)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_distinct_counts(neighbors: list[list[int]]) -> np.ndarray:
    """Maximum bipartite matching size for every prefix, without duplicate hits.

    Append one recommendation at a time and find an augmenting path. Existing
    matches may move, so a greedy nearest-inventory assignment is not sufficient.
    """
    right_to_left: dict[int, int] = {}
    left_to_right: dict[int, int] = {}
    counts = []
    for start in range(len(neighbors)):
        queue = deque([start])
        seen_left = {start}
        parent: dict[int, int] = {}
        free = None
        while queue and free is None:
            left = queue.popleft()
            for right in neighbors[left]:
                if right in parent:
                    continue
                parent[right] = left
                if right not in right_to_left:
                    free = right
                    break
                next_left = right_to_left[right]
                if next_left not in seen_left:
                    seen_left.add(next_left)
                    queue.append(next_left)
        while free is not None:
            left = parent[free]
            previous = left_to_right.get(left)
            left_to_right[left] = free
            right_to_left[free] = left
            free = previous
        counts.append(len(right_to_left))
    return np.asarray(counts, dtype=int)


def precision_curve(sites: pd.DataFrame, inventory_xy: np.ndarray) -> pd.DataFrame:
    neighbors = cKDTree(inventory_xy).query_ball_point(sites[["x_m", "y_m"]].to_numpy(float), 20.0)
    hits = prefix_distinct_counts(neighbors)
    k = np.arange(1, len(sites) + 1)
    return pd.DataFrame({
        "recommendations": k, "distinct_confirmed": hits,
        "precision": hits / k, "distinct_site_recall": hits / len(inventory_xy),
    })


def threshold_curve(sites: pd.DataFrame, score: str, inventory_xy: np.ndarray) -> pd.DataFrame:
    # This order enumerates selected SETS for scalar score>=threshold only.
    # No new representatives are chosen; field output retains archived rank.
    ordered = sites.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    curve = precision_curve(ordered, inventory_xy)
    curve["score"] = score
    curve["threshold_inclusive"] = ordered[score].to_numpy(float)
    # A numeric cutoff includes all equal-score sites; never split a tie.
    curve = curve.loc[~curve["threshold_inclusive"].duplicated(keep="last")].copy()
    return curve


def summary_row(curve: pd.DataFrame, target: float, scope: str) -> dict:
    eligible = curve.loc[curve["precision"].ge(target)]
    if eligible.empty:
        return {"target_precision": target, "scope": scope, "maximum_k": 0,
                "distinct_confirmed": 0, "precision": None, "distinct_site_recall": None,
                "threshold_inclusive": None}
    row = eligible.sort_values("recommendations").iloc[-1]
    return {"target_precision": target, "scope": scope,
            "maximum_k": int(row["recommendations"]), "distinct_confirmed": int(row["distinct_confirmed"]),
            "precision": float(row["precision"]), "distinct_site_recall": float(row["distinct_site_recall"]),
            "threshold_inclusive": float(row["threshold_inclusive"]) if "threshold_inclusive" in row else None}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("precision_existing_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = sorted((ROOT / "models").glob("flow_c_frozen_v1.joblib*")) + [
        ROOT / "src/culvert_ai/candidates.py", ROOT / "src/culvert_ai/features.py",
        ROOT / "src/culvert_ai/inference.py", ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg",
        POOL, SCORE, FLOW, POINT, EVALUATOR, INVENTORY,
    ]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    columns = ["candidate_id", "route", "x_m", "y_m", "latitude", "longitude", "flow_c_score", "pointwise_score", "route_z_score"]
    pool = pd.read_parquet(POOL, columns=columns)
    if len(pool) != 51417 or not pool["candidate_id"].is_unique:
        raise AssertionError("The frozen pool must contain 51,417 unique candidates.")
    source_differences = {}
    for path, column in ((SCORE, "route_z_score"), (FLOW, "flow_c_score"), (POINT, "pointwise_score")):
        saved = pd.read_parquet(path).set_index("candidate_id")["outer_oof_score"]
        actual = pool["candidate_id"].map(saved).to_numpy(float)
        mismatch = actual != pool[column].to_numpy(float)
        source_differences[column] = pool.loc[mismatch, ["candidate_id", "route"]].to_dict("records")
        if not np.isfinite(actual).all() or (mismatch & pool["route"].ne("<unassigned>").to_numpy()).any():
            raise AssertionError(f"Archived score mismatch: {column}")
        # Use original saved scores verbatim, including its unassigned fallback.
        pool[column] = actual

    def normalized_error(column: str) -> float:
        grouped = pool.groupby("route")[column]
        center = grouped.transform("mean")
        scale = grouped.transform(lambda x: x.std(ddof=0)).where(lambda x: x > 1e-12, 1.0)
        error = np.abs((pool[column] - center) / scale - pool["route_z_score"])
        return float(error.loc[pool["route"].ne("<unassigned>")].max())

    point_error, flow_error = normalized_error("pointwise_score"), normalized_error("flow_c_score")
    if point_error > 1e-12:
        raise AssertionError("Saved ROUTE-Z provenance changed.")
    evaluator = load_evaluator()
    real = evaluator.load_real()
    inventory_xy = real[["x", "y"]].to_numpy(float)
    sites = evaluator.nms_sites(pool, "route_z_score", "ARCHIVED_ROUTE_Z")
    curve = precision_curve(sites, inventory_xy)
    # Cross-check every requested K and the complete queue against the historical
    # scipy sparse maximum-bipartite-matching evaluator.
    for k in (*range(1, 251), len(sites)):
        matches, _ = evaluator.match_sites(sites, inventory_xy, top_k=k)
        if len(matches) != int(curve.iloc[k - 1]["distinct_confirmed"]):
            raise AssertionError(f"Distinct matching differs at K={k}")
    if tuple(curve.iloc[[99, 249]]["distinct_confirmed"]) != (18, 41):
        raise AssertionError("Historical 18/100 and 41/250 baselines failed reproduction.")
    curve.to_csv(OUT / "cumulative_precision_full_queue.csv", index=False)
    curve.head(250).to_csv(OUT / "cumulative_precision_1_to_250.csv", index=False)
    requested = curve.loc[curve["recommendations"].isin(KS)]
    requested.to_csv(OUT / "precision_at_k.csv", index=False)
    operating = [summary_row(part, target, scope)
                 for scope, part in (("K_1_to_250", curve.head(250)), ("full_queue", curve))
                 for target in TARGETS]
    pd.DataFrame(operating).to_csv(OUT / "operating_points.csv", index=False)

    thresholds = pd.concat([threshold_curve(sites, score, inventory_xy)
                            for score in ("route_z_score", "flow_c_score")], ignore_index=True)
    thresholds.to_csv(OUT / "score_thresholds.csv", index=False)
    threshold_points = [summary_row(part, target, score)
                        for score, part in thresholds.groupby("score") for target in TARGETS]
    pd.DataFrame(threshold_points).to_csv(OUT / "score_threshold_operating_points.csv", index=False)

    bin_rows = []
    for score in ("route_z_score", "flow_c_score"):
        # Keep the canonical site population. Ties are included in the higher
        # score bin; the actual bin count is explicitly reported.
        order = sites[score].sort_values(ascending=False).to_numpy(float)
        previous = pd.Series(False, index=sites.index)
        for frac in (.01, .02, .05, .10, 1.0):
            threshold = float(order[int(np.ceil(len(order) * frac)) - 1])
            included = sites[score].ge(threshold)
            groups = [("cumulative", included), ("exclusive", included & ~previous)] if frac < 1 else [("exclusive", ~previous)]
            for kind, mask in groups:
                part = sites.loc[mask]
                matches, _ = evaluator.match_sites(part, inventory_xy)
                bin_rows.append({
                    "score": score, "bin_type": kind,
                    "upper_percentile": int(frac * 100),
                    "recommendations": len(part), "distinct_confirmed": len(matches),
                    "precision": len(matches) / len(part) if len(part) else None,
                    "distinct_site_recall": len(matches) / len(real),
                    "minimum_score": float(part[score].min()) if len(part) else None,
                    "maximum_score": float(part[score].max()) if len(part) else None,
                })
            previous = included
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(OUT / "score_bin_precision.csv", index=False)

    route_rows = []
    # Context at Top250 is reported explicitly as REVIEW, never as a 70% tier.
    top = sites.head(250)
    global_matches, _ = evaluator.match_sites(top, inventory_xy)
    match_rows = set(global_matches["site_row"])
    top = top.assign(distinct_match=[i in match_rows for i in range(len(top))])
    for route, part in top.groupby("route"):
        route_rows.append({"scope": "Top250_review_only", "route": route,
                           "recommendations": len(part), "distinct_confirmed": int(part["distinct_match"].sum()),
                           "precision": float(part["distinct_match"].mean()), "share": len(part) / 250})
    pd.DataFrame(route_rows).to_csv(OUT / "route_distribution_top250_review.csv", index=False)
    sites.head(250)[["rank", "candidate_id", "route", "latitude", "longitude", "route_z_score", "flow_c_score"]].to_csv(
        OUT / "archived_top250_review.csv", index=False)

    provenance = {
        "archived_score_source": str(SCORE.relative_to(ROOT)),
        "archived_route_z_base_estimator": "POINTWISE-GB__AUTHORITATIVE-INVENTORY (V18)",
        "production_route_z_base_estimator": "FLOW-C RandomForestClassifier",
        "max_abs_error_saved_route_z_vs_z_of_pointwise": point_error,
        "max_abs_error_saved_route_z_vs_z_of_flow_c": flow_error,
        "historical_41_of_250_is_not_a_production_rf_route_z_validation": True,
        "candidate_count": len(pool), "archived_consolidated_sites": len(sites),
        "coverage_650_of_714_origin": "FLOW-C score-ordered 20m representatives, not raw candidate coverage; prior benchmark retained as historical reference only",
        "score_values_unchanged": True, "new_models_fit": 0, "new_features_built": 0,
        "later_consensus_copy_differences": source_differences,
        "provenance_error_scope": "named routes; original archived unassigned fallback retained verbatim",
        "threshold_selection_validation": {
            "status": "not_supported_by_saved_artifacts",
            "reason": "Each route has only its single outer OOF score. Other routes' OOF models can include the evaluated route's labels. Selecting a cutoff on those scores is not an independent outer-fold validation. Nested inner OOF scores excluding each outer route are not preserved by these runners; no models were fit.",
        },
    }
    (OUT / "provenance_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")

    empirical70 = [r for r in threshold_points if r["target_precision"] == .7 and r["maximum_k"] > 0]
    best = curve.loc[curve["precision"].idxmax()]
    # Fail closed: a label-selected OOF threshold alone is descriptive, not an
    # independently validated field guarantee; the score source also differs.
    policy = {
        "target_precision": .70, "high_confidence_enabled": False,
        "validated_threshold": None, "high_confidence_recommendation_count": 0,
        "empirical_70_percent_scalar_threshold_found": bool(empirical70),
        "all_recommendations_mode": "preserve existing FLOW-C + ROUTE-Z ranking; REVIEW / field verification required",
        "score_interpretation": "relative priority, not probability or validated confidence",
        "high_confidence_mode_message": "No validated 70% precision tier is available. Use All recommendations for research field review.",
        "default_field_use": "review_only",
        "conclusion": "70% NOT SUPPORTED BY CURRENT FEATURES",
        "production_threshold_transfer_allowed": False,
        "historical_maximum_prefix_precision": float(best["precision"]),
        "historical_maximum_prefix_precision_k": int(best["recommendations"]),
        "nested_threshold_validation_performed": False,
        "high_confidence_route_coverage": {
            "routes_represented": 0, "precision_by_route": [],
            "largest_route_share": None, "worst_route_precision": None,
            "reason": "No qualifying operating point exists; empty recommendations have undefined precision.",
        },
    }
    (OUT / "field_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    if before != after:
        raise AssertionError("A protected input or production asset changed during the audit.")
    (OUT / "input_integrity.json").write_text(json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": True}, indent=2) + "\n")
    lines = [
        "# Precision-controlled field recommendation audit — 2026-10-14", "",
        "**70% NOT SUPPORTED BY CURRENT FEATURES.** No nonempty global prefix or single-score threshold on the archived final sites reaches even 50% precision.", "",
        f"The best cumulative precision is {int(best['distinct_confirmed'])}/{int(best['recommendations'])} = {best['precision']:.2%}. This is a retrospective maximum, not an independently validated field threshold.", "",
        "## Exact scope and reproducibility", "",
        "This audit reads saved whole-route OOF scores; it fits no model, builds no features, changes no score, and does not regenerate candidates. Run `scripts/python.sh scripts/audit_field_precision.py`. Input and frozen production hashes are recorded before and after in `input_integrity.json`.", "",
        "The 51,417 frozen candidates are consolidated into 15,305 sites using the historical descending ROUTE-Z / candidate-ID tie-break and greedy 20 m rule. Distinct matches use the unchanged 20 m maximum one-to-one bipartite assignment against 714 inventory records. Every K from 1–250 and the complete queue are cross-checked against the historical evaluator. Unmatched recommendations are inventory-unmatched, not necessarily field-proven negatives.", "",
        "## Precision by inspection budget", "",
        "| Recommendations | Distinct confirmed | Precision | Distinct site recall |",
        "|---:|---:|---:|---:|",
    ]
    for row in requested.itertuples():
        lines.append(f"| {row.recommendations} | {row.distinct_confirmed} | {row.precision:.2%} | {row.distinct_site_recall:.2%} |")
    lines += ["", "## Abstention operating points", "",
              "| Target precision | Largest qualifying K (1–250) | Largest qualifying K (full queue) |",
              "|---:|---:|---:|"]
    for target in TARGETS:
        values = [row["maximum_k"] for row in operating if row["target_precision"] == target]
        lines.append(f"| {target:.0%} | {values[0]} | {values[1]} |")
    lines += ["", "Zero means no nonempty qualifying set, not a precision estimate for an empty set. All unique scalar cutoffs for saved ROUTE-Z and saved FLOW-C on the same final site population were examined. Cutoffs include every tied score; no route filtering, two-score combination search, or label-based site selection was used. FLOW-C thresholds are diagnostic subsets of the frozen final sites, not a replacement global ranking.", "",
              "## Score-bin precision", "",
              "Top-percent rows below are cumulative. Remaining is the disjoint bottom 90%. Percentages refer to the 15,305 consolidated recommendations, with all cutoff ties included.", "",
              "| Score | Bin | Confirmed / recommendations | Precision |",
              "|---|---|---:|---:|"]
    selected_bins = bins.loc[bins["bin_type"].eq("cumulative") | bins["upper_percentile"].eq(100)]
    for row in selected_bins.itertuples():
        label = "Remaining 90%" if row.upper_percentile == 100 else f"Top {row.upper_percentile}%"
        lines.append(f"| {row.score} | {label} | {row.distinct_confirmed}/{row.recommendations} | {row.precision:.2%} |")
    lines += ["", "Separate non-overlapping 0–1%, 1–2%, 2–5%, 5–10%, and remaining bins are also saved in `score_bin_precision.csv`. Distinct matching is computed independently for each subset; counts from different bins are not additive when the same inventory record is near recommendations in multiple bins.", "",
              "The broader upper tail is enriched relative to the remainder, but the first 10 recommendations have no matches. Thus coarse score enrichment does not establish a reliable tiny high-confidence tier. Neither score is a calibrated probability.", "",
              "## Route generalization and threshold validation", "",
              "There is no 70% operating point: qualifying sites = 0, represented routes = 0, and largest-route share/worst-route precision are undefined. For context only, the ordinary Top250 REVIEW queue has:", "",
              "| Route | Recommendations | Confirmed | Precision | Share |",
              "|---|---:|---:|---:|---:|"]
    for row in route_rows:
        lines.append(f"| {row['route']} | {row['recommendations']} | {row['distinct_confirmed']} | {row['precision']:.2%} | {row['share']:.1%} |")
    lines += ["", "The review queue represents 3 routes; the largest share is 72.4% and worst represented-route precision is 13.73%. These are not high-confidence results.", "",
              provenance["threshold_selection_validation"]["reason"], "",
              "Therefore no training-only/nested threshold validation is claimed. A simple leave-one-route-out cutoff over the already pooled OOF table would not resolve that dependence. No qualifying descriptive 70% cutoff exists to propose for independent validation in any event.", "",
              "## Provenance correction affecting deployment", "",
              f"The archived ROUTE-Z vector agrees with z-normalized POINTWISE-GB OOF scores within {point_error:.3g}, but differs from z-normalized Random Forest FLOW-C OOF scores by as much as {flow_error:.6f}. This check verifies lineage only; normalized diagnostic vectors are never used as a new ranking.", "",
              "The source is explicit in V19 `run_experiment.py`: `target_raw = target['raw_point']`; the later consensus runner also normalizes `pointwise_score`. Production `src/culvert_ai/inference.py` normalizes `flow_c_score`. Consequently the 41/250 historical benchmark is reproduced here but must not be attributed to the current production Random Forest + ROUTE-Z implementation. No threshold from that archived vector can be transferred as validated production confidence.", "",
              "The later consensus copy changes the single unassigned candidate's fallback to zero. This audit uses the original V19 scores verbatim, including its original fallback; named routes agree exactly. The requested Top100 and Top250 reproduce 18 and 41.", "",
              "The 650/714 = 91.04% number originates in the real-inventory benchmark's FLOW-C score-ordered consolidated representatives (15,650 sites). It is not raw candidate coverage and is not a score-independent invariant. No candidate coordinates have changed in this audit; that historical figure is retained as provenance, not hard-coded as newly verified coverage.", "",
              "The preceding V23 comparison also used this mislabeled 41/250 baseline against an augmented Random Forest. Its comparison therefore did not isolate feature changes alone. The saved 14/250 result remains an observed result of that implementation, but it cannot establish a causal feature-only performance change. No V23 refit or further experiment was performed.", "",
              "## Exact field policy", "",
              "- Keep HIGH-CONFIDENCE FIELD MODE unavailable; validated threshold is null and no site is designated as meeting 70% precision.",
              "- Allow ALL RECOMMENDATIONS for research field review, preserving the existing production ranking. Mark every exported recommendation `review`.",
              "- Show: 'Field review only. A 70% precision tier has not been validated. Priority shows relative rank; every recommendation needs field verification.'",
              "- Record confirmed, absent, and uncertain field outcomes separately. Do not display scores as culvert probabilities or promote the model based on this audit.",
              "- A 70% deployment claim requires independent evidence using the exact deployed score and operating context. The current saved evidence does not support it; new evidence is required, and no further same-data model search is proposed.", "",
              "Machine-readable policy: `field_policy.json`. Detailed curves, tie-complete thresholds, bins, route counts, and provenance are saved alongside this report. Production model, candidates, features, and normalization code remained byte-identical during the audit."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(requested.to_string(index=False))
    print(json.dumps({"policy": policy, "operating_points": operating, "threshold_operating_points": threshold_points, "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
