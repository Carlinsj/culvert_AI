"""Bounded retrospective admission/ranking study; sealed inputs are read-only.

Run with scripts/python.sh scripts/evaluate_selective_retrieval.py --output DIR.
The output must contain METHOD.md and must not already be complete. Never served
automatically, never overwrites a frozen model, never labels unmatched sites zero.
"""

# ruff: noqa: E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from culvert_ai.flow_c_bundle import sha256_file
from scripts.diagnose_summer_discovery import nearest, xy
from scripts.evaluate_hydrology_sites import protected
from scripts.validate_inventory_heldout_roads import match_one_to_one
from scripts.verify_physical_discovery import independent_count
from staging import selective_sites as selective
from staging import site_ranker

OLD = ROOT / "reports/experiments/physical_discovery_20261016"
SUMMER = ROOT / "reports/experiments/summer_inventory_20261016"
HYDRO = ROOT / "reports/experiments/hydrology_driven_sites_20261015"
KS = (100, 250, 500, 1000)
TOLERANCES = (5, 10, 15, 20, 25, 50)
WEIGHTS = (0.0, 0.1, 0.25)
PENALTIES = (0.0, 0.1)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def matches(queue, truth, tolerance=20):
    result = match_one_to_one(queue, xy(truth), tolerance)
    assert len(result) == independent_count(queue, truth, tolerance)
    return result


def match_ids(queue, truth):
    return set(truth.iloc[matches(queue, truth).real_row].record_id)


def source(frame, baseline):
    baseline_ids = set(baseline.candidate_id)
    values = frame.proposal_kind.astype(str).copy()
    values.loc[frame.proposal_kind.eq("internal_road_probe")] = "adaptive_other_probe"
    values.loc[selective.flag(frame, "adaptive_profile")] = "adaptive_profile"
    values.loc[selective.flag(frame, "adaptive_flow")] = "adaptive_flow"
    values.loc[
        selective.flag(frame, "adaptive_profile") & selective.flag(frame, "adaptive_flow")
    ] = "adaptive_both"
    values.loc[frame.candidate_id.isin(baseline_ids)] = "baseline_representative"
    return values


def diagnose(out, baseline, hybrid, grouped, truth):
    rows, distributions, ablations = [], [], []
    queues = {"BASELINE": selective.order(baseline), "FULL-HYBRID": selective.order(grouped)}
    for name, queue in queues.items():
        rows.append(
            dict(
                system=name,
                sites=len(queue),
                full=len(matches(queue, truth)),
                top1000=len(matches(queue.head(1000), truth)),
            )
        )
    assert [(r["full"], r["top1000"]) for r in rows] == [(347, 123), (483, 102)]
    pd.DataFrame(rows).to_csv(out / "reproduction.csv", index=False)
    represented = set().union(*(set(json.loads(v)) for v in baseline.member_candidate_ids))
    save(
        out / "representation_overlap.json",
        dict(
            baseline_member_IDs=len(represented),
            hybrid_candidates_already_in_baseline_members=int(
                hybrid.candidate_id.isin(represented).sum()
            ),
            additional_representations_of_baseline_members=int(
                (
                    hybrid.candidate_id.isin(represented)
                    & ~hybrid.candidate_id.isin(baseline.candidate_id)
                ).sum()
            ),
            hybrid_to_original_identity_groups_reduction=len(hybrid) - len(grouped),
            interpretation="documented representation overlap, not verified duplicate physical pipes",
        ),
    )
    base_ids = match_ids(queues["BASELINE"], truth)
    new_ids = match_ids(queues["FULL-HYBRID"], truth)
    paired = truth.drop(columns="geometry").copy()
    paired["baseline_full_match"] = paired.record_id.isin(base_ids)
    paired["hybrid_full_match"] = paired.record_id.isin(new_ids)
    paired["status"] = np.select(
        [
            paired.baseline_full_match & paired.hybrid_full_match,
            ~paired.baseline_full_match & paired.hybrid_full_match,
            paired.baseline_full_match & ~paired.hybrid_full_match,
        ],
        ["retained", "added", "lost"],
        default="unmatched_both",
    )
    paired.to_csv(out / "baseline_hybrid_record_changes.csv", index=False)
    for name, frame in [("BASELINE", baseline), ("HYBRID", hybrid), ("HYBRID-GROUPED", grouped)]:
        frame = frame.copy()
        frame["research_source"] = source(frame, baseline)
        for label, part in frame.groupby("research_source", sort=True):
            distances = cKDTree(xy(truth)).query(xy(part))[0]
            scores = part.score_RF.to_numpy(float)
            distributions.append(
                dict(
                    system=name,
                    source=label,
                    candidates=len(part),
                    score_mean=float(scores.mean()),
                    score_p10=float(np.quantile(scores, 0.1)),
                    score_p50=float(np.median(scores)),
                    score_p90=float(np.quantile(scores, 0.9)),
                    inventory_distance_p10_m=float(np.quantile(distances, 0.1)),
                    inventory_distance_p50_m=float(np.median(distances)),
                    inventory_distance_p90_m=float(np.quantile(distances, 0.9)),
                    within20m=int((distances <= 20).sum()),
                    interpretation="unmatched locations are unverified; distance is not a training label",
                )
            )
            if name == "HYBRID-GROUPED":
                without = frame.loc[~frame.research_source.eq(label)].reset_index(drop=True)
                ablations.append(
                    dict(
                        source=label,
                        sites=len(part),
                        standalone_matches=len(matches(part, truth)),
                        full_union_matches=len(new_ids),
                        leave_source_out_matches=len(matches(without, truth)),
                        marginal_capacity_matches=len(new_ids) - len(matches(without, truth)),
                    )
                )
        detail = frame[
            [
                "candidate_id",
                "road_id",
                "road_name",
                "proposal_kind",
                "research_source",
                "score_RF",
                "score_HGB",
            ]
        ].copy()
        detail["nearest_inventory_m"] = cKDTree(xy(truth)).query(xy(frame))[0]
        detail.to_csv(out / f"source_candidates_{name}.csv", index=False)
    pd.DataFrame(distributions).to_csv(out / "source_distributions.csv", index=False)
    pd.DataFrame(ablations).to_csv(out / "source_counterfactuals.csv", index=False)
    oldq, newq = queues.values()
    oldtop, newtop = oldq.head(1000), newq.head(1000)
    oldm, newm = match_ids(oldtop, truth), match_ids(newtop, truth)
    # A controlled union retains original reps, so displacement can be isolated
    # from the existing hybrid's different score-free representative selection.
    extras = grouped.loc[~grouped.candidate_id.isin(baseline.candidate_id)].copy()
    union = gpd.GeoDataFrame(pd.concat([baseline, extras], ignore_index=True), crs=32618)
    unionq = selective.order(union)
    same_ids = set(oldtop.candidate_id) & set(newtop.candidate_id)
    redundancy = []
    for name, queue in queues.items():
        for k in KS:
            part = queue.head(k)
            pairs = cKDTree(xy(part)).query_pairs(40)
            same_road = [
                (i, j)
                for i, j in pairs
                if str(part.iloc[i].road_id) == str(part.iloc[j].road_id)
                and part.iloc[i].road_part_index == part.iloc[j].road_part_index
                and abs(part.iloc[i].road_chainage_m - part.iloc[j].road_chainage_m) <= 40
            ]
            counts = part.road_name.fillna(part.road_id).value_counts()
            redundancy.append(
                dict(
                    system=name,
                    budget=k,
                    stops=len(part),
                    road_names=len(counts),
                    max_name_share=float(counts.max() / len(part)),
                    near40m_same_road_pairs=len(same_road),
                    near40m_sites=len({i for pair in same_road for i in pair}),
                    note="Nearby hypotheses may be distinct culverts; no automatic duplicate or false-positive label",
                )
            )
    pd.DataFrame(redundancy).to_csv(out / "concentration_redundancy.csv", index=False)
    save(
        out / "displacement.json",
        dict(
            full_retained=len(base_ids & new_ids),
            full_added=len(new_ids - base_ids),
            full_lost=len(base_ids - new_ids),
            top1000_retained=len(oldm & newm),
            top1000_added=len(newm - oldm),
            top1000_lost=len(oldm - newm),
            identical_representatives_in_both_top1000=len(same_ids),
            baseline_top1000_representatives_not_in_hybrid_top1000=1000 - len(same_ids),
            protected_union_full=len(matches(unionq, truth)),
            protected_union_top1000=len(matches(unionq.head(1000), truth)),
            protected_union_stops=len(unionq),
            newly_ranked_union_stops_at1000=int(
                (~unionq.head(1000).candidate_id.isin(baseline.candidate_id)).sum()
            ),
        ),
    )
    # Exact schema/preprocessing audit: the saved study transformed HGB with its
    # own contract; RF columns stored on the combined frame must not feed HGB.
    compatibility = pd.read_csv(OLD / "feature_compatibility_distribution.csv")
    compatibility.to_csv(out / "existing_model_distribution_audit.csv", index=False)
    profile = []
    for name, frame in [("baseline", baseline), ("hybrid", hybrid), ("grouped", grouped)]:
        for label, part in frame.assign(research_source=source(frame, baseline)).groupby(
            "research_source"
        ):
            profile.append(
                dict(
                    population=name,
                    source=label,
                    rows=len(part),
                    profile_available_fraction=float(
                        selective.flag(part, "road_profile_available").mean()
                    ),
                    dip_missing_fraction=float(
                        selective.numeric(part, "route_elevation_dip_60m").isna().mean()
                    ),
                    dip_zero_fraction=float(
                        selective.numeric(part, "route_elevation_dip_60m").eq(0).mean()
                    ),
                    dip_ge015_fraction=float(
                        selective.numeric(part, "route_elevation_dip_60m").ge(0.15).mean()
                    ),
                )
            )
    pd.DataFrame(profile).to_csv(out / "profile_sampling_audit.csv", index=False)
    schemas = []
    for family in ("RF", "HGB"):
        bundle = joblib.load(HYDRO / f"hydrology-sites-v2_{family}_experimental.joblib")
        assert len(bundle["feature_columns"]) == 48
        assert list(bundle["medians"].index) == bundle["feature_columns"]
        assert bundle["model"].n_features_in_ == 48
        if family == "RF":
            assert bundle["model"].n_estimators == 800
        schemas.append(
            dict(
                family=family,
                features=48,
                own_contract=True,
                own_medians=True,
                original_model_unchanged=True,
            )
        )
    save(
        out / "model_compatibility.json",
        dict(
            schema=schemas,
            current_probe_interval_m=10,
            historical_production_interval_m=75,
            original_v1_differing_representations=35,
            not_calibrated=True,
            cause_warning="Distribution differences alone do not establish causal model failure.",
        ),
    )
    print(
        "Reproduced baseline347/123 vs hybrid483/102; source and displacement diagnosis saved.",
        flush=True,
    )


def peak_reasons(group, row, column):
    chain = group.road_chainage_m.to_numpy(float)
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    p, value = float(row.road_chainage_m), float(row[column])
    window = np.abs(chain - p) <= 60 + 1e-6
    left, right = window & (chain < p - 1e-6), window & (chain > p + 1e-6)
    if p - chain.min() < 60 - 1e-6 or chain.max() - p < 60 - 1e-6:
        return "incomplete_60m_road_window"
    if not np.isfinite(value) or not np.isfinite(values[window]).all():
        return "missing_window_measurements"
    if value < values[window].max():
        return "stronger_neighbour_within60m"
    if not ((values[left] < value).any() and (values[right] < value).any()):
        return "no_strict_two_sided_peak"
    equal_ids = group.loc[window & (values == value), "candidate_id"].astype(str)
    if str(row.candidate_id) != min(equal_ids):
        return "plateau_deterministic_tie"
    return "peak_passed"


def audit_admission(out):
    audit = pd.read_csv(OLD / "all_959_record_diagnosis.csv")
    misses = audit.loc[audit.failure_substage.eq("physical_evidence_admission")]
    assert len(misses) == 458
    raw = gpd.read_parquet(OLD / "baseline_raw.parquet")
    tree = cKDTree(xy(raw))
    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet").set_index("record_id")
    details, summaries = [], []
    for (ix, iy), records in misses.groupby(["cell_x", "cell_y"]):
        full = gpd.read_parquet(OLD / "cells" / f"{ix}_{iy}" / "raw_features.parquet")
        probes = full.loc[full.proposal_kind.eq("internal_road_probe")]
        groups = {key: val for key, val in probes.groupby(["road_id", "road_part_index"])}
        for rec in records.itertuples():
            point = truth.loc[rec.record_id].geometry
            near = raw.iloc[sorted(tree.query_ball_point([point.x, point.y], 20))]
            reasons = []
            for _, row in near.iterrows():
                group = groups.get((row.road_id, row.road_part_index))
                if group is None:
                    # A neighbouring owning cell can use a different halo.
                    path = (
                        OLD
                        / "cells"
                        / f"{int(row.cell_x)}_{int(row.cell_y)}"
                        / "raw_features.parquet"
                    )
                    neighbor = gpd.read_parquet(path)
                    group = neighbor.loc[
                        neighbor.road_id.eq(row.road_id)
                        & neighbor.road_part_index.eq(row.road_part_index)
                        & neighbor.proposal_kind.eq("internal_road_probe")
                    ]
                pr = peak_reasons(group, row, "route_elevation_dip_60m")
                fr = peak_reasons(group, row, "upstream_area_m2")
                profile_value = (
                    bool(row.road_profile_available)
                    and row.route_elevation_dip_60m >= 0.15
                    and row.route_low_point_score > 0
                )
                flow_value = row.upstream_area_m2 > 0 and row.negative_tpi_9x9_m > 0
                reasons.append((profile_value, flow_value, pr, fr))
                details.append(
                    dict(
                        record_id=rec.record_id,
                        source_excel_row=rec.source_excel_row,
                        candidate_id=row.candidate_id,
                        distance_m=point.distance(row.geometry),
                        profile_available=bool(row.road_profile_available),
                        dip_m=row.route_elevation_dip_60m,
                        lowpoint=row.route_low_point_score,
                        upstream_area_m2=row.upstream_area_m2,
                        concavity_m=row.negative_tpi_9x9_m,
                        profile_value_gate=profile_value,
                        flow_value_gate=flow_value,
                        profile_peak_rule=pr,
                        flow_peak_rule=fr,
                        evidence_admitted=bool(row.evidence_admitted),
                    )
                )
            any_physical_value = any(p or f for p, f, _, _ in reasons)
            category = (
                "window_peak_selection" if any_physical_value else "no_measured_value_support"
            )
            if not any(near.road_profile_available.fillna(False)):
                category = "missing_profile_and_no_other_admitted_evidence"
            summaries.append(
                dict(
                    record_id=rec.record_id,
                    source_excel_row=rec.source_excel_row,
                    Region=rec.Region,
                    Route=rec.Route,
                    nearby_candidates=len(near),
                    primary_rule_category=category,
                    any_profile_value_gate=any(p for p, _, _, _ in reasons),
                    any_flow_value_gate=any(f for _, f, _, _ in reasons),
                    any_profile_peak_passed=any(p == "peak_passed" for _, _, p, _ in reasons),
                    any_flow_peak_passed=any(f == "peak_passed" for _, _, _, f in reasons),
                )
            )
    pd.DataFrame(details).to_csv(out / "admission_458_candidate_rules.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "admission_458_records.csv", index=False)
    save(
        out / "admission_rule_counts.json",
        pd.Series([r["primary_rule_category"] for r in summaries]).value_counts().to_dict(),
    )
    print("Audited458 admission losses against exact profile/flow/window rules.", flush=True)


def road_folds(truth, populations, out):
    roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618)
    roads["id"] = roads.id.astype(str)
    roads["research_fold"] = np.searchsorted([4625000, 4645000], roads.geometry.centroid.y)
    recovered = gpd.read_parquet(
        ROOT
        / "reports/experiments/real_inventory_heldout_roads_20261015/recovered_road_segments.parquet"
    )
    corridor_folds = {
        str(key): int(np.searchsorted([4625000, 4645000], group.geometry.union_all().centroid.y))
        for key, group in recovered.groupby("corridor")
    }
    recovered["research_fold"] = recovered.corridor.astype(str).map(corridor_folds)
    segment_corridor = dict(zip(recovered.segment_id.astype(str), recovered.corridor.astype(str)))
    roads["heldout_corridor"] = roads.id.map(segment_corridor).fillna("")
    use = roads.heldout_corridor.ne("")
    roads.loc[use, "research_fold"] = roads.loc[use, "heldout_corridor"].map(corridor_folds)
    lookup = roads.set_index("id").research_fold.to_dict()
    for frame in populations.values():
        frame["research_fold"] = frame.road_id.astype(str).map(lookup)
        assert frame.research_fold.notna().all()
    (_, right), _ = roads.sindex.nearest(truth.geometry, return_all=False, return_distance=True)
    truth["research_fold"] = roads.iloc[right].research_fold.to_numpy()
    truth.drop(columns="geometry").to_csv(out / "inventory_fold_assignment.csv", index=False)
    roads[["id", "heldout_corridor", "research_fold", "geometry"]].to_parquet(
        out / "road_fold_assignment.parquet", index=False
    )
    save(out / "corridor_fold_assignment.json", corridor_folds)
    return roads


def evaluate(
    name,
    queue,
    truth,
    baseline_ids,
    candidates,
    out,
    metrics,
    route_rows,
    record_rows,
    scope="global_retrospective",
    **meta,
):
    queue = queue.reset_index(drop=True)
    # Geometry/rank queues refer to archived source populations. Do not repeat
    # hundreds of raw feature columns and large evidence JSON for every ordering.
    keep = [
        c
        for c in queue
        if c
        in {
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
    ]
    queue[keep].to_parquet(out / f"queue_{name}.parquet", index=False)
    for tolerance in TOLERANCES:
        assigned = matches(queue, truth, tolerance)
        ids = set(truth.iloc[assigned.real_row].record_id)
        nd, nearest_row = nearest(queue, truth)
        metrics.append(
            dict(
                system=name,
                scope=scope,
                **meta,
                budget="all",
                tolerance_m=tolerance,
                candidates=candidates,
                sites=len(queue),
                stops=len(queue),
                inventory=len(truth),
                matched=len(assigned),
                recall=len(assigned) / len(truth),
                covered_any=int((nd <= tolerance).sum()),
                added_vs_baseline20=len(ids - baseline_ids) if tolerance == 20 else None,
                lost_vs_baseline20=len(baseline_ids - ids) if tolerance == 20 else None,
                matched_distance_median_m=float(assigned.distance_m.median())
                if len(assigned)
                else None,
                independent=False,
            )
        )
        if tolerance == 20:
            for (region, route), part in truth.groupby(["Region", "Route"]):
                route_rows.append(
                    dict(
                        system=name,
                        scope=scope,
                        budget="all",
                        Region=region,
                        Route=route,
                        records=len(part),
                        matched=int(part.record_id.isin(ids).sum()),
                    )
                )
            for i, row in truth.iterrows():
                record_rows.append(
                    dict(
                        system=name,
                        record_id=row.record_id,
                        source_excel_row=row.source_excel_row,
                        Region=row.Region,
                        Route=row.Route,
                        matched20=row.record_id in ids,
                        baseline_matched20=row.record_id in baseline_ids,
                        nearest_site_m=nd[i],
                        nearest_site_id=queue.iloc[nearest_row[i]].candidate_id
                        if len(queue)
                        else None,
                        independent=False,
                    )
                )
    base_queue = selective.order(gpd.read_parquet(OLD / "variant_BASELINE.parquet"))
    for k in KS:
        part = queue.head(k)
        assigned = matches(part, truth)
        ids = set(truth.iloc[assigned.real_row].record_id)
        budget_baseline = match_ids(base_queue.head(k), truth)
        metrics.append(
            dict(
                system=name,
                scope=scope,
                **meta,
                budget=k,
                tolerance_m=20,
                candidates=candidates,
                sites=len(queue),
                stops=len(part),
                inventory=len(truth),
                matched=len(assigned),
                recall=len(assigned) / len(truth),
                covered_any=int((nearest(part, truth)[0] <= 20).sum()),
                added_vs_baseline20=len(ids - budget_baseline),
                lost_vs_baseline20=len(budget_baseline - ids),
                matched_distance_median_m=float(assigned.distance_m.median())
                if len(assigned)
                else None,
                independent=False,
            )
        )
        for (region, route), sub in truth.groupby(["Region", "Route"]):
            route_rows.append(
                dict(
                    system=name,
                    scope=scope,
                    budget=k,
                    Region=region,
                    Route=route,
                    records=len(sub),
                    matched=int(sub.record_id.isin(ids).sum()),
                )
            )


def training_sites(out):
    manifest = pd.read_csv(
        ROOT / "reports/experiments/site_challenger_20261015/final_training_rows.csv"
    )
    measured = gpd.read_parquet(HYDRO / "physical_measurements_v2.parquet")
    measured = measured.loc[measured.proposal_kind.eq("training_measurement_only")].merge(
        manifest, on="candidate_id", validate="one_to_one"
    )
    chosen = []
    for _, group in measured.groupby("site_key", sort=True):
        group = group.sort_values("candidate_id", kind="stable")
        assert group.target.nunique() == 1
        coords = xy(group)
        costs = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).sum(axis=1)
        chosen.append(
            group.sort_values("candidate_id", kind="stable").index[0]
            if np.all(costs == costs[0])
            else group.index[np.argmin(costs)]
        )
    sites = measured.loc[chosen].sort_values("site_key", kind="stable").reset_index(drop=True)
    # Contradictory supervised locations are not resolved by relabeling.
    positives = sites.loc[sites.target.eq(1)]
    negative = sites.target.eq(0)
    conflicts = np.zeros(len(sites), bool)
    conflicts[negative] = cKDTree(xy(positives)).query(xy(sites.loc[negative]))[0] <= 20
    sites["excluded_conflicting_label_within20m"] = conflicts
    sites.drop(columns="geometry").to_csv(out / "supervised_site_manifest.csv", index=False)
    save(
        out / "supervised_label_audit.json",
        dict(
            source_positive_rows=int(manifest.target.eq(1).sum()),
            source_verified_negative_rows=int(manifest.target.eq(0).sum()),
            positive_source_sites=int(sites.target.eq(1).sum()),
            verified_negative_sites=int(sites.target.eq(0).sum()),
            spatial_label_conflicts=int(conflicts.sum()),
            unvisited_negative_rows=0,
            summer_rows_added_to_training=0,
        ),
    )
    return sites.loc[~conflicts].reset_index(drop=True)


def site_models(out, roads, populations, reuse=False):
    sites = training_sites(out)
    audits = []
    for fold in range(3):
        heldout = roads.loc[roads.research_fold.eq(fold)]
        # All sibling records are represented by one source site; guard medoid
        # approximation by excluding source sites if ANY original sibling is near.
        original = gpd.read_parquet(HYDRO / "physical_measurements_v2.parquet")
        labels = pd.read_csv(
            ROOT / "reports/experiments/site_challenger_20261015/final_training_rows.csv"
        )
        original = original.loc[original.proposal_kind.eq("training_measurement_only")].merge(
            labels, on="candidate_id"
        )
        heldout_corridors = set(heldout.heldout_corridor) - {""}
        keep, distance, _ = site_ranker.training_exclusion(sites, original, heldout)
        train = sites.loc[keep].copy()
        row = dict(
            fold=fold,
            train_sites=len(train),
            positives=int(train.target.eq(1).sum()),
            negatives=int(train.target.eq(0).sum()),
            excluded_sites=int((~keep).sum()),
            minimum_training_to_heldout_road_m=float(distance[keep].min()) if keep.any() else None,
            heldout_corridors=sorted(heldout_corridors),
            independent=False,
            purge_m=1000,
        )
        try:
            model_path = out / f"experimental_site_ranker_fold{fold}.joblib"
            if reuse:
                previous = pd.read_csv(out / f"training_fold{fold}.csv")
                assert previous.candidate_id.tolist() == train.candidate_id.tolist()
                assert previous.target.tolist() == train.target.tolist()
                bundle = joblib.load(model_path)
            else:
                with threadpool_limits(limits=2):
                    bundle = site_ranker.train(train)
                joblib.dump(bundle, model_path)
            row["features"] = bundle["features"]
            row["available"] = True
            train.drop(columns="geometry").to_csv(out / f"training_fold{fold}.csv", index=False)
            for frame in populations.values():
                use = frame.research_fold.eq(fold)
                frame.loc[use, "score_site_oof"] = site_ranker.score(bundle, frame.loc[use])
        except ValueError as error:
            row["available"] = False
            row["limitation"] = str(error)
        audits.append(row)
    save(out / "model_fold_audit.json", audits)
    if reuse:
        bundle = joblib.load(out / "experimental_site_ranker_descriptive.joblib")
    else:
        bundle = site_ranker.train(sites)
        joblib.dump(bundle, out / "experimental_site_ranker_descriptive.joblib")
    for frame in populations.values():
        frame["score_site_descriptive"] = site_ranker.score(bundle, frame)
    save(
        out / "model_training_summary.json",
        dict(
            fits=1 + sum(row["available"] for row in audits),
            fits_this_execution=0 if reuse else 1 + sum(row["available"] for row in audits),
            reused_unchanged_site_models=reuse,
            frozen_model_fits=0,
            crossfit_available=all(row["available"] for row in audits),
            model="fixed logistic C=1 balanced, physical features only",
            independent=False,
            production_changed=False,
            label_source_leakage_features_excluded=True,
            features=bundle["features"],
        ),
    )
    print(
        "Saved separate physical-site model and three spatial/whole-road-purged fold models.",
        flush=True,
    )


def nested_selection(out, populations, truth):
    options, fold_rows, selected_rows = [], [], []
    # All six grouped admission/physical-order combinations are predetermined.
    for rule in selective.RULES:
        frame = populations[f"SELECTIVE-GROUPED-{rule}"]
        for weight in WEIGHTS:
            for penalty in PENALTIES:
                counts = []
                for fold in range(3):
                    part = frame.loc[frame.research_fold.eq(fold)]
                    queue = selective.order(part, weight=weight, penalty=penalty)
                    actual = truth.loc[truth.research_fold.eq(fold)].reset_index(drop=True)
                    count = len(matches(queue.head(1000), actual))
                    counts.append(count)
                    fold_rows.append(
                        dict(
                            rule=rule,
                            weight=weight,
                            penalty=penalty,
                            fold=fold,
                            requested_budget=1000,
                            actual_stops=min(1000, len(queue)),
                            sites=len(queue),
                            inventory=len(actual),
                            matched=count,
                            independent=False,
                        )
                    )
                options.append((rule, weight, penalty, counts, len(frame)))
    parts = []
    for outer in range(3):
        other = [f for f in range(3) if f != outer]
        best = sorted(
            options,
            key=lambda v: (
                -sum(v[3][f] for f in other),
                v[4],
                selective.RULES.index(v[0]),
                WEIGHTS.index(v[1]),
                PENALTIES.index(v[2]),
            ),
        )[0]
        rule, weight, penalty, counts, _ = best
        frame = populations[f"SELECTIVE-GROUPED-{rule}"]
        part = selective.order(
            frame.loc[frame.research_fold.eq(outer)], weight=weight, penalty=penalty
        )
        part["chosen_rule"], part["chosen_weight"], part["chosen_penalty"] = rule, weight, penalty
        # Store comparable [0,1.25] priorities, not per-fold ranks, when pooling.
        parts.append(part)
        selected_rows.append(
            dict(
                outer_fold=outer,
                selected_rule=rule,
                selected_weight=weight,
                selected_penalty=penalty,
                inner_folds=other,
                inner_total_stops=2000,
                inner_matches=sum(counts[f] for f in other),
                outer_matches_at1000=counts[outer],
                outer_inventory=int(truth.research_fold.eq(outer).sum()),
                independent=False,
            )
        )
    pd.DataFrame(fold_rows).to_csv(out / "development_grid.csv", index=False)
    save(out / "nested_selection.json", selected_rows)
    pooled = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=32618)
    pooled = pooled.sort_values(
        ["research_priority", "candidate_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    best = sorted(
        options,
        key=lambda v: (
            -sum(v[3]),
            v[4],
            selective.RULES.index(v[0]),
            WEIGHTS.index(v[1]),
            PENALTIES.index(v[2]),
        ),
    )[0]
    save(
        out / "all_development_choice.json",
        dict(
            rule=best[0],
            weight=best[1],
            penalty=best[2],
            selection="all three exposed development folds; descriptive, not test",
            fold_top1000_matches=best[3],
        ),
    )
    return pooled, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Replay changed grouping/reporting with unchanged experimental models; verify original training-row membership",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    assert out != OLD and (out / "METHOD.md").exists() and not (out / "COMPLETE.json").exists()
    assert str(out).startswith(str(ROOT / "reports/experiments") + "/")
    before = protected()
    sealed = json.loads((OLD / "COMPLETE.json").read_text())["output_hashes"]
    assert all(sha256_file(OLD / p) == digest for p, digest in sealed.items())
    guards = json.loads((OLD / "baseline_input_hashes.json").read_text()) | json.loads(
        (OLD / "experiment_input_hashes.json").read_text()
    )
    assert all(sha256_file(Path(p)) == digest for p, digest in guards.items())
    save(
        out / "input_hashes.json",
        guards
        | {str(OLD / p): digest for p, digest in sealed.items()}
        | {str(out / "METHOD.md"): sha256_file(out / "METHOD.md")},
    )
    audit = pd.read_csv(OLD / "all_959_record_diagnosis.csv").set_index("record_id")
    truth = gpd.read_parquet(SUMMER / "inventory_coverage.parquet")
    truth = (
        truth.loc[truth.record_id.map(audit.query_success)]
        .sort_values("source_excel_row", kind="stable")
        .reset_index(drop=True)
    )
    baseline = gpd.read_parquet(OLD / "variant_BASELINE.parquet")
    hybrid = gpd.read_parquet(OLD / "variant_HYBRID.parquet")
    grouped = gpd.read_parquet(OLD / "variant_HYBRID-GROUPED.parquet")
    diagnose(out, baseline, hybrid, grouped, truth)
    audit_admission(out)
    if args.diagnose_only:
        return
    populations = {"BASELINE": baseline, "FULL-HYBRID": grouped}
    for rule in selective.RULES:
        frame = selective.population(baseline, hybrid, rule)
        populations[f"SELECTIVE-{rule}"] = frame
        populations[f"SELECTIVE-GROUPED-{rule}"] = selective.anchored_groups(frame)
        print(
            f"{rule}: {len(frame)} hypotheses -> {len(populations[f'SELECTIVE-GROUPED-{rule}'])} anchored sites",
            flush=True,
        )
    roads = road_folds(truth, populations, out)
    site_models(out, roads, populations, reuse=args.reuse_models)
    nested, best = nested_selection(out, populations, truth)
    metrics, route_rows, record_rows = [], [], []
    baseline_ids = match_ids(selective.order(baseline), truth)
    for name, frame in populations.items():
        for family in ("RF", "HGB"):
            evaluate(
                f"{name}-{family}",
                selective.order(frame, score=f"score_{family}"),
                truth,
                baseline_ids,
                len(frame),
                out,
                metrics,
                route_rows,
                record_rows,
            )
    rule, weight, penalty, _, _ = best
    chosen = populations[f"SELECTIVE-GROUPED-{rule}"]
    for name, queue in [
        ("BASELINE-FIRST", selective.order(chosen, baseline_first=True)),
        ("PHYSICAL-RERANK", selective.order(chosen, weight=weight, penalty=penalty)),
        ("SITE-MODEL-DESCRIPTIVE", selective.order(chosen, score="score_site_descriptive")),
        ("NESTED-SELECTIVE", nested),
        ("BASELINE-DIVERSITY", selective.order(baseline, penalty=0.1)),
        ("HYBRID-DIVERSITY", selective.order(grouped, penalty=0.1)),
    ]:
        evaluate(
            name,
            queue,
            truth,
            baseline_ids,
            len(chosen) if name not in ("BASELINE-DIVERSITY", "HYBRID-DIVERSITY") else len(queue),
            out,
            metrics,
            route_rows,
            record_rows,
            scope="crossfit_retrospective"
            if name in ("SITE-MODEL-CROSSFIT", "NESTED-SELECTIVE")
            else "global_retrospective",
        )
    if "score_site_oof" in chosen and chosen.score_site_oof.notna().all():
        evaluate(
            "SITE-MODEL-CROSSFIT",
            selective.order(chosen, score="score_site_oof"),
            truth,
            baseline_ids,
            len(chosen),
            out,
            metrics,
            route_rows,
            record_rows,
            scope="crossfit_retrospective",
        )
    # Fixed rules per complete-road geographic fold: include baseline RF and HGB,
    # the physical reranker and a ranker trained with held-out roads purged.
    fold_metrics = []
    for fold in range(3):
        actual = truth.loc[truth.research_fold.eq(fold)].reset_index(drop=True)
        for name, frame, score, w, p in [
            ("BASELINE-RF", baseline, "score_RF", 0.0, 0.0),
            ("BASELINE-HGB", baseline, "score_HGB", 0.0, 0.0),
            ("FULL-HYBRID-RF", grouped, "score_RF", 0.0, 0.0),
            ("SELECTIVE-RF", chosen, "score_RF", 0.0, 0.0),
            ("PHYSICAL-RERANK", chosen, "score_RF", weight, penalty),
            ("SITE-MODEL", chosen, "score_site_oof", 0.0, 0.0),
        ]:
            if score not in frame or frame.loc[frame.research_fold.eq(fold), score].isna().any():
                continue
            queue = selective.order(
                frame.loc[frame.research_fold.eq(fold)], score=score, weight=w, penalty=p
            )
            for k in (*KS, "all"):
                part = queue if k == "all" else queue.head(k)
                fold_metrics.append(
                    dict(
                        system=name,
                        fold=fold,
                        budget=k,
                        stops=len(part),
                        inventory=len(actual),
                        matched=len(matches(part, actual)),
                        independent=False,
                        training_purged=name == "SITE-MODEL",
                    )
                )
    pd.DataFrame(metrics).to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "region_route_results.csv", index=False)
    pd.DataFrame(record_rows).to_csv(out / "record_results.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "road_spatial_fold_results.csv", index=False)
    assert protected() == before
    assert all(sha256_file(OLD / p) == digest for p, digest in sealed.items())
    assert all(sha256_file(Path(p)) == digest for p, digest in guards.items())
    fits = json.loads((out / "model_training_summary.json").read_text())["fits"]
    save(
        out / "COMPUTATIONS_COMPLETE.json",
        dict(
            fits=fits,
            frozen_model_fits=0,
            metric_rows=len(metrics),
            independently_rechecked_metric_rows=len(metrics),
            old_sealed_outputs_unchanged=len(sealed),
            guarded_inputs_unchanged=len(guards),
            protected_artifacts_unchanged=len(before),
            production_changed=False,
            deployed=False,
            independent=False,
            default_changed=False,
        ),
    )
    print(
        pd.DataFrame(metrics)
        .query("budget==1000")[["system", "matched", "stops"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
