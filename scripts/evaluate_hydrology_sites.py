"""Fixed-spec, versioned candidate/feature/site ablations. No production writes."""

# ruff: noqa: E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.flow_c_bundle import load_frozen_flow_c_bundle, sha256_file
from culvert_ai.inference import RouteZConfig, dynamic_route_z
from scripts.validate_inventory_heldout_roads import (
    evaluate,
    load_module,
    match_one_to_one,
    fold_for,
)
from staging.ranking_policy import rank_candidates
from staging.hydrology_sites import (
    THRESHOLDS,
    VERSION,
    apply_feature_contract,
    consolidate_events,
    detect_events,
    fit_feature_contract,
    physical_measurements,
    search_candidates,
)

PRIOR = ROOT / "reports/experiments/real_inventory_heldout_roads_20261015"
HYDRO = ROOT / "data/processed/hydrology/usgs_3dep_10m"
DEFAULT_OUT = ROOT / "reports/experiments/hydrology_driven_sites_20261015"


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def protected():
    before = json.loads((PRIOR / "protected_before.json").read_text())
    assert len(before) == 291
    assert all(sha256_file(ROOT / p) == digest for p, digest in before.items())
    return before


def training_sources():
    manifests = []
    for fold in range(3):
        for model in ("RF", "HGB"):
            manifests.append(pd.read_csv(PRIOR / f"training_manifest_{model}_fold{fold}.csv"))
    manifests.append(pd.read_csv(ROOT / "models/flow_c_frozen_v1.joblib.training_records.csv"))
    manifests.append(
        pd.read_csv(ROOT / "reports/experiments/site_challenger_20261015/final_training_rows.csv")
    )
    return set().union(*(set(frame.candidate_id) for frame in manifests))


def prepare(out, columns):
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    roads["route_id"] = roads.corridor
    roads["id"] = roads.segment_id
    all_roads = gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618)
    streams = gpd.read_file(ROOT / "data/raw/streams.gpkg").to_crs(32618)
    networks = {key: gpd.read_parquet(HYDRO / f"drainage_{key}.parquet") for key in THRESHOLDS}
    print("Generating transverse mapped and DEM drainage-road intersections", flush=True)
    events, rejected = detect_events(roads, streams, networks)
    events["fold"] = events.route_id.map(fold_for)
    events.to_parquet(out / "crossing_events.parquet", index=False)
    rejected.to_csv(out / "rejected_contacts.csv", index=False)
    probes = search_candidates(roads)
    probes["fold"] = probes.route_id.map(fold_for)
    probes.to_parquet(out / "dense_search_probes.parquet", index=False)
    control = gpd.read_parquet(PRIOR / "current75_features.parquet")
    control["route_id"] = control.corridor
    control["source"] = "route_interval_sample"
    control["proposal_kind"] = "control75"
    native = gpd.read_file(ROOT / "data/processed/actual_ulster_unlabeled_features.gpkg")
    train = native.loc[native.candidate_id.isin(training_sources())].copy()
    route_lookup = {}
    for fold in range(3):
        for model in ("RF", "HGB"):
            rows = pd.read_csv(PRIOR / f"training_manifest_{model}_fold{fold}.csv")
            route_lookup.update(dict(zip(rows.candidate_id, rows.corridor)))
    train["route_id"] = (
        train.candidate_id.map(route_lookup).fillna(train.matched_route).fillna("<unassigned>")
    )
    train["proposal_kind"] = "training_measurement_only"
    # Keep only geometric generation metadata. No labels or inspection attributes
    # can influence the builder, its normalizers or proposal generation.
    metadata = [
        "candidate_id",
        "road_id",
        "road_name",
        "route_id",
        "source",
        "proposal_kind",
        "route_sample_distance_m",
        "route_part_index",
        "geometry",
    ]
    tables = []
    for table in (probes, events, control, train):
        tables.append(table.reindex(columns=metadata))
    points = gpd.GeoDataFrame(pd.concat(tables, ignore_index=True), geometry="geometry", crs=32618)
    points["road_id"] = points.road_id.astype(str)
    assert points.candidate_id.is_unique
    print(
        f"Measuring {len(points)} positions: {len(probes)} probes, {len(events)} physical crossing proposals, {len(train)} training locations",
        flush=True,
    )
    measured = physical_measurements(points, all_roads, streams, HYDRO)
    measured.to_parquet(out / "physical_measurements_v2.parquet", index=False)
    return measured, probes, events, control, roads


def numeric_features(frame, columns, medians):
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )


def score(fitted, frame, columns):
    with threadpool_limits(limits=2):
        return fitted["model"].predict_proba(numeric_features(frame, columns, fitted["medians"]))[
            :, 1
        ]


def fit_copy(model, training, contract, frozen_rf, hgb_config, helper):
    columns = contract["feature_columns"]
    if model == "RF":
        medians = training[columns].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
        fitted = dict(model=clone(frozen_rf["model"]), medians=medians, family="rf")
        fitted["model"].fit(numeric_features(training, columns, medians), training.target)
    else:
        with threadpool_limits(limits=2):
            fitted = helper.fit_model(hgb_config, training, columns)
    fitted["contract"] = contract
    fitted["feature_columns"] = columns
    fitted["experimental"] = True
    return fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output
    if (out / "COMPLETE.json").exists():
        raise FileExistsError("Completed experiment is immutable.")
    out.mkdir(parents=True, exist_ok=args.resume)
    before = protected()
    save_json(out / "protected_before.json", before)
    frozen = load_frozen_flow_c_bundle(ROOT / "models/flow_c_frozen_v1.joblib")
    hgb_bundle = joblib.load(
        ROOT / "reports/experiments/site_challenger_20261015/challenger.joblib"
    )
    helper = load_module("hydro_v2_fixed_hgb", ROOT / "scripts/train_site_challenger.py")
    columns = frozen["feature_columns"]
    save_json(
        out / "preregistered_method.json",
        dict(
            version=VERSION,
            unchanged_protected_files=291,
            production_changed=False,
            folds=json.loads((PRIOR / "preregistered_method.json").read_text())["route_folds"],
            training_ids="Exact prior per-fold manifests and labels; no new negative selection or model search",
            variants=[
                "production75_v1",
                "repaired75_v2",
                "dense10_v2",
                "hybrid_all_v2",
                "event_queue_v2",
                "physical_sites_v2",
            ],
            search_interval_m=10,
            event_gate="actual intersection, acute angle>=30deg and bilateral water geometry within60m, 2m side tolerance",
            consolidation="same road+part+fine drainage channel, <=20m Euclidean AND chainage, bounded representative clustering, distinct mapped waters preserved",
            representative="mapped intersection first, otherwise actual DEM intersection; deterministic candidate ID, never moved",
            ranking="raw RF, dynamic ROUTE-Z, raw fixed HGB evaluated separately; no arbitrary boosts or road quotas",
            new_model_inputs=False,
            feature_columns=columns,
            v2_contract="same48 names but corrected10m profiles and real edge flags, training-only DEM composite references; retrain experimental copies consistently",
            proposal_sensitivity_angles=[15, 30, 45],
            matching_tolerance_m=[10, 20, 25],
            independent_test=False,
            label_reuse="All714 previously exposed; retrospective comparison only",
            warning="Recommendations require field verification.",
        ),
    )
    measured_path = out / "physical_measurements_v2.parquet"
    if args.resume and measured_path.exists():
        measured = gpd.read_parquet(measured_path)
        probes = gpd.read_parquet(out / "dense_search_probes.parquet")
        events = gpd.read_parquet(out / "crossing_events.parquet")
        control = gpd.read_parquet(PRIOR / "current75_features.parquet")
        roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    else:
        measured, probes, events, control, roads = prepare(out, columns)
    context = measured.loc[measured.proposal_kind.eq("internal_road_probe")]
    indexed = measured.set_index("candidate_id", drop=False)
    inventory = pd.read_csv(PRIOR / "inventory_split_manifest.csv")
    real = gpd.GeoDataFrame(
        inventory, geometry=gpd.points_from_xy(inventory.source_x, inventory.source_y), crs=26918
    ).to_crs(32618)
    baseline = (
        pd.read_csv(PRIOR / "heldout_group_metrics.csv").query("cohort == 'current75'").copy()
    )
    baseline["cohort"] = "production75_v1"
    metric_rows = baseline.to_dict("records")
    missingness, spacing, per_road, assignments = [], [], [], []
    for fold in range(3):
        truth = real.loc[real.fold.eq(fold)].sort_values("record_key").reset_index(drop=True)
        reference = RouteZConfig()  # Tiny/constant routes remain neutral, raw tie-break.
        for model in ("RF", "HGB"):
            manifest = pd.read_csv(PRIOR / f"training_manifest_{model}_fold{fold}.csv")
            train_raw = indexed.loc[manifest.candidate_id].reset_index(drop=True)
            contract = fit_feature_contract(train_raw, columns)
            transformed = apply_feature_contract(measured, contract, normalization_context=context)
            transformed_index = transformed.set_index("candidate_id", drop=False)
            training = transformed_index.loc[manifest.candidate_id].reset_index(drop=True)
            for column in ("target", "route", "site_key", "weight"):
                training[column] = manifest["corridor" if column == "route" else column].to_numpy()
            path = out / f"{VERSION}_{model}_fold{fold}.joblib"
            if args.resume and path.exists():
                fitted = joblib.load(path)
            else:
                print(
                    f"Fitting fixed experimental {model} fold{fold} with v2 contract, {len(training)} unchanged training rows",
                    flush=True,
                )
                fitted = fit_copy(
                    model, training, contract, frozen, hgb_bundle["configuration"], helper
                )
                joblib.dump(fitted, path, compress=3)
            save_json(out / f"feature_contract_{model}_fold{fold}.json", contract)
            held = set(roads.loc[roads.fold.eq(fold), "corridor"])
            use = transformed.route_id.isin(held) & ~transformed.proposal_kind.eq(
                "training_measurement_only"
            )
            tests = transformed.loc[use].copy().reset_index(drop=True)
            tests["model_score"] = score(fitted, tests, columns)
            tests[
                ["candidate_id", "road_id", "route_id", "proposal_kind", "model_score", "geometry"]
            ].to_parquet(out / f"scores_{model}_fold{fold}.parquet", index=False)
            for name, raw in (
                ("repaired75_v2", tests.loc[tests.proposal_kind.eq("control75")]),
                ("dense10_v2", tests.loc[tests.proposal_kind.eq("internal_road_probe")]),
                ("hybrid_all_v2", tests.loc[~tests.proposal_kind.eq("control75")]),
                (
                    "event_queue_v2",
                    tests.loc[
                        tests.proposal_kind.isin(
                            ["mapped_crossing", "dem_crossing", "terrain_convergence"]
                        )
                    ],
                ),
            ):
                raw = raw.copy().reset_index(drop=True)
                if raw.empty:
                    continue
                policies = ["raw_rf", "dynamic_route_z"] if model == "RF" else ["raw_rf"]
                for policy in policies:
                    if name == "event_queue_v2":
                        joined = (
                            events.set_index("candidate_id", drop=False)
                            .loc[raw.candidate_id]
                            .reset_index(drop=True)
                        )
                        joined["model_score"] = raw.model_score.to_numpy()
                        if policy == "dynamic_route_z":
                            joined["priority"] = dynamic_route_z(
                                raw.rename(columns={"model_score": "flow_c_score"}),
                                config=reference,
                            ).route_z_score
                        else:
                            joined["priority"] = joined.model_score
                        queue = joined.sort_values(
                            ["priority", "model_score", "candidate_id"],
                            ascending=[False, False, True],
                            kind="stable",
                        ).reset_index(drop=True)
                        queue["global_rank"] = np.arange(1, len(queue) + 1)
                        variants = [(name, queue)]
                        physical = consolidate_events(joined, raw.model_score.to_numpy())
                        physical = physical.sort_values(
                            ["priority", "model_score", "candidate_id"],
                            ascending=[False, False, True],
                            kind="stable",
                        ).reset_index(drop=True)
                        physical["global_rank"] = np.arange(1, len(physical) + 1)
                        variants.append(("physical_sites_v2", physical))
                    else:
                        scored = raw.rename(columns={"model_score": "flow_c_score"})
                        queue = rank_candidates(
                            scored, policy=policy, route_z_config=reference
                        ).rename(columns={"flow_c_score": "model_score"})
                        variants = [(name, queue)]
                    for variant, sites in variants:
                        candidates = raw.rename(columns={"route_id": "corridor"})
                        sites = sites.rename(columns={"route_id": "corridor"})
                        ctx = dict(cohort=variant, fold=fold, model=model, policy=policy)
                        metric_rows.extend(evaluate(sites, candidates, truth, ctx))
                        sites.to_parquet(
                            out / f"sites_{variant}_{model}_{policy}_fold{fold}.parquet",
                            index=False,
                        )
                        for route in sorted(held):
                            per_road.extend(
                                evaluate(
                                    sites.loc[sites.corridor.eq(route)],
                                    candidates.loc[candidates.corridor.eq(route)],
                                    truth.loc[truth.corridor.eq(route)],
                                    dict(**ctx, corridor=route),
                                )
                            )
                        if len(sites) > 1:
                            from scipy.spatial import cKDTree

                            distances = cKDTree(np.c_[sites.geometry.x, sites.geometry.y]).query(
                                np.c_[sites.geometry.x, sites.geometry.y], k=2
                            )[0][:, 1]
                            spacing.append(
                                dict(
                                    **ctx,
                                    sites=len(sites),
                                    q10=float(np.quantile(distances, 0.1)),
                                    median=float(np.median(distances)),
                                    q90=float(np.quantile(distances, 0.9)),
                                    within_1m_of_75m=float((abs(distances - 75) < 1).mean()),
                                )
                            )
                        for radius in (10, 20, 25):
                            hit = match_one_to_one(
                                sites.head(250),
                                np.c_[truth.geometry.x, truth.geometry.y],
                                radius=radius,
                            )
                            assignments.append(
                                dict(
                                    **ctx,
                                    radius_m=radius,
                                    k=250,
                                    denominator=min(250, len(sites)),
                                    matched=len(hit),
                                )
                            )
                for column in columns:
                    values = pd.to_numeric(raw[column], errors="coerce").replace(
                        [np.inf, -np.inf], np.nan
                    )
                    missingness.append(
                        dict(
                            fold=fold,
                            model=model,
                            cohort=name,
                            feature=column,
                            rows=len(raw),
                            missing=int(values.isna().sum()),
                            valid_zero=int(values.eq(0).sum()),
                        )
                    )
        print(f"Completed controlled fold {fold}", flush=True)
    pd.DataFrame(metric_rows).to_csv(out / "heldout_group_metrics.csv", index=False)
    pd.DataFrame(per_road).to_csv(out / "heldout_road_metrics.csv", index=False)
    pd.DataFrame(missingness).to_csv(out / "feature_missingness.csv", index=False)
    pd.DataFrame(spacing).to_csv(out / "recommendation_spacing.csv", index=False)
    pd.DataFrame(assignments).to_csv(out / "matching_tolerance_sensitivity.csv", index=False)
    # Final *experimental* artifacts use the preserved final manifests, not a
    # model selected/tuned on the held-out results. Never replace models/ files.
    for model in ("RF", "HGB"):
        source = ROOT / (
            "models/flow_c_frozen_v1.joblib.training_records.csv"
            if model == "RF"
            else "reports/experiments/site_challenger_20261015/final_training_rows.csv"
        )
        manifest = pd.read_csv(source)
        train_raw = indexed.loc[manifest.candidate_id].reset_index(drop=True)
        contract = fit_feature_contract(train_raw, columns)
        transformed = apply_feature_contract(measured, contract, normalization_context=context)
        training = (
            transformed.set_index("candidate_id", drop=False)
            .loc[manifest.candidate_id]
            .reset_index(drop=True)
        )
        training["target"] = (
            manifest.experiment_target.to_numpy() if model == "RF" else manifest.target.to_numpy()
        )
        if model == "HGB":
            for column in ("route", "site_key", "weight"):
                training[column] = manifest[column].to_numpy()
        fitted = fit_copy(model, training, contract, frozen, hgb_bundle["configuration"], helper)
        fitted["model_version"] = f"{VERSION}-{model}-experimental"
        joblib.dump(fitted, out / f"{VERSION}_{model}_experimental.joblib", compress=3)
    assert before == protected()
    save_json(
        out / "input_integrity.json",
        dict(files=291, unchanged=True, before=before, after=protected()),
    )
    for path in (ROOT / "staging/hydrology_sites.py", Path(__file__)):
        shutil.copyfile(path, out / (path.name + ".txt"))
    save_json(
        out / "COMPLETE.json",
        dict(
            experimental=True,
            fit_count=8,
            production_changed=False,
            independent_test=False,
            version=VERSION,
        ),
    )
    print(
        "Completed hydrology candidate/feature evaluation; production and all291 files unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    main()
