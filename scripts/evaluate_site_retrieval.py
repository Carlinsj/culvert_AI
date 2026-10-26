"""Fixed grouping/sampling audit. No model fitting, inference or production writes."""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hydrology_sites import protected, save_json
from scripts.validate_density_controls import (
    HYBRID,
    PREVIOUS,
    PRIOR,
    inventory,
    uniform,
    xy,
    adjacency,
    prefix_matches,
    conditional_order,
    stream_seed,
    summary,
    provenance,
)
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.hybrid_sites import cluster_sites
from staging.feature_sites import ranked
from staging.retrieval_sites import identity_groups, ranked_identity, RADII

OUT = ROOT / "reports/experiments/site_retrieval_20261016"
PREVIOUS_FEATURE = ROOT / "reports/experiments/feature_localization_20261015"
TOLERANCES = (5, 10, 15, 20, 25, 50)
KS = (100, 250, 500, 1000, 2000)


def spatial_alternative(points, radius, method):
    """Unsafe global diagnostic: geodesic NMS or DBSCAN(min_samples=1)."""
    frame = points.sort_values("candidate_id").reset_index(drop=True)
    coords = xy(frame)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius * 1.01, output_type="ndarray")
    ll = frame.to_crs(4326)
    if len(pairs):
        a, b = pairs.T
        _, _, distances = Geod(ellps="WGS84").inv(
            ll.geometry.x.to_numpy()[a],
            ll.geometry.y.to_numpy()[a],
            ll.geometry.x.to_numpy()[b],
            ll.geometry.y.to_numpy()[b],
        )
        pairs = pairs[np.asarray(distances) <= radius]
    if method == "geodesic_dbscan_min1":
        a, b = pairs.T if len(pairs) else ([], [])
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(frame), len(frame)))
        _, labels = connected_components(graph, directed=False)
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    else:
        neighbours = {i: [] for i in range(len(frame))}
        for a, b in pairs:
            neighbours[a].append(b)
            neighbours[b].append(a)
        used = set()
        groups = []
        for i in range(len(frame)):
            if i in used:
                continue
            members = [i] + [j for j in neighbours[i] if j not in used]
            groups.append(np.asarray(members))
            used.update(members)
    cross = sum(frame.iloc[g].road_id.astype(str).nunique() > 1 for g in groups)
    chain = sum(np.ptp(coords[g], axis=0).max() > 2 * radius for g in groups)
    result = frame.iloc[[g[0] for g in groups]].copy()
    return result, dict(cross_road_groups=cross, groups_spanning_over_2radius=chain)


def metrics(name, stage, frame, truth, km, radius=20, **kwargs):
    distances = (
        cKDTree(xy(frame)).query(xy(truth))[0] if len(frame) else np.full(len(truth), np.inf)
    )
    matches = match_one_to_one(frame, xy(truth), radius)
    unique = len(np.unique(xy(frame), axis=0))
    return dict(
        method=name,
        stage=stage,
        tolerance_m=radius,
        candidates=len(frame),
        unique_coordinates=unique,
        road_km=km,
        density_per_km=len(frame) / km,
        inventory=len(truth),
        covered_any=int((distances <= radius).sum()),
        matches_one_to_one=len(matches),
        missed=int((distances > radius).sum()),
        recall_one_to_one=len(matches) / len(truth),
        coverage_fraction=float((distances <= radius).mean()),
        nearest_median_m=float(np.median(distances)),
        nearest_p90_m=float(np.quantile(distances, 0.9)),
        matched_median_m=float(matches.distance_m.median()) if len(matches) else None,
        matches_per_100_sites=100 * len(matches) / len(frame) if len(frame) else None,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    out = parser.parse_args().output
    out.mkdir(exist_ok=True, parents=True)
    assert not (out / "COMPUTATIONS_COMPLETE.json").exists(), "Never overwrite a completed study"
    assert (out / "FIXED_METHOD.md").exists(), "Copy the fixed plan before running"
    before = protected()
    original_paths = [ROOT / p for p in before]
    original_paths += [
        p
        for base in (ROOT / "web", ROOT / "api", ROOT / "models", PREVIOUS_FEATURE)
        for p in base.rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    ]
    original_paths += [
        HYBRID / "all_raw_proposals.parquet",
        PREVIOUS / "physical_measurements_v2.parquet",
        PRIOR / "recovered_road_segments.parquet",
        PRIOR / "inventory_split_manifest.csv",
    ]
    original_paths += list(PREVIOUS.glob("scores_RF_fold*.parquet"))
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(original_paths))}
    save_json(out / "protected_inputs.json", hashes)
    (out / "grouping_code_as_run.txt").write_bytes(
        (ROOT / "staging/retrieval_sites.py").read_bytes()
    )
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    roads["id"] = roads.segment_id.astype(str)
    truth = inventory()
    real = truth.loc[truth.fold.ge(0)].reset_index(drop=True)
    provenance(out, roads, real)
    km = float(roads.length.sum() / 1000)
    raw = gpd.read_parquet(HYBRID / "all_raw_proposals.parquet")
    feature = gpd.read_parquet(PREVIOUS_FEATURE / "feature_proposals.parquet")
    old = gpd.read_parquet(PREVIOUS_FEATURE / "scorefree_sites.parquet")
    new = identity_groups(feature)
    new.to_parquet(out / "identity_feature_sites.parquet", index=False)
    new.to_crs(4326).to_file(out / "identity_feature_sites.geojson", driver="GeoJSON")
    coverage_rows, grouping_rows, road_rows, per_truth = [], [], [], []
    populations = {"all_raw_sources": raw, "feature_only": feature}
    for spacing in (10, 20, 40, 75):
        samples = uniform(roads, spacing=spacing)
        populations[f"sampling_{spacing}m"] = samples
        populations[f"feature_plus_{spacing}m"] = gpd.GeoDataFrame(
            pd.concat([feature, samples], ignore_index=True), crs=32618
        )
    for name, frame in populations.items():
        for radius in TOLERANCES:
            coverage_rows.append(metrics(name, "candidates", frame, real, km, radius))
        # Existing NMS only here: samples do not carry physical channel identities.
        sites = cluster_sites(frame)
        for radius in TOLERANCES:
            coverage_rows.append(metrics(name, "legacy20_sites", sites, real, km, radius))
        sites.to_parquet(out / f"sites_{name}_legacy20.parquet", index=False)
        for road in roads.itertuples():
            own = frame.loc[frame.road_id.astype(str).eq(str(road.id))]
            road_rows.append(
                dict(
                    method=name,
                    road_id=str(road.id),
                    route=str(road.corridor),
                    road_km=road.geometry.length / 1000,
                    candidates=len(own),
                    unique_coordinates=len(np.unique(xy(own), axis=0)),
                    candidates_per_km=len(own) / (road.geometry.length / 1000),
                )
            )
        counts = cKDTree(xy(frame)).query_ball_point(xy(truth), 20, return_length=True)
        nearest = cKDTree(xy(frame)).query(xy(truth))[0]
        per_truth.extend(
            dict(
                method=name,
                record_key=str(r.record_key),
                fold=int(r.fold),
                candidates_within20=int(n),
                nearest_m=float(d),
            )
            for r, n, d in zip(truth.itertuples(), counts, nearest)
        )
        print(name, len(frame), "candidates", len(sites), "legacy sites", flush=True)
    for radius in RADII:
        for method, frame in [
            ("legacy_road_nms", cluster_sites(feature, radius)),
            ("evidence_identity", identity_groups(feature, radius)),
        ]:
            for tolerance in TOLERANCES:
                grouping_rows.append(
                    metrics(
                        method,
                        "grouping_sensitivity",
                        frame,
                        real,
                        km,
                        tolerance,
                        grouping_distance_m=radius,
                        cross_road_groups=0,
                        groups_spanning_over_2radius=0,
                    )
                )
        for method in ("geodesic_nms", "geodesic_dbscan_min1"):
            frame, diag = spatial_alternative(feature, radius, method)
            for tolerance in TOLERANCES:
                grouping_rows.append(
                    metrics(
                        method,
                        "unsafe_spatial_diagnostic",
                        frame,
                        real,
                        km,
                        tolerance,
                        grouping_distance_m=radius,
                        **diag,
                    )
                )
    pd.DataFrame(coverage_rows).to_csv(out / "candidate_generation.csv", index=False)
    pd.DataFrame(grouping_rows).to_csv(out / "grouping_sensitivity.csv", index=False)
    pd.DataFrame(road_rows).to_csv(out / "density_by_road.csv", index=False)
    pd.DataFrame(per_truth).to_csv(out / "candidate_counts_per_inventory.csv", index=False)
    # Disjoint nearest-coverage waterfall, then one-to-one capacity and ranking separately.
    audit = truth[["record_key", "fold", "corridor", "geometry"]].copy()
    for name, frame in [
        ("raw", raw),
        ("admitted", feature),
        ("old_site", old),
        ("identity_site", new),
    ]:
        audit[name + "_nearest_m"] = cKDTree(xy(frame)).query(xy(truth))[0]
        pairs = match_one_to_one(frame, xy(real))
        matched = set(real.iloc[pairs.real_row.to_numpy(int)].record_key)
        audit[name + "_matched_one_to_one"] = audit.record_key.isin(matched)
    audit["legacy_loss_category"] = np.select(
        [
            audit.fold.lt(0),
            audit.raw_nearest_m.gt(20),
            audit.admitted_nearest_m.gt(20),
            audit.old_site_nearest_m.gt(20),
            ~audit.old_site_matched_one_to_one,
        ],
        [
            "outside_eligible_road_footprint",
            "no_raw_candidate_within20m",
            "feature_admission_loss",
            "grouping_representative_loss",
            "one_to_one_capacity_conflict",
        ],
        default="represented_at_site_level",
    )
    audit.to_parquet(out / "coverage_failure_audit.parquet", index=False)
    audit.drop(columns="geometry").to_csv(out / "coverage_failure_audit.csv", index=False)
    audit.legacy_loss_category.value_counts().rename_axis("category").to_csv(
        out / "recall_waterfall.csv", header=["records"]
    )
    # Fixed scores only: all anchors + strict crossings have saved compatible predictions.
    score_cache = pd.concat(
        [gpd.read_parquet(PREVIOUS / f"scores_RF_fold{f}.parquet") for f in range(3)],
        ignore_index=True,
    ).set_index("candidate_id")
    dense = raw.loc[
        raw.proposal_kind.isin(["internal_road_probe", "mapped_crossing", "dem_crossing"])
    ].copy()
    dense["model_score"] = score_cache.model_score.reindex(dense.candidate_id).to_numpy()
    dense["localization_evidence"] = dense.candidate_id.map(
        feature.set_index("candidate_id").localization_evidence
    ).fillna("[]")
    assert dense.model_score.notna().all()
    draw_rows, rank_rows, route_rows = [], [], []
    for fold in range(3):
        held = set(roads.loc[roads.fold.eq(fold), "corridor"])
        t = real.loc[real.fold.eq(fold)].reset_index(drop=True)
        feat = feature.loc[feature.route_id.isin(held)]
        combined = dense.loc[dense.route_id.isin(held)]
        for method, frame, grouping in [
            ("feature_legacy20", feat, ranked),
            ("feature_identity20", feat, ranked_identity),
            ("dense_legacy20", combined, ranked),
        ]:
            for policy in ("raw_rf", "dynamic_route_z"):
                sites = grouping(frame, policy)
                sites.to_parquet(out / f"ranked_{method}_{policy}_fold{fold}.parquet", index=False)
                edges = adjacency(sites, t)
                order = np.arange(len(sites))
                observed = prefix_matches(edges, order, KS)
                blocks = (
                    sites.route_id.astype(str)
                    + "|"
                    + sites.road_id.astype(str)
                    + "|"
                    + sites.road_part_index.astype(str)
                    + "|"
                    + (sites.road_chainage_m // 2000).astype(int).astype(str)
                ).to_numpy()
                values = np.empty((1000, len(KS)), int)
                for rep in range(1000):
                    perm = conditional_order(
                        order, blocks, stream_seed("retrieval-20261016", fold, method, policy, rep)
                    )
                    values[rep] = prefix_matches(edges, perm, KS)
                for col, k in enumerate(KS):
                    top = sites.head(k)
                    assert observed[col] == len(match_one_to_one(top, xy(t)))
                    rank_rows.append(
                        dict(
                            method=method,
                            fold=fold,
                            policy=policy,
                            k=k,
                            actual_slots=len(top),
                            inventory=len(t),
                            represented_sites=len(sites),
                            inventory_match_fraction=observed[col] / len(top) if len(top) else None,
                            field_precision_at_k=None,
                            site_recall=observed[col] / len(t),
                            **summary(values[:, col], observed[col]),
                        )
                    )
                    draw_rows.extend(
                        dict(
                            method=method,
                            fold=fold,
                            policy=policy,
                            k=k,
                            replicate=i,
                            matched=int(v),
                        )
                        for i, v in enumerate(values[:, col])
                    )
                    for route in sorted(held):
                        own = top.loc[top.route_id.eq(route)]
                        route_rows.append(
                            dict(
                                method=method,
                                fold=fold,
                                policy=policy,
                                k=k,
                                route=route,
                                slots=len(own),
                                slot_share=len(own) / len(top) if len(top) else 0,
                                matches=len(match_one_to_one(own, xy(t.loc[t.corridor.eq(route)]))),
                            )
                        )
                print("Ranking", fold, method, policy, observed.tolist(), flush=True)
    pd.DataFrame(rank_rows).to_csv(out / "ranking_by_fold.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "ranking_by_route.csv", index=False)
    draws = pd.DataFrame(draw_rows)
    draws.to_parquet(out / "ranking_local_null_draws.parquet", index=False)
    combined_rows = []
    rank_frame = pd.DataFrame(rank_rows)
    for (method, policy, k), group in draws.groupby(["method", "policy", "k"]):
        part = rank_frame.loc[
            rank_frame.method.eq(method) & rank_frame.policy.eq(policy) & rank_frame.k.eq(k)
        ]
        observed = int(part.observed.sum())
        slots = int(part.actual_slots.sum())
        combined_rows.append(
            dict(
                method=method,
                policy=policy,
                k=int(k),
                actual_slots=slots,
                inventory_match_fraction=observed / slots,
                site_recall=observed / len(real),
                field_precision_at_k=None,
                **summary(group.groupby("replicate").matched.sum(), observed),
            )
        )
    pd.DataFrame(combined_rows).to_csv(out / "ranking_combined.csv", index=False)
    assert before == protected()
    assert all(sha256_file(ROOT / p) == h for p, h in hashes.items())
    save_json(
        out / "COMPUTATIONS_COMPLETE.json",
        dict(
            model_fits=0,
            prediction_calls=0,
            unchanged_inputs=len(hashes),
            grouping_default="legacy20 retained; identity20 explicit option",
            null_draws=18000,
            eligible=705,
            all_inventory=714,
            warning="Recommendations require field verification.",
        ),
    )
    print("Complete; originals unchanged; no model training or scoring.", flush=True)


if __name__ == "__main__":
    main()
