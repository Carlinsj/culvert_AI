"""Additional route, grouping and whole-road-excluded-model checks; zero fits."""

# ruff: noqa: E402
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
import numpy as np
import pandas as pd

from scripts.capture_summer_discovery import DEFAULT as OUT, PRIOR
from scripts.diagnose_summer_discovery import xy, own
from scripts.evaluate_physical_discovery import VARIANTS, KS
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.discovery_sites import ordered, physical_groups
from staging.feature_sites import ranked
from scripts.verify_physical_discovery import independent_count


def route_rows(queue, truth, **metadata):
    matches = match_one_to_one(queue, xy(truth), 20)
    ids = set(truth.iloc[matches.real_row].record_id)
    return [
        dict(
            **metadata,
            region=region,
            route=route,
            records=len(group),
            matched20=int(group.record_id.isin(ids).sum()),
            independent=False,
        )
        for (region, route), group in truth.groupby(["Region", "Route"])
    ]


def main():
    assert (OUT / "COMPUTATIONS_COMPLETE.json").exists()
    audit = pd.read_csv(OUT / "all_959_record_diagnosis.csv").set_index("record_id")
    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
    truth = truth.loc[truth.record_id.map(audit.query_success)].reset_index(drop=True)
    routes = []
    for variant in VARIANTS:
        frame = gpd.read_parquet(OUT / f"variant_{variant}.parquet")
        for family in ["RF", "HGB"]:
            for policy in ["raw_rf", "dynamic_route_z"]:
                queue = ordered(frame.assign(model_score=frame[f"score_{family}"]), policy=policy)
                for k in KS:
                    routes.extend(
                        route_rows(
                            queue.iloc[:k].reset_index(drop=True),
                            truth,
                            variant=variant,
                            family=family,
                            policy=policy,
                            budget=k,
                        )
                    )
    pd.DataFrame(routes).to_csv(OUT / "ranked_region_route_performance.csv", index=False)
    tasks = json.loads((PRIOR / "regional_fixed_method.json").read_text())["tasks"]
    parts = []
    for ix, iy, _, _ in tasks:
        path = OUT / "cells" / f"{ix}_{iy}" / "challenger/raw.parquet"
        if path.exists():
            raw = gpd.read_parquet(path)
            parts.append(own(raw.loc[raw.inside_query].copy(), ix, iy))
    raw = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=32618)
    assert raw.candidate_id.is_unique
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
    results = []
    road_results = []
    for family in ["RF", "HGB"]:
        frame = raw.loc[raw[f"score_{family}_roadheldout"].notna()].copy()
        frame["model_score"] = frame[f"score_{family}_roadheldout"]
        for variant in VARIANTS:
            population = {
                "BASELINE": frame.loc[frame.evidence_admitted],
                "DENSE": frame.loc[frame.proposal_kind.eq("internal_road_probe")],
                "STREAM": frame.loc[frame.proposal_kind.eq("mapped_crossing")],
                "DEM": frame.loc[frame.proposal_kind.isin(["dem_crossing", "small_dem_crossing"])],
                "HYBRID": frame.loc[frame.discovery_admitted],
                "HYBRID-GROUPED": frame.loc[frame.discovery_admitted],
            }[variant].copy()
            if variant == "HYBRID-GROUPED":
                population = physical_groups(population)
            for policy in ["raw_rf", "dynamic_route_z"]:
                # Crucially BASELINE NMS is rerun using excluded-road scores,
                # NOT geometry selected earlier by the final exposed model.
                queue = (
                    ranked(population, policy)
                    if variant == "BASELINE"
                    else ordered(population, policy=policy)
                )
                for k in [*KS, "all"]:
                    selected = queue if k == "all" else queue.iloc[:k]
                    selected = selected.reset_index(drop=True)
                    matches = match_one_to_one(selected, xy(actual), 20)
                    assert independent_count(selected, actual, 20) == len(matches)
                    results.append(
                        dict(
                            variant=variant,
                            family=family,
                            policy=policy,
                            budget=k,
                            recommendations=len(selected),
                            records=len(actual),
                            matched=len(matches),
                            recall=len(matches) / len(actual),
                            whole_corridor_model_exclusion=True,
                            geometry_selected_by_heldout_model=variant == "BASELINE",
                            independent=False,
                        )
                    )
                    road_results.extend(
                        route_rows(
                            selected,
                            actual,
                            variant=variant,
                            family=family,
                            policy=policy,
                            budget=k,
                        )
                    )
    pd.DataFrame(results).to_csv(OUT / "proper_complete_road_holdout_comparison.csv", index=False)
    pd.DataFrame(road_results).to_csv(OUT / "proper_complete_road_holdout_routes.csv", index=False)
    # Common records: nearest-point change is descriptive, not assigned-pair error.
    distances = pd.read_csv(OUT / "variant_record_distances.csv")
    old = distances.loc[distances.variant.eq("BASELINE")].set_index("record_id")
    new = distances.loc[distances.variant.eq("HYBRID-GROUPED")].set_index("record_id")
    paired = (
        old[["source_excel_row", "region", "route", "nearest_distance_m", "matched20"]]
        .rename(
            columns={"nearest_distance_m": "baseline_nearest_m", "matched20": "baseline_matched"}
        )
        .join(
            new[["nearest_distance_m", "matched20"]].rename(
                columns={"nearest_distance_m": "hybrid_nearest_m", "matched20": "hybrid_matched"}
            )
        )
    )
    paired["nearest_change_m"] = paired.hybrid_nearest_m - paired.baseline_nearest_m
    paired.to_csv(OUT / "paired_localization.csv")
    print(
        "Saved ranked route breakdown, independently checked whole-road-excluded comparisons and paired localization."
    )


if __name__ == "__main__":
    main()
