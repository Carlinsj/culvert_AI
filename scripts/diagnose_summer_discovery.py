"""Stage-by-stage summer diagnosis, without fitting or changing any pipeline."""

# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.capture_summer_discovery import DEFAULT as OUT, PRIOR
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.hydrology_service import datasets

TOLERANCES = (5, 10, 15, 20, 25, 50)


def xy(frame):
    return np.c_[frame.geometry.x, frame.geometry.y]


def own(frame, ix, iy):
    grid = np.floor(xy(frame) / 2000).astype(int)
    return frame.loc[(grid[:, 0] == ix) & (grid[:, 1] == iy)].copy()


def nearest(frame, truth):
    if not len(frame):
        return np.full(len(truth), np.inf), np.full(len(truth), -1)
    return cKDTree(xy(frame)).query(xy(truth))


def primary_failure(row):
    """Ordered, exhaustive categories. Full-list losses cannot be Top25 losses."""
    if not np.isfinite([row.Latitude, row.Longitude]).all():
        return "Coordinate or matching ambiguity", "invalid_coordinates"
    if not row.query_success:
        if not row.inside_terrain:
            return "Search/query limitation", "outside_loaded_terrain"
        return "No road coverage", "query_rejected_no_loaded_roads"
    if row.full_list_matched20:
        return "Matched", "baseline_full_match"
    if row.road_distance_m > 100:
        return "No road coverage", "record_far_from_loaded_roads"
    if row.nearest_raw_owned_m > 100:
        return "No raw candidate generated", "no_candidate_within100m"
    if row.nearest_raw_owned_m > 20:
        if row.nearest_raw_query_m <= 20:
            return "Search/query limitation", "cell_ownership_boundary"
        return "Candidate poorly localized", "raw_candidate_outside20m"
    if row.nearest_proposals_owned_m > 20:
        if row.nearest_proposals_query_m <= 20:
            return "Search/query limitation", "admitted_site_outside_owned_cell"
        return "Candidate removed by grouping/filtering", "physical_evidence_admission"
    if row.nearest_sites_owned_m > 20:
        if row.nearest_sites_query_m <= 20:
            return "Search/query limitation", "representative_outside_owned_cell"
        return "Candidate removed by grouping/filtering", "score_ordered_representative"
    return "Coordinate or matching ambiguity", "one_to_one_capacity_competition"


def main():
    assert (OUT / "BASELINE_CAPTURE_COMPLETE.json").exists()
    assert not (OUT / "DIAGNOSIS_COMPLETE.json").exists()
    audit = pd.read_csv(PRIOR / "regional_record_audit.csv")
    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet").set_index("record_id")
    truth = truth.loc[audit.record_id].reset_index()
    for column in audit.columns:
        truth[column] = audit[column].to_numpy()
    pools = {k: [] for k in ["raw", "proposals", "consolidated", "sites"]}
    for (ix, iy), target in truth.loc[truth.query_success].groupby(["cell_x", "cell_y"]):
        folder = OUT / "cells" / f"{ix}_{iy}"
        for stage in pools:
            full = gpd.read_parquet(
                folder / ("raw_features.parquet" if stage == "raw" else f"{stage}.parquet")
            )
            if stage == "raw":
                full = full.loc[full.inside_query].copy()
            kept = own(full, ix, iy)
            kept["cell_x"], kept["cell_y"] = ix, iy
            pools[stage].append(kept)
            distance, _ = nearest(full, target)
            truth.loc[target.index, f"nearest_{stage}_query_m"] = distance
    frames = {}
    metrics = []
    target = truth.loc[truth.query_success].reset_index(drop=True)
    for stage, parts in pools.items():
        frame = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=32618)
        frame = frame.sort_values(["cell_x", "cell_y", "candidate_id"], kind="stable").reset_index(
            drop=True
        )
        assert frame.candidate_id.is_unique
        frames[stage] = frame
        frame.to_parquet(OUT / f"baseline_{stage}.parquet", index=False)
        distance, positions = nearest(frame, truth)
        truth[f"nearest_{stage}_owned_m"] = distance
        truth[f"nearest_{stage}_id"] = frame.iloc[positions].candidate_id.to_numpy()
        if stage == "raw":
            for col in [
                "model_score",
                "evidence_admitted",
                "proposal_kind",
                "road_id",
                "road_name",
                "road_profile_available",
                "route_elevation_dip_60m",
                "upstream_area_m2",
                "negative_tpi_9x9_m",
                "distance_to_nearest_stream_m",
                "hydrology_edge_risk",
            ]:
                if col in frame:
                    truth[f"nearest_raw_{col}"] = frame.iloc[positions][col].to_numpy()
        for tolerance in TOLERANCES:
            matches = match_one_to_one(frame, xy(target), tolerance)
            nd, _ = nearest(frame, target)
            metrics.append(
                dict(
                    stage=stage,
                    tolerance_m=tolerance,
                    candidates=len(frame),
                    inventory=len(target),
                    covered_any=int((nd <= tolerance).sum()),
                    matched_one_to_one=len(matches),
                    recall=len(matches) / len(target),
                )
            )
    sites = (
        frames["sites"]
        .sort_values(["cell_x", "cell_y", "global_rank", "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )
    for budget, expected in [(None, 347), (25, 155)]:
        queue = (
            sites
            if budget is None
            else sites.loc[sites.global_rank.le(budget)].reset_index(drop=True)
        )
        actual = match_one_to_one(queue, xy(truth), 20)
        assert len(actual) == expected
    categories = truth.apply(primary_failure, axis=1)
    truth["primary_failure_category"] = categories.map(lambda v: v[0])
    truth["failure_substage"] = categories.map(lambda v: v[1])
    truth["retained_but_below_default_budget"] = (
        truth.full_list_matched20 & ~truth.top25_per_query_matched20
    )
    truth["possible_coordinate_road_mismatch"] = truth.road_distance_m.gt(20)
    # These describe missing mapped-water support, not confirmed ditch identity.
    truth["possible_unmapped_small_drainage"] = truth.nearest_raw_distance_to_nearest_stream_m.gt(
        20
    ) & truth.nearest_raw_owned_m.le(20)
    roads, streams, networks = datasets()
    for name, layer in [("mapped_water", streams), ("dem_0p1km2", networks["0p1km2"])]:
        (left, _), distance = layer.sindex.nearest(
            truth.geometry, return_all=False, return_distance=True
        )
        truth.loc[left, f"distance_to_{name}_m"] = distance
    close = cKDTree(xy(truth)).query_ball_point(xy(truth), 20, return_length=True)
    truth["other_inventory_records_within20m"] = close - 1
    truth["coordinate_datum_assumed_not_survey_verified"] = True
    truth["independent_validation"] = False
    truth.drop(columns="geometry").to_csv(OUT / "all_959_record_diagnosis.csv", index=False)
    misses = truth.loc[truth.query_success & ~truth.full_list_matched20]
    assert len(misses) == 500 and int((~truth.query_success).sum()) == 112
    misses.groupby(["primary_failure_category", "failure_substage"]).size().rename(
        "records"
    ).to_csv(OUT / "supported_500_failure_counts.csv")
    truth.loc[~truth.query_success].groupby(
        ["Region", "Route", "primary_failure_category", "failure_substage"]
    ).size().rename("records").to_csv(OUT / "unsupported_112_breakdown.csv")
    pd.DataFrame(metrics).to_csv(OUT / "baseline_stage_coverage.csv", index=False)
    summary = dict(
        records=len(truth),
        supported=847,
        full_matches=347,
        top25_matches=155,
        supported_misses=len(misses),
        unsupported=112,
        baseline_reproduced=True,
        failure_counts=misses.failure_substage.value_counts().to_dict(),
        represented_below_default_budget=int(truth.retained_but_below_default_budget.sum()),
        fits=0,
    )
    import json

    (OUT / "DIAGNOSIS_COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
