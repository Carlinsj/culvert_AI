"""Supplemental same-road adaptive computational coverage, not detections."""

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
from scripts.diagnose_summer_discovery import own, xy, nearest, TOLERANCES
from scripts.validate_inventory_heldout_roads import match_one_to_one
from scripts.verify_physical_discovery import independent_count


def adaptive(coarse, fine, seeds, radius=75):
    """Actual road/part chainage windows, not spatial shortcuts or label snapping."""
    selected = pd.Series(False, index=fine.index)
    for (road, part), group in seeds.groupby(["road_id", "road_part_index"], sort=True):
        use = fine.road_id.astype(str).eq(str(road)) & fine.road_part_index.eq(part)
        if not use.any():
            continue
        chain = np.sort(group.road_chainage_m.to_numpy(float))
        positions = fine.loc[use, "road_chainage_m"].to_numpy(float)
        ix = np.searchsorted(chain, positions)
        left = chain[np.maximum(ix - 1, 0)]
        right = chain[np.minimum(ix, len(chain) - 1)]
        selected.loc[use] = np.minimum(abs(positions - left), abs(positions - right)) <= radius
    result = gpd.GeoDataFrame(pd.concat([coarse, fine.loc[selected]], ignore_index=True), crs=32618)
    return (
        result.sort_values("candidate_id", kind="stable")
        .drop_duplicates("candidate_id")
        .reset_index(drop=True)
    )


def main():
    assert (OUT / "ADAPTIVE_CONTROL_METHOD.md").exists()
    tasks = json.loads((PRIOR / "regional_fixed_method.json").read_text())["tasks"]
    pieces = []
    for ix, iy, _, _ in tasks:
        folder = OUT / "cells" / f"{ix}_{iy}" / "challenger"
        if not (folder / "COMPLETE.json").exists():
            continue
        coarse = gpd.read_parquet(folder / "sampling_75m.parquet")
        fine = gpd.read_parquet(folder / "sampling_10m.parquet")
        seeds = gpd.read_parquet(folder / "HYBRID.parquet")
        points = own(adaptive(coarse, fine, seeds), ix, iy)
        pieces.append(points)
    points = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=32618)
    assert points.candidate_id.is_unique
    points.to_parquet(OUT / "adaptive_computational_probes.parquet", index=False)
    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
    audit = pd.read_csv(OUT / "all_959_record_diagnosis.csv").set_index("record_id")
    truth = truth.loc[truth.record_id.map(audit.query_success)].reset_index(drop=True)
    distances, _ = nearest(points, truth)
    rows = []
    for tolerance in TOLERANCES:
        matches = match_one_to_one(points, xy(truth), tolerance)
        assert independent_count(points, truth, tolerance) == len(matches)
        rows.append(
            dict(
                variant="adaptive_evidence_windows_75_10m",
                candidates=len(points),
                inventory=len(truth),
                tolerance_m=tolerance,
                covered_any=int((distances <= tolerance).sum()),
                matched=len(matches),
                recall=len(matches) / len(truth),
                runtime_saving_demonstrated=False,
                default_recommendations=False,
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "adaptive_sampling_coverage.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
