"""Rock Ledge diagnostic after the fixed regional diagnosis, never a tuning loop."""

# ruff: noqa: E402
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.capture_summer_discovery import DEFAULT as OUT, PRIOR
from scripts.diagnose_summer_discovery import xy, nearest, TOLERANCES
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.discovery_service import analyze


def frame(response, key):
    return gpd.GeoDataFrame.from_features(response[key], crs=4326).to_crs(32618)


def main():
    assert (OUT / "DIAGNOSIS_COMPLETE.json").exists()
    assert (OUT / "FIXED_METHOD.md").exists()
    old = json.loads((PRIOR / "current_rock.json").read_text())
    path = OUT / "rock_physical_discovery.json"
    if path.exists():
        new = json.loads(path.read_text())
    else:
        new = analyze(41.880519, -74.110507, 1500, "dynamic_route_z")
        path.write_text(json.dumps(new, allow_nan=False) + "\n")
    center = gpd.GeoSeries([Point(-74.110507, 41.880519)], crs=4326).to_crs(32618).iloc[0]
    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
    truth = truth.loc[truth.geometry.distance(center).le(1500)].reset_index(drop=True)
    assert len(truth) == 7
    queues = dict(
        baseline_raw=frame(old, "raw_candidates"),
        baseline_admitted=frame(old, "proposals"),
        baseline_sites=frame(old, "sites"),
        hybrid_raw=frame(new, "raw_candidates"),
        hybrid_hypotheses=frame(new, "proposals"),
        hybrid_grouped=frame(new, "sites"),
    )
    old_sites = queues["baseline_sites"].sort_values("global_rank").reset_index(drop=True)
    queues["baseline_209_filter"] = old_sites.loc[old_sites.road_name.eq("US Hwy 209")].copy()
    rows = []
    details = truth.drop(columns="geometry").copy()
    panels = []
    for name, queue in queues.items():
        queue.to_parquet(OUT / f"rock_{name}.parquet", index=False)
        nd, indices = nearest(queue, truth)
        details[f"{name}_nearest_m"] = nd
        details[f"{name}_nearest_id"] = queue.iloc[indices].candidate_id.to_numpy()
        if "global_rank" in queue:
            details[f"{name}_nearest_rank"] = queue.iloc[indices].global_rank.to_numpy()
        for k in [25, 50, 100, 250, 500, 1000, "all"]:
            selected = (
                queue.sort_values(["global_rank", "candidate_id"], kind="stable")
                .iloc[:k]
                .reset_index(drop=True)
                if k != "all" and "global_rank" in queue
                else queue.reset_index(drop=True)
            )
            if k != "all" and "global_rank" not in queue:
                continue
            for tolerance in TOLERANCES:
                matches = match_one_to_one(selected, xy(truth), tolerance)
                rows.append(
                    dict(
                        variant=name,
                        budget=k,
                        tolerance_m=tolerance,
                        recommendations=len(selected),
                        matched=len(matches),
                        records=7,
                        recall=len(matches) / 7,
                    )
                )
        matches = match_one_to_one(queue, xy(truth), 20)
        if name in ["baseline_raw", "baseline_sites", "hybrid_hypotheses", "hybrid_grouped"]:
            panels.append(
                dict(
                    name=name,
                    count=len(queue),
                    matched=len(matches),
                    points=xy(queue).tolist(),
                    actual=[
                        dict(
                            x=r.geometry.x,
                            y=r.geometry.y,
                            row=int(r.source_excel_row),
                            found=i in set(matches.real_row),
                        )
                        for i, r in truth.iterrows()
                    ],
                )
            )
    details["baseline_failure"] = [
        "Matched"
        if d <= 20
        else (
            "Physical-evidence admission"
            if raw <= 20 and admitted > 20
            else "Other; inspect stage distances"
        )
        for d, raw, admitted in zip(
            details.baseline_sites_nearest_m,
            details.baseline_raw_nearest_m,
            details.baseline_admitted_nearest_m,
        )
    ]
    details.to_csv(OUT / "rock_record_diagnosis.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "rock_variant_metrics.csv", index=False)
    roads = gpd.GeoDataFrame.from_features(new["analysis"]["roads"], crs=4326).to_crs(32618)
    lines = []
    for geometry in roads.clip(center.buffer(1500)).geometry:
        lines.extend(
            [
                list(p.coords)
                for p in getattr(geometry, "geoms", [geometry])
                if p.geom_type == "LineString"
            ]
        )
    (OUT / "rock_plot_data.json").write_text(
        json.dumps(dict(center=[center.x, center.y], roads=lines, panels=panels)) + "\n"
    )
    print(pd.DataFrame(rows).query("tolerance_m==20 and budget=='all'").to_string(index=False))


if __name__ == "__main__":
    main()
