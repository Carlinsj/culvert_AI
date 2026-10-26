"""Exact-budget random-phase road-position nulls;no model or policy changes."""

# ruff:noqa:E402
import sys
from pathlib import Path
import argparse
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_feature_sites import OUT
from scripts.validate_density_controls import (
    PRIOR,
    inventory,
    xy,
    cardinality,
    stream_seed,
    summary,
)
from scripts.evaluate_hydrology_sites import save_json, protected
from scripts.validate_inventory_heldout_roads import match_one_to_one
from staging.hydrology_sites import line_parts


def plan(roads, frame, control):
    parts = {
        (str(r.segment_id), j): line
        for r in roads.sort_values("segment_id").itertuples()
        for j, line in enumerate(line_parts(r.geometry))
        if line.length > 0
    }
    if control == "global_chainage":
        lines = list(parts.values())
        lengths = np.array([line.length for line in lines])
        ends = lengths.cumsum()
        return dict(
            lines=np.asarray(lines, dtype=object), ends=ends, total=float(ends[-1]), n=len(frame)
        )
    counts = Counter()
    for row in frame.itertuples():
        key = (str(row.road_id), int(row.road_part_index))
        line = parts[key]
        block = (
            min(int(row.road_chainage_m // 2000), max(int(np.ceil(line.length / 2000)) - 1, 0))
            if control == "road_part_block2km"
            else 0
        )
        counts[(key, block)] += 1
    geoms = []
    bases = []
    steps = []
    group_ids = []
    rows = []
    for i, ((key, block), n) in enumerate(sorted(counts.items())):
        line = parts[key]
        start = block * 2000 if control == "road_part_block2km" else 0
        end = min(start + 2000, line.length) if control == "road_part_block2km" else line.length
        step = (end - start) / n
        assert step > 0
        geoms.extend([line] * n)
        bases.extend(start + np.arange(n) * step)
        steps.extend([step] * n)
        group_ids.extend([i] * n)
        rows.append(
            dict(
                road_id=key[0],
                part=key[1],
                block=block,
                start_m=start,
                end_m=end,
                n=n,
                grid_step_m=step,
                tiny_below10=n < 10,
            )
        )
    return dict(
        geoms=np.asarray(geoms, dtype=object),
        bases=np.array(bases),
        steps=np.array(steps),
        groups=np.array(group_ids),
        strata=rows,
    )


def draw(p, rng, control):
    if control == "global_chainage":
        positions = (np.arange(p["n"]) + rng.random()) * p["total"] / p["n"]
        index = np.searchsorted(p["ends"], positions, side="right")
        starts = np.r_[0, p["ends"][:-1]]
        return shapely.line_interpolate_point(p["lines"][index], positions - starts[index])
    phases = rng.random(len(p["strata"]))
    return shapely.line_interpolate_point(p["geoms"], p["bases"] + p["steps"] * phases[p["groups"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT)
    out = parser.parse_args().input
    assert not (out / "spatial_null_summary.csv").exists(), "Never overwrite previous null results"
    save_json(
        out / "spatial_null_fixed_method.json",
        dict(
            seed=20261015,
            replicates=1000,
            controls=["global_chainage", "road_part", "road_part_block2km"],
            method="Uniform spacing with random phase per fixed geometry stratum,exact original per-stratum proposal counts;no label-driven boundaries.",
            stages=["candidate", "scorefree_site"],
            matching="joint705inclusive20m maximum cardinality",
            note="Control positions are proposed road locations,not confirmed physical structures;not reconsolidated to alter the fixed budget. Boundary-close points remain subject to one-to-one capacity.",
            models_trained=0,
            estimator_calls=0,
        ),
    )
    before = protected()
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    truth = inventory()
    truth = truth.loc[truth.fold.ge(0)]
    tree = cKDTree(xy(truth))
    results = []
    replicates = []
    strata = []
    for stage, file in [
        ("candidate", "feature_proposals.parquet"),
        ("scorefree_site", "scorefree_sites.parquet"),
    ]:
        frame = gpd.read_parquet(out / file)
        observed = len(match_one_to_one(frame, xy(truth)))
        for control in ["global_chainage", "road_part", "road_part_block2km"]:
            prepared = plan(roads, frame, control)
            values = []
            for row in prepared.get("strata", []):
                strata.append(dict(stage=stage, control=control, **row))
            for rep in range(1000):
                points = draw(
                    prepared, stream_seed("feature-spatial-null", stage, control, rep), control
                )
                assert len(points) == len(frame)
                edges = tree.query_ball_point(
                    np.c_[shapely.get_x(points), shapely.get_y(points)], 20
                )
                matched = cardinality(edges, range(len(points)), len(truth))
                any_count = len(set(j for links in edges for j in links))
                values.append(matched)
                replicates.append(
                    dict(
                        stage=stage,
                        control=control,
                        replicate=rep,
                        budget=len(points),
                        matched=matched,
                        covered_any=any_count,
                    )
                )
            result = dict(
                stage=stage,
                control=control,
                budget=len(frame),
                **summary(np.array(values), observed),
            )
            results.append(result)
            print(result, flush=True)
    pd.DataFrame(results).to_csv(out / "spatial_null_summary.csv", index=False)
    pd.DataFrame(replicates).to_csv(out / "spatial_null_replicates.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "spatial_null_strata.csv", index=False)
    assert before == protected()


if __name__ == "__main__":
    main()
