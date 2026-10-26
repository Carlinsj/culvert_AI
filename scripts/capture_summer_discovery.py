"""Recompute the current summer evaluation and retain all stage evidence; zero fits."""

# ruff: noqa: E402 -- thread limits before numerical imports
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
import numpy as np
import pandas as pd

from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hydrology_sites import protected

PRIOR = ROOT / "reports/experiments/summer_inventory_20261016"
DEFAULT = ROOT / "reports/experiments/physical_discovery_20261016"


def capture(task):
    out, ix, iy, lat, lon = task
    out = Path(out)
    folder = out / "cells" / f"{ix}_{iy}"
    done = folder / "COMPLETE.json"
    if done.exists():
        return json.loads(done.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    import staging.feature_service as service

    captured = {}
    original_measure = service.physical_measurements
    original_transform = service.apply_feature_contract

    def measurement(*args, **kwargs):
        value = original_measure(*args, **kwargs)
        captured["measurements"] = value.copy()
        return value

    def transformation(*args, **kwargs):
        value = original_transform(*args, **kwargs)
        captured["transformed"] = value.copy()
        return value

    service.physical_measurements = measurement
    service.apply_feature_contract = transformation
    service.analyze.cache_clear()
    start = time.monotonic()
    record = dict(cell_x=ix, cell_y=iy, latitude=lat, longitude=lon)
    try:
        response = service.analyze(lat, lon, 1500, "dynamic_route_z", "feature", "legacy20")
    except ValueError as error:
        if not any(
            s in str(error)
            for s in ["Outside supported coverage", "No supported roads", "No usable road geometry"]
        ):
            raise
        record.update(success=False, error=str(error))
    else:
        with gzip.open(PRIOR / "tiles" / f"{ix}_{iy}.json.gz", "rt") as stream:
            previous = json.load(stream)
        assert previous["success"]
        old = sorted(previous["sites"]["features"], key=lambda f: f["properties"]["global_rank"])
        new = sorted(response["sites"]["features"], key=lambda f: f["properties"]["global_rank"])
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert a["properties"]["candidate_id"] == b["properties"]["candidate_id"]
            assert a["properties"]["global_rank"] == b["properties"]["global_rank"]
            np.testing.assert_allclose(
                a["geometry"]["coordinates"], b["geometry"]["coordinates"], atol=1e-10, rtol=0
            )
            assert np.isclose(
                a["properties"]["model_score"], b["properties"]["model_score"], atol=1e-12, rtol=0
            )
        raw = pd.DataFrame(
            [f["properties"] for f in response["raw_candidates"]["features"]]
        ).set_index("candidate_id")
        transformed = captured["transformed"]
        transformed["inside_query"] = transformed.candidate_id.isin(raw.index)
        for column in [
            "model_score",
            "evidence_admitted",
            "physical_crossing",
            "profile_local_peak",
            "flow_valley_local_peak",
            "ml_local_peak",
            "localization_evidence",
            "evidence_type",
        ]:
            transformed[column] = transformed.candidate_id.map(raw[column])
        for label, frame in [
            ("measurements", captured["measurements"]),
            ("raw_features", transformed),
        ]:
            frame.to_parquet(folder / f"{label}.parquet", index=False)
        for label in ["sites", "proposals", "consolidated"]:
            frame = gpd.GeoDataFrame.from_features(response[label], crs=4326)
            if frame.empty:
                frame = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
            frame.to_crs(32618).to_parquet(folder / f"{label}.parquet", index=False)
        roads = gpd.GeoDataFrame.from_features(response["analysis"]["roads"], crs=4326).to_crs(
            32618
        )
        roads.to_parquet(folder / "roads.parquet", index=False)
        record.update(
            success=True,
            sites=response["site_count"],
            raw=len(raw),
            model=response["model"],
            exact_prior_replay=True,
        )
    finally:
        service.physical_measurements = original_measure
        service.apply_feature_contract = original_transform
        service.analyze.cache_clear()
    record["seconds"] = time.monotonic() - start
    done.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "COMPLETE.json").exists():
        raise FileExistsError("Completed research is immutable; use a new output directory")
    if not (out / "DIAGNOSTIC_PLAN.md").exists():
        raise FileNotFoundError("Copy the diagnostic plan into the output directory first")
    before = protected()
    prior = json.loads((PRIOR / "regional_fixed_method.json").read_text())
    for name, digest in prior["input_hashes"].items():
        assert sha256_file(Path(name)) == digest
    # Preserve previous research outputs and the exact baseline source as inputs.
    sources = [
        p
        for root in [
            PRIOR,
            ROOT / "reports/experiments/site_retrieval_20261016",
            ROOT / "reports/experiments/feature_localization_20261015",
        ]
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    ]
    sources += [
        ROOT / "staging/feature_service.py",
        ROOT / "staging/feature_sites.py",
        ROOT / "staging/hydrology_sites.py",
        ROOT / "staging/hybrid_sites.py",
        ROOT / "staging/retrieval_sites.py",
    ]
    hashes = {str(p): sha256_file(p) for p in sources}
    saved = out / "baseline_input_hashes.json"
    if saved.exists():
        assert json.loads(saved.read_text()) == hashes
    else:
        saved.write_text(json.dumps(hashes, indent=2) + "\n")
    tasks = [(str(out), *task) for task in prior["tasks"]]
    records = []
    print(f"Replaying {len(tasks)} fixed queries with native stage capture; zero fits.", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
        for future in as_completed([pool.submit(capture, task) for task in tasks]):
            record = future.result()
            records.append(record)
            print(
                f"{len(records)}/{len(tasks)} cell={record['cell_x']},{record['cell_y']} success={record['success']} seconds={record['seconds']:.1f}",
                flush=True,
            )
    pd.DataFrame(records).drop(columns="model", errors="ignore").sort_values(
        ["cell_x", "cell_y"]
    ).to_csv(out / "baseline_capture.csv", index=False)
    assert protected() == before
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    (out / "BASELINE_CAPTURE_COMPLETE.json").write_text(
        json.dumps(
            dict(
                queries=len(records),
                successful=sum(r["success"] for r in records),
                prior_queries_reproduced=True,
                fits=0,
                protected_unchanged=len(before),
                historical_inputs_unchanged=len(hashes),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
