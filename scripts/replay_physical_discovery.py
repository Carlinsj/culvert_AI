"""Reproduce the fixed study into a NEW directory, with no fitting or deployment.

This deliberately expensive command is never invoked by npm startup. Existing
completed research is immutable. It creates only the explicitly chosen output.
"""

# ruff: noqa: E402
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from scripts.capture_summer_discovery import DEFAULT as ORIGINAL, PRIOR
from scripts.evaluate_hydrology_sites import protected


def run_cell(payload):
    import scripts.evaluate_physical_discovery as experiment

    output, task = payload
    experiment.OUT = Path(output)
    return experiment.cell(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(
            "Choose a new, unused output directory; completed studies are never overwritten"
        )
    output.mkdir(parents=True)
    for name in ["DIAGNOSTIC_PLAN.md", "FIXED_METHOD.md", "ADAPTIVE_CONTROL_METHOD.md"]:
        shutil.copyfile(ORIGINAL / name, output / name)
    before = protected()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/capture_summer_discovery.py"),
            "--output",
            str(output),
            "--workers",
            str(args.workers),
        ],
        check=True,
    )
    import scripts.diagnose_summer_discovery as diagnosis

    diagnosis.OUT = output
    diagnosis.main()
    import scripts.evaluate_physical_discovery as experiment

    experiment.OUT = output
    tasks = json.loads((PRIOR / "regional_fixed_method.json").read_text())["tasks"]
    inputs = [
        ROOT / "staging/discovery_sites.py",
        ROOT / "staging/discovery_service.py",
        ROOT / "scripts/evaluate_physical_discovery.py",
        output / "FIXED_METHOD.md",
    ]
    inputs += list((ROOT / "reports/experiments/hydrology_driven_sites_20261015").glob("*.joblib"))
    hashes = {str(p): sha256_file(p) for p in inputs}
    (output / "experiment_input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
        for i, future in enumerate(
            as_completed([pool.submit(run_cell, (str(output), task)) for task in tasks]), 1
        ):
            result = future.result()
            print(i, len(tasks), result["cell_x"], result["cell_y"], result["success"], flush=True)
    frames = experiment.pools(tasks)
    import geopandas as gpd

    truth = gpd.read_parquet(PRIOR / "inventory_coverage.parquet")
    experiment.summarize(frames, truth)
    experiment.distribution(tasks, frames)
    assert protected() == before
    assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    (output / "COMPUTATIONS_COMPLETE.json").write_text(
        json.dumps(
            dict(
                cells=len(tasks),
                fits=0,
                production_changed=False,
                frozen_models_unchanged=True,
                baseline_inputs_unchanged=True,
            ),
            indent=2,
        )
        + "\n"
    )
    import scripts.audit_physical_discovery as audit

    audit.OUT = output
    audit.main()
    import scripts.case_physical_discovery as case

    case.OUT = output
    case.main()
    import scripts.verify_physical_discovery as verify

    verify.OUT = output
    verify.main()
    import scripts.summarize_physical_discovery as summarize

    summarize.OUT = output
    summarize.main()
    import scripts.adaptive_discovery_control as adaptive

    adaptive.OUT = output
    adaptive.main()
    print(f"Fixed study reproduced in {output}. No model fits or deployment.")


if __name__ == "__main__":
    main()
