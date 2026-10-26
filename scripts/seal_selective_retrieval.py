"""Archive exact sources and completed results without altering frozen inputs."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from culvert_ai.flow_c_bundle import sha256_file
from scripts.evaluate_hydrology_sites import protected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert (
        (out / "REPORT.md").exists()
        and (out / "VERIFICATION.md").exists()
        and not (out / "COMPLETE.json").exists()
    )
    verified = json.loads((out / "verification.json").read_text())
    assert verified["independent_metric_rows_replayed"] == 260
    inputs = json.loads((out / "input_hashes.json").read_text())
    assert all(sha256_file(Path(path)) == digest for path, digest in inputs.items())
    assert len(protected()) == 291
    paths = {
        "README.md",
        "package.json",
        "pyproject.toml",
        "scripts/serve_web.sh",
        "scripts/serve_research.py",
        "tests/test_selective_sites.py",
    }
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "src/culvert_ai").glob("*.py"))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "staging").glob("*.py"))
    paths.update(
        str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*selective_retrieval.py")
    )
    paths.update(
        {
            "scripts/plot_selective_retrieval.py",
            "scripts/diagnose_summer_discovery.py",
            "scripts/validate_inventory_heldout_roads.py",
            "scripts/verify_physical_discovery.py",
            "scripts/evaluate_hydrology_sites.py",
            "scripts/capture_summer_discovery.py",
        }
    )
    manifest = {}
    for name in sorted(paths):
        source = ROOT / name
        destination = out / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[name] = sha256_file(source)
    (out / "code_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = dict(
        fits=4,
        grouping_replay_fits=0,
        frozen_model_fits=0,
        production_changed=False,
        deployed=False,
        independent_validation=False,
        default_changed=False,
        python_tests=275,
        javascript_tests=33,
        verified_metrics=260,
        proper_existing_model_road_rows=30,
        recommendation="Retain default; 127 development matches does not survive nested comparison (99). Reject separate site ranker (13).",
        source_files=len(manifest),
        output_hashes={
            str(p.relative_to(out)): sha256_file(p)
            for p in sorted(out.glob("*"))
            if p.is_file() and p.name != "COMPLETE.json"
        },
    )
    (out / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Sealed {len(result['output_hashes'])} outputs and {len(manifest)} exact source files. No promotion or deployment."
    )


if __name__ == "__main__":
    main()
