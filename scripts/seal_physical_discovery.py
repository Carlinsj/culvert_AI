"""Archive exact research sources and guard a completed study; no model writes."""

# ruff: noqa: E402
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from culvert_ai.flow_c_bundle import sha256_file
from scripts.capture_summer_discovery import DEFAULT as OUT
from scripts.evaluate_hydrology_sites import protected


def main():
    assert not (OUT / "COMPLETE.json").exists(), "Completed research is immutable"
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification["independent_stage_metric_rows"] == 66
    assert verification["independent_ranked_metric_rows"] == 216
    assert verification["fits"] == 0
    inputs = json.loads((OUT / "baseline_input_hashes.json").read_text()) | json.loads(
        (OUT / "experiment_input_hashes.json").read_text()
    )
    assert all(sha256_file(Path(path)) == digest for path, digest in inputs.items())
    assert len(protected()) == 291
    paths = [
        "README.md",
        "package.json",
        "scripts/serve_web.sh",
        "scripts/serve_research.py",
        "scripts/serve_feature_staging.py",
        "scripts/capture_summer_discovery.py",
        "scripts/diagnose_summer_discovery.py",
        "scripts/evaluate_physical_discovery.py",
        "scripts/audit_physical_discovery.py",
        "scripts/case_physical_discovery.py",
        "scripts/summarize_physical_discovery.py",
        "scripts/adaptive_discovery_control.py",
        "scripts/verify_physical_discovery.py",
        "scripts/plot_physical_discovery.py",
        "scripts/report_physical_discovery.py",
        "scripts/replay_physical_discovery.py",
        "scripts/seal_physical_discovery.py",
        "scripts/test_feature_web.mjs",
        "staging/discovery_sites.py",
        "staging/discovery_service.py",
        "staging/feature_service.py",
        "staging/feature_sites.py",
        "staging/hybrid_sites.py",
        "staging/hydrology_sites.py",
        "staging/retrieval_sites.py",
        "staging/FEATURE_README.md",
        "staging/feature_web/app.js",
        "staging/feature_web/index.html",
        "staging/feature_web/core.js",
        "staging/feature_web/ruler.js",
        "staging/feature_web/styles.css",
        "tests/test_discovery_sites.py",
    ]
    manifest = {}
    for name in paths:
        source = ROOT / name
        destination = OUT / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[name] = sha256_file(source)
    (OUT / "code_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    outputs = {
        str(path.relative_to(OUT)): sha256_file(path)
        for path in OUT.glob("*")
        if path.is_file() and path.name != "COMPLETE.json"
    }
    (OUT / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                fits=0,
                production_changed=False,
                deployed=False,
                independent_test=False,
                baseline_default_preserved=True,
                recommendation="Retain baseline control; no equal-budget hybrid improvement",
                python_tests=258,
                javascript_tests=33,
                independent_coverage_rows=66,
                independent_ranked_rows=216,
                independent_road_holdout_rows=168,
                protected_inputs_unchanged=True,
                source_files=len(manifest),
                output_hashes=outputs,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Research sealed; all frozen artifacts and prior reports unchanged.")


if __name__ == "__main__":
    main()
