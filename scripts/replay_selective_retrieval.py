"""Replay the fixed study into a NEW research directory, never a live app path."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/selective_admission_20261016"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / "reports/experiments"):
        parser.error(
            "Use an unused directory below reports/experiments; completed studies are immutable"
        )
    out.mkdir(parents=True)
    shutil.copyfile(SOURCE / "METHOD.md", out / "METHOD.md")
    for script in (
        "evaluate_selective_retrieval.py",
        "audit_selective_retrieval.py",
        "verify_selective_retrieval.py",
        "report_selective_retrieval.py",
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--output", str(out)],
            cwd=ROOT,
            check=True,
        )
    print(
        "Fixed experiment replay complete. Four NEW experimental fits; no frozen models or defaults changed."
    )
    print(f"Report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
