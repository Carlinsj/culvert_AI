from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Write compact model-quality summary for the UI.")
    parser.add_argument("--metrics", default="reports/actual_ulster_field_report_metrics.json")
    parser.add_argument("--point-analysis", default="reports/extracted_points_analysis.json")
    parser.add_argument("--training-points", default="data/processed/high_confidence_training_points.csv")
    parser.add_argument("--output", default="web/data/model_summary.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_summary(
                metrics_path=Path(args.metrics),
                point_analysis_path=Path(args.point_analysis),
                training_points_path=Path(args.training_points),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def build_summary(
    metrics_path: Path,
    point_analysis_path: Path,
    training_points_path: Path,
) -> dict[str, Any]:
    metrics = _read_json(metrics_path)
    point_analysis = _read_json(point_analysis_path)
    class_counts = metrics.get("class_counts", {}) if metrics else {}
    training_point_rows = _csv_row_count(training_points_path)

    if not metrics:
        return {
            "available": False,
            "reason": f"Metrics file not found: {metrics_path}",
            "training_points": training_point_rows,
            "point_qc": _point_qc(point_analysis),
            "source_files": _source_files(metrics_path, point_analysis_path, training_points_path),
        }

    feature_columns = metrics.get("feature_columns") or []
    return {
        "available": True,
        "selected_model": metrics.get("selected_model"),
        "selection_metric": metrics.get("selection_metric"),
        "rows": metrics.get("rows"),
        "positive_labels": _int_or_none(class_counts.get("1")),
        "negative_labels": _int_or_none(class_counts.get("0")),
        "training_points": training_point_rows,
        "feature_count": len(feature_columns),
        "random_holdout_average_precision": _nested_number(
            metrics,
            "random_holdout",
            "average_precision",
        ),
        "random_holdout_roc_auc": _nested_number(metrics, "random_holdout", "roc_auc"),
        "spatial_holdout_average_precision": _nested_number(
            metrics,
            "spatial_holdout",
            "average_precision",
        ),
        "spatial_holdout_roc_auc": _nested_number(metrics, "spatial_holdout", "roc_auc"),
        "spatial_holdout_top10_precision": _top_k_precision(metrics, "spatial_holdout", 10),
        "point_qc": _point_qc(point_analysis),
        "source_files": _source_files(metrics_path, point_analysis_path, training_points_path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _row in reader)


def _point_qc(point_analysis: dict[str, Any]) -> dict[str, Any]:
    if not point_analysis:
        return {}
    flags = point_analysis.get("analysis_flags") or {}
    return {
        "rows": point_analysis.get("rows"),
        "matched_existing_candidates": point_analysis.get("matched_existing_candidates"),
        "outside_analysis_extent": point_analysis.get("outside_analysis_extent"),
        "analysis_flags": flags,
    }


def _source_files(*paths: Path) -> list[dict[str, Any]]:
    files = []
    for path in paths:
        files.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return files


def _nested_number(payload: dict[str, Any], section: str, key: str) -> float | None:
    value = (payload.get(section) or {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_k_precision(payload: dict[str, Any], section: str, k: int) -> float | None:
    for item in (payload.get(section) or {}).get("top_k") or []:
        if item.get("k") == k:
            return _float_or_none(item.get("precision_at_k"))
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
