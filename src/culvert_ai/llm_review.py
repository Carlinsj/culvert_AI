from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from culvert_ai.field_reports import _deduplicate_records, extract_field_report_records
from culvert_ai.io import ensure_parent_dir, write_vector


LLM_REVIEW_INSTRUCTION = (
    "Validate this extracted culvert field-report row. Do not invent coordinates. "
    "Accept only if the context clearly indicates a culvert/site-location coordinate in New York. "
    "Correct route, culvert_id, latitude, or longitude only when the context supports it."
)


def write_llm_label_review_queue(
    input_path: str | Path,
    output_path: str | Path,
    dedupe_precision: int = 6,
) -> dict:
    records = extract_field_report_records(input_path)
    if not records:
        raise ValueError(f"No extracted field-report records found in {input_path}.")

    table = pd.DataFrame([record.__dict__ for record in records])
    table = _deduplicate_records(table, dedupe_precision)

    output_path = Path(output_path)
    ensure_parent_dir(output_path)
    rows = [_queue_row(row) for row in table.to_dict(orient="records")]
    output_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return {
        "llm_review_queue": output_path,
        "rows": len(rows),
        "source_files": int(table["source_file"].nunique()),
    }


def import_llm_reviewed_labels(
    review_path: str | Path,
    output_path: str | Path,
    csv_output: str | Path | None = None,
) -> dict:
    review_path = Path(review_path)
    rows = [_reviewed_record(row) for row in _read_jsonl(review_path)]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError(f"No accepted LLM-reviewed labels found in {review_path}.")

    table = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        table,
        geometry=[Point(lon, lat) for lon, lat in zip(table["longitude"], table["latitude"])],
        crs="EPSG:4326",
    )
    write_vector(gdf, output_path)
    result = {
        "reviewed_labels": Path(output_path),
        "rows": int(len(gdf)),
        "source_files": int(gdf["source_file"].nunique()),
    }
    if csv_output:
        ensure_parent_dir(csv_output)
        gdf.drop(columns="geometry").to_csv(csv_output, index=False)
        result["reviewed_labels_csv"] = Path(csv_output)
    return result


def _queue_row(row: dict[str, Any]) -> dict:
    review_id = _review_id(row)
    return {
        "review_id": review_id,
        "task": "culvert_field_report_label_review",
        "instruction": LLM_REVIEW_INSTRUCTION,
        "source": {
            "source_file": row.get("source_file", ""),
            "report_date": row.get("report_date", ""),
            "context_text": row.get("context_text", ""),
            "raw_coordinate_text": row.get("raw_coordinate_text", ""),
        },
        "extracted": {
            "latitude": _clean_float(row.get("latitude")),
            "longitude": _clean_float(row.get("longitude")),
            "route": _clean_string(row.get("route")),
            "nysdot_region": _clean_string(row.get("nysdot_region")),
            "culvert_id": _clean_string(row.get("culvert_id")),
            "label": _clean_string(row.get("label")) or "field_observed_culvert",
            "label_confidence": _clean_float(row.get("label_confidence")) or 0.85,
        },
        "expected_response_schema": {
            "accepted": "boolean",
            "latitude": "number or null",
            "longitude": "number or null",
            "route": "string",
            "culvert_id": "string",
            "label_confidence": "number from 0 to 1",
            "reason": "short string",
        },
    }


def _reviewed_record(row: dict[str, Any]) -> dict | None:
    extracted = row.get("extracted") if isinstance(row.get("extracted"), dict) else row
    review = _review_payload(row)
    accepted = review.get("accepted", row.get("accepted", True))
    if str(accepted).lower() in {"false", "0", "no", "reject", "rejected"}:
        return None

    latitude = _clean_float(review.get("latitude", extracted.get("latitude")))
    longitude = _clean_float(review.get("longitude", extracted.get("longitude")))
    if latitude is None or longitude is None or not _valid_new_york_lat_lon(latitude, longitude):
        return None

    source = row.get("source") if isinstance(row.get("source"), dict) else row
    confidence = _clean_float(review.get("label_confidence", extracted.get("label_confidence")))
    if confidence is None:
        confidence = 0.9

    return {
        "source_file": _clean_string(source.get("source_file")),
        "report_date": _clean_string(source.get("report_date")),
        "nysdot_region": _clean_string(review.get("nysdot_region", extracted.get("nysdot_region"))),
        "route": _clean_string(review.get("route", extracted.get("route"))),
        "latitude": latitude,
        "longitude": longitude,
        "raw_coordinate_text": _clean_string(source.get("raw_coordinate_text")),
        "culvert_id": _clean_string(review.get("culvert_id", extracted.get("culvert_id"))),
        "context_text": _clean_string(source.get("context_text")),
        "label": _clean_string(review.get("label")) or "llm_reviewed_field_observed_culvert",
        "label_confidence": max(0.0, min(float(confidence), 1.0)),
        "llm_review_id": _clean_string(row.get("review_id")),
        "llm_review_reason": _clean_string(review.get("reason")),
    }


def _review_payload(row: dict[str, Any]) -> dict:
    for key in ("review", "reviewed", "llm_response", "response"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _review_id(row: dict[str, Any]) -> str:
    payload = "|".join(
        str(row.get(key, ""))
        for key in ("source_file", "report_date", "latitude", "longitude", "raw_coordinate_text")
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _clean_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _valid_new_york_lat_lon(latitude: float, longitude: float) -> bool:
    return 39.0 <= latitude <= 45.5 and -80.5 <= longitude <= -70.0
