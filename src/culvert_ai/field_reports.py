from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from culvert_ai.io import (
    add_wgs84_coordinates,
    clean_geometry,
    ensure_parent_dir,
    read_vector,
    write_vector,
)


COORDINATE_RE = re.compile(
    r"(?P<a>[+-]?\d{1,3}\.\d+)\s*°?\s*(?P<adir>[NSEW])?\s*[,;/\s]+"
    r"(?P<b>[+-]?\d{1,3}\.\d+)\s*°?\s*(?P<bdir>[NSEW])?",
    re.IGNORECASE,
)
ROUTE_RE = re.compile(
    r"\b(?:NY|US|I|CR|Co\s*Rd|County\s*(?:Road|Route)|State\s*(?:Rte|Route))\s*-?\s*"
    r"\d+[A-Za-z]?\b",
    re.IGNORECASE,
)
REGION_ROUTE_RE = re.compile(
    r"\b(?P<region>[1-9])\s+(?P<route>(?:NY|US|I|CR)\s*-?\s*\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)
CULVERT_ID_RE = re.compile(
    r"\b(?P<prefix>SC|CID)\s*[-:]?\s*(?P<value>\d+|NOT\s+ASSIGNED)\b",
    re.IGNORECASE,
)
NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[._/-](\d{1,2})[._/-](\d{2}|\d{4})\b")
MONTH_DATE_RE = re.compile(
    r"\b(?P<month>"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?"
    r")\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{2}|\d{4})\b",
    re.IGNORECASE,
)
DAY_MONTH_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?"
    r")\.?[,]?\s+(?P<year>\d{2}|\d{4})\b",
    re.IGNORECASE,
)
BARE_ROUTE_TABLE_RE = re.compile(
    r"^\s*(?:\d+\s+)?R?\s*(?P<region>[1-9])\s+(?P<route>\d{1,3}[A-Za-z]?)\b",
    re.IGNORECASE,
)
MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class CoordinateRecord:
    source_file: str
    report_date: str
    nysdot_region: str
    route: str
    latitude: float
    longitude: float
    raw_coordinate_text: str
    culvert_id: str
    context_text: str = ""
    label: str = "field_observed_culvert"
    label_confidence: float = 0.85


def import_field_reports(
    input_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    csv_output: str | Path | None = None,
    dedupe_precision: int = 6,
) -> dict:
    records = extract_field_report_records(input_path)
    if not records:
        raise ValueError(f"No usable latitude/longitude records found in {input_path}.")

    table = pd.DataFrame([record.__dict__ for record in records])
    table = _deduplicate_records(table, dedupe_precision)
    gdf = gpd.GeoDataFrame(
        table,
        geometry=[Point(lon, lat) for lon, lat in zip(table["longitude"], table["latitude"])],
        crs="EPSG:4326",
    )

    write_vector(gdf, output_path)
    result = {
        "field_report_points": Path(output_path),
        "rows": int(len(gdf)),
        "source_files": int(gdf["source_file"].nunique()),
    }

    if csv_output:
        ensure_parent_dir(csv_output)
        gdf.drop(columns="geometry").to_csv(csv_output, index=False)
        result["field_report_points_csv"] = Path(csv_output)

    return result


def append_field_report_candidates(
    candidates_path: str | Path,
    field_reports_path: str | Path,
    output_path: str | Path,
    boundary_path: str | Path | None = None,
) -> dict:
    candidates = read_vector(candidates_path)
    field_points = read_vector(field_reports_path).to_crs(candidates.crs)

    if boundary_path:
        boundary = read_vector(boundary_path).to_crs(candidates.crs)
        field_points = gpd.clip(field_points, boundary)

    field_points = clean_geometry(field_points)
    if field_points.empty:
        write_vector(candidates, output_path)
        return {
            "candidates": Path(output_path),
            "base_rows": int(len(candidates)),
            "field_report_rows": 0,
            "rows": int(len(candidates)),
        }

    first_field_index = _next_field_candidate_index(candidates)
    field_candidates = gpd.GeoDataFrame(
        {
            "candidate_id": [
                f"field_{first_field_index + index:06d}" for index in range(len(field_points))
            ],
            "road_id": field_points.get("route", pd.Series("", index=field_points.index)).fillna(""),
            "stream_id": field_points.get("culvert_id", pd.Series("", index=field_points.index)).fillna(""),
            "road_name": field_points.get("route", pd.Series("Field report", index=field_points.index))
            .fillna("Field report")
            .replace("", "Field report"),
            "stream_name": field_points.get("culvert_id", pd.Series("", index=field_points.index))
            .fillna("")
            .replace("", "Field observed culvert"),
            "source": "field_report_observed_culvert",
            "road_stream_distance_m": np.nan,
            "crossing_angle_degrees": np.nan,
            "field_report_source_file": field_points.get(
                "source_file",
                pd.Series("", index=field_points.index),
            ).fillna(""),
            "field_report_date": field_points.get(
                "report_date",
                pd.Series("", index=field_points.index),
            ).fillna(""),
            "geometry": field_points.geometry.values,
        },
        geometry="geometry",
        crs=candidates.crs,
    )
    field_candidates = add_wgs84_coordinates(field_candidates)

    merged = gpd.GeoDataFrame(
        pd.concat([candidates, field_candidates], ignore_index=True),
        geometry="geometry",
        crs=candidates.crs,
    )
    write_vector(merged, output_path)
    return {
        "candidates": Path(output_path),
        "base_rows": int(len(candidates)),
        "field_report_rows": int(len(field_candidates)),
        "rows": int(len(merged)),
    }


def _next_field_candidate_index(candidates: gpd.GeoDataFrame) -> int:
    if "candidate_id" not in candidates.columns:
        return 1

    values = (
        candidates["candidate_id"]
        .fillna("")
        .astype(str)
        .str.extract(r"^field_(\d+)$", expand=False)
    )
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 1
    return int(numeric.max()) + 1


def extract_field_report_records(input_path: str | Path | Sequence[str | Path]) -> list[CoordinateRecord]:
    if not isinstance(input_path, (str, Path)):
        records: list[CoordinateRecord] = []
        for path in input_path:
            records.extend(extract_field_report_records(path))
        return records

    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Field report path not found: {input_path}")

    if input_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="culvert-field-reports-") as tmp:
            with ZipFile(input_path) as archive:
                archive.extractall(tmp)
            return _extract_from_files(_report_files(Path(tmp)))

    if input_path.is_dir():
        return _extract_from_files(_report_files(input_path))

    return _extract_from_files([input_path])


def _report_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".pdf", ".docx"} and not path.name.startswith("~$")
    )


def _extract_from_files(paths: list[Path]) -> list[CoordinateRecord]:
    records: list[CoordinateRecord] = []
    for path in paths:
        text = _normalize_extracted_text(_extract_text(path))
        records.extend(_records_from_text(path, text))
    return records


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx_text(path)
    if suffix == ".pdf":
        return _pdf_text(path)
    raise ValueError(f"Unsupported field report format: {path}")


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []

    for row in root.iter(ns + "tr"):
        cells = []
        for cell in row.iter(ns + "tc"):
            cell_text = " ".join(text.text or "" for text in cell.iter(ns + "t")).strip()
            if cell_text:
                cells.append(cell_text)
        if cells:
            lines.append(" ".join(cells))

    for paragraph in root.iter(ns + "p"):
        paragraph_text = " ".join(text.text or "" for text in paragraph.iter(ns + "t")).strip()
        if paragraph_text:
            lines.append(paragraph_text)

    return "\n".join(lines)


def _pdf_text(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext is required to import PDF field reports.")

    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _records_from_text(path: Path, text: str) -> list[CoordinateRecord]:
    report_date = _report_date(path)
    records: list[CoordinateRecord] = []
    nearby_culvert_id = ""
    nearby_culvert_lines = 0

    for line in text.splitlines():
        line = _clean_text_line(line)
        if not line:
            continue

        line_culvert_ids = _culvert_ids(line)
        if line_culvert_ids:
            nearby_culvert_id = line_culvert_ids[0]
            nearby_culvert_lines = 3

        for match in COORDINATE_RE.finditer(line):
            lat_lon = _infer_lat_lon(match)
            if not lat_lon:
                continue

            latitude, longitude = lat_lon
            route = _route_for_line(line, match.start())
            region = _region_for_line(line, route)
            culvert_id = line_culvert_ids[0] if line_culvert_ids else nearby_culvert_id
            records.append(
                CoordinateRecord(
                    source_file=path.name,
                    report_date=report_date,
                    nysdot_region=region,
                    route=route,
                    latitude=latitude,
                    longitude=longitude,
                    raw_coordinate_text=match.group(0),
                    culvert_id=culvert_id,
                    context_text=line,
                )
            )

        if not line_culvert_ids and nearby_culvert_lines > 0:
            nearby_culvert_lines -= 1
            if nearby_culvert_lines == 0:
                nearby_culvert_id = ""

    return records


def _infer_lat_lon(match: re.Match) -> tuple[float, float] | None:
    a = _signed_coordinate(match.group("a"), match.group("adir"))
    b = _signed_coordinate(match.group("b"), match.group("bdir"))

    if _valid_new_york_lat_lon(a, b):
        return a, b
    if _valid_new_york_lat_lon(b, a):
        return b, a
    return None


def _signed_coordinate(value: str, direction: str | None) -> float:
    coordinate = float(value)
    if direction and direction.upper() in {"S", "W"}:
        coordinate = -abs(coordinate)
    return coordinate


def _valid_new_york_lat_lon(latitude: float, longitude: float) -> bool:
    return 39.0 <= latitude <= 45.5 and -80.5 <= longitude <= -70.0


def _route_for_line(line: str, coordinate_start: int) -> str:
    before_coordinate = line[:coordinate_start]
    match = ROUTE_RE.search(before_coordinate) or ROUTE_RE.search(line)
    if match:
        return _normalize_route(match.group(0))

    bare_match = BARE_ROUTE_TABLE_RE.search(before_coordinate)
    if bare_match:
        return _normalize_route(f"NY{bare_match.group('route')}")

    return ""


def _region_for_line(line: str, route: str) -> str:
    match = REGION_ROUTE_RE.search(line)
    if match:
        return match.group("region")

    route_region = re.search(r"\bR\s*[-:]?\s*(?P<region>[1-9])\b", line, re.IGNORECASE)
    if route_region:
        return route_region.group("region")

    bare_route = BARE_ROUTE_TABLE_RE.search(line)
    if bare_route:
        return bare_route.group("region")

    if route:
        prefix = route.split()[0]
        loose = re.search(rf"\b(?P<region>[1-9])\s+{re.escape(prefix)}\b", line, re.IGNORECASE)
        if loose:
            return loose.group("region")
    return ""


def _normalize_route(route: str) -> str:
    normalized = re.sub(r"\s+", " ", route.strip().upper())
    normalized = normalized.replace("STATE RTE", "NY")
    normalized = normalized.replace("STATE ROUTE", "NY")
    normalized = normalized.replace("COUNTY ROAD", "CR")
    normalized = normalized.replace("COUNTY ROUTE", "CR")
    normalized = normalized.replace("CO RD", "CR")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "")
    return normalized


def _culvert_ids(text: str) -> list[str]:
    ids = []
    for match in CULVERT_ID_RE.finditer(text):
        prefix = match.group("prefix").upper()
        value = re.sub(r"\s+", "", match.group("value").upper())
        ids.append(f"{prefix}-{value}" if value == "NOTASSIGNED" else f"{prefix}{value}")
    return ids


def _report_date(path: Path) -> str:
    numeric_match = NUMERIC_DATE_RE.search(path.name)
    if numeric_match:
        month, day, year = numeric_match.groups()
        return _format_date(month=int(month), day=int(day), year=int(year))

    month_match = MONTH_DATE_RE.search(path.name)
    if month_match:
        month_name = month_match.group("month").lower().rstrip(".")
        month = MONTH_LOOKUP.get(month_name)
        if month:
            return _format_date(
                month=month,
                day=int(month_match.group("day")),
                year=int(month_match.group("year")),
            )

    day_month_match = DAY_MONTH_DATE_RE.search(path.name)
    if day_month_match:
        month_name = day_month_match.group("month").lower().rstrip(".")
        month = MONTH_LOOKUP.get(month_name)
        if month:
            return _format_date(
                month=month,
                day=int(day_month_match.group("day")),
                year=int(day_month_match.group("year")),
            )

    return ""


def _normalize_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = []
    for character in normalized:
        if character in "\r\n":
            cleaned.append(character)
        elif unicodedata.category(character) == "Cf":
            cleaned.append(" ")
        else:
            cleaned.append(character)
    return "".join(cleaned).replace("\xa0", " ")


def _clean_text_line(line: str) -> str:
    return re.sub(r"\s+", " ", _normalize_extracted_text(line)).strip()


def _format_date(month: int, day: int, year: int) -> str:
    if year < 100:
        year += 2000
    try:
        parsed = date(year, month, day)
    except ValueError:
        return ""
    return parsed.isoformat()


def _deduplicate_records(table: pd.DataFrame, precision: int) -> pd.DataFrame:
    deduped = table.copy()
    deduped["_lat_key"] = deduped["latitude"].round(precision)
    deduped["_lon_key"] = deduped["longitude"].round(precision)
    deduped["_has_culvert_id"] = deduped["culvert_id"].fillna("").astype(str).str.strip().ne("")
    deduped["_has_route"] = deduped["route"].fillna("").astype(str).str.strip().ne("")
    deduped["_context_len"] = deduped["context_text"].fillna("").astype(str).str.len()

    grouped = (
        deduped.sort_values(
            [
                "_lat_key",
                "_lon_key",
                "_has_culvert_id",
                "_has_route",
                "_context_len",
                "report_date",
                "source_file",
            ],
            ascending=[True, True, False, False, False, True, True],
        )
        .groupby(["_lat_key", "_lon_key"], as_index=False)
        .agg(
            {
                "source_file": lambda values: "; ".join(dict.fromkeys(values)),
                "report_date": _first_non_empty,
                "nysdot_region": _first_non_empty,
                "route": _first_non_empty,
                "latitude": "first",
                "longitude": "first",
                "raw_coordinate_text": "first",
                "culvert_id": _first_non_empty,
                "context_text": "first",
                "label": "first",
                "label_confidence": "max",
            }
        )
    )
    return grouped.drop(columns=["_lat_key", "_lon_key"])


def _first_non_empty(values: pd.Series) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
