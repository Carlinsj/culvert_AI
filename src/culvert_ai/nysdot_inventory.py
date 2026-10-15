"""Ingest the NYSDOT small-culvert inventory without assigning training labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


SOURCE_DATASET = "NYSDOT Small Culverts R8 2026.07.14"
DEFAULT_SHEET_NAME = "Small Culverts - R8"
SOURCE_CRS = "EPSG:26918"
SOURCE_CRS_NAME = "NAD83 / UTM Zone 18N"
SOURCE_CRS_UNITS = "meters"
ULSTER_HUDSON_VALLEY_BOUNDS = {
    "min_longitude": -75.0,
    "max_longitude": -73.5,
    "min_latitude": 41.3,
    "max_latitude": 42.4,
}
SOURCE_FIELDS = {
    "culvert_id": "Small Culvert ID",
    "legacy_culvert_id": "Legacy Culvert ID",
    "region": "DOT Region",
    "county": "County",
    "route": "Route",
    "x": "x",
    "y": "y",
    "inspection_2026_inspector_name": "2026 Inspector Name",
    "inspection_2026_date": "2026 Inspection Date",
    "inspection_2026_comments": "2026 Inspection Comments",
    "user_updated": "User Updated",
    "date_updated": "Date Updated",
    "last_edited_user": "last_edited_user",
    "last_edited_date": "last_edited_date",
}
UPDATE_DATE_COLUMNS = ("Date Updated", "last_edited_date")
ACTIVITY_DATE_COLUMNS = ("2026 Inspection Date", *UPDATE_DATE_COLUMNS)


def read_nysdot_small_culverts(
    workbook_path: str | Path, sheet_name: str = DEFAULT_SHEET_NAME
) -> pd.DataFrame:
    """Read the source sheet as source values, keeping all original columns."""

    return pd.read_excel(workbook_path, sheet_name=sheet_name)


def inventory_sheet_names(workbook_path: str | Path) -> list[str]:
    """Return workbook sheet names without selecting or changing a sheet."""

    return list(pd.ExcelFile(workbook_path).sheet_names)


def normalize_nysdot_small_culverts(
    source: pd.DataFrame,
    *,
    source_dataset: str = SOURCE_DATASET,
    source_crs: str = SOURCE_CRS,
) -> gpd.GeoDataFrame:
    """Add stable inventory fields while retaining every source row and column.

    The normalizer never deduplicates records. Source ``x`` and ``y`` columns
    remain unchanged; ``source_x_raw`` and ``source_y_raw`` provide explicit
    provenance copies alongside native EPSG:26918 geometry.
    """

    _require_columns(source, SOURCE_FIELDS.values())
    if source_crs != SOURCE_CRS:
        raise ValueError(f"Expected authoritative NYSDOT source CRS {SOURCE_CRS}, got {source_crs}.")
    source_table = source.copy()
    x_raw = source_table["x"].copy()
    y_raw = source_table["y"].copy()
    x = pd.to_numeric(x_raw, errors="coerce")
    y = pd.to_numeric(y_raw, errors="coerce")
    valid_x = _finite(x)
    valid_y = _finite(y)
    valid_pair = valid_x & valid_y

    culvert_id = _identifier(source_table["Small Culvert ID"])
    duplicate_id = culvert_id.ne("") & culvert_id.duplicated(keep=False)
    coordinate_key = pd.Series("", index=source_table.index, dtype="string")
    coordinate_key.loc[valid_pair] = [
        f"{x_value:.10f}|{y_value:.10f}"
        for x_value, y_value in zip(x.loc[valid_pair], y.loc[valid_pair])
    ]
    duplicate_coordinate = coordinate_key.ne("") & coordinate_key.duplicated(keep=False)

    parsed_dates = {
        column: _parse_date(source_table[column]) for column in ACTIVITY_DATE_COLUMNS
    }
    latest_update_date, latest_update_fields = _latest_date_and_fields(
        parsed_dates, UPDATE_DATE_COLUMNS
    )
    latest_activity_date, latest_activity_fields = _latest_date_and_fields(
        parsed_dates, ACTIVITY_DATE_COLUMNS
    )

    quality_flags = [
        _quality_flags(
            missing_culvert_id=culvert_id.iloc[position] == "",
            missing_x=not valid_x.iloc[position],
            missing_y=not valid_y.iloc[position],
            invalid_x=x_raw.notna().iloc[position] and not valid_x.iloc[position],
            invalid_y=y_raw.notna().iloc[position] and not valid_y.iloc[position],
            duplicate_culvert_id=duplicate_id.iloc[position],
            duplicate_coordinate=duplicate_coordinate.iloc[position],
        )
        for position in range(len(source_table))
    ]

    normalized = source_table.copy()
    normalized.insert(0, "source_dataset", source_dataset)
    normalized.insert(1, "source_row_id", range(2, len(source_table) + 2))
    normalized.insert(2, "culvert_id", culvert_id)
    normalized.insert(3, "legacy_culvert_id", _identifier(source_table["Legacy Culvert ID"]))
    normalized.insert(4, "region", _text(source_table["DOT Region"]))
    normalized.insert(5, "county", _text(source_table["County"]))
    normalized.insert(6, "route", _text(source_table["Route"]))
    normalized.insert(7, "source_x_raw", x_raw)
    normalized.insert(8, "source_y_raw", y_raw)
    normalized.insert(9, "source_crs", source_crs)
    normalized.insert(10, "source_crs_name", SOURCE_CRS_NAME)
    normalized.insert(11, "source_crs_units", SOURCE_CRS_UNITS)
    normalized.insert(12, "inspection_2026_inspector_name", _text(source_table["2026 Inspector Name"]))
    normalized.insert(13, "inspection_2026_date", parsed_dates["2026 Inspection Date"])
    normalized.insert(14, "inspection_2026_comments", _text(source_table["2026 Inspection Comments"]))
    normalized.insert(15, "latest_update_date", latest_update_date)
    normalized.insert(16, "latest_update_source_fields", latest_update_fields)
    normalized.insert(17, "latest_source_activity_date", latest_activity_date)
    normalized.insert(18, "latest_source_activity_fields", latest_activity_fields)
    normalized.insert(19, "source_record_quality_flags", [json.dumps(flags) for flags in quality_flags])
    native_geometry = [
        Point(x_value, y_value) if is_valid else None
        for x_value, y_value, is_valid in zip(x, y, valid_pair)
    ]
    native = gpd.GeoDataFrame(normalized, geometry=native_geometry, crs=source_crs)
    wgs84 = native.to_crs("EPSG:4326")
    native["longitude"] = wgs84.geometry.x
    native["latitude"] = wgs84.geometry.y
    _validate_transformation(native, valid_pair)
    return native


def inventory_summary(source: pd.DataFrame) -> dict:
    """Summarize workbook content and source-quality findings without mutation."""

    normalized = normalize_nysdot_small_culverts(source)
    flags = normalized["source_record_quality_flags"].map(json.loads)
    return {
        "row_count": len(normalized),
        "column_count": len(source.columns),
        "columns": list(source.columns),
        "culvert_id_fields": ["Small Culvert ID", "Legacy Culvert ID"],
        "county_field": "County",
        "region_field": "DOT Region",
        "route_field": "Route",
        "coordinate_fields": ["x", "y"],
        "duplicate_culvert_id_rows": _flag_count(flags, "duplicate_culvert_id"),
        "duplicate_culvert_id_groups": int(
            normalized.loc[
                flags.map(lambda values: "duplicate_culvert_id" in values), "culvert_id"
            ].nunique()
        ),
        "duplicate_coordinate_rows": _flag_count(flags, "duplicate_coordinate"),
        "duplicate_coordinate_groups": int(
            normalized.loc[
                flags.map(lambda values: "duplicate_coordinate" in values), ["x", "y"]
            ].drop_duplicates().shape[0]
        ),
        "missing_x": _flag_count(flags, "missing_x"),
        "missing_y": _flag_count(flags, "missing_y"),
        "invalid_x": _flag_count(flags, "invalid_x"),
        "invalid_y": _flag_count(flags, "invalid_y"),
        "crs": source_crs_summary(),
        "transformed_wgs84_bounds": _wgs84_bounds(normalized),
        "last_ten_columns": last_ten_column_summary(source),
    }


def last_ten_column_summary(source: pd.DataFrame) -> list[dict]:
    """Report completeness and date coverage of the source sheet's last ten fields."""

    records: list[dict] = []
    for column in source.columns[-10:]:
        values = source[column]
        record = {
            "field": column,
            "non_null_count": int(values.notna().sum()),
            "missing_count": int(values.isna().sum()),
            "appears_to_represent": _field_meaning(column),
        }
        if column in {"2026 Inspection Date", "Date Updated", "last_edited_date"}:
            dates = _parse_date(values)
            record.update(
                {
                    "parseable_date_count": int(dates.notna().sum()),
                    "unparseable_non_null_count": int((values.notna() & dates.isna()).sum()),
                    "date_min": _date_string(dates.min()),
                    "date_max": _date_string(dates.max()),
                }
            )
        records.append(record)
    return records


def source_crs_summary() -> dict:
    """State the authoritative CRS supplied for the NYSDOT source dataset."""

    return {
        "confirmed": True,
        "value": SOURCE_CRS,
        "name": SOURCE_CRS_NAME,
        "units": SOURCE_CRS_UNITS,
        "conclusion": "Authoritative NYSDOT Environmental Viewer Directory metadata.",
    }


def write_nysdot_inventory_outputs(
    source: pd.DataFrame,
    output_dir: str | Path,
    *,
    source_dataset: str = SOURCE_DATASET,
    source_crs: str = SOURCE_CRS,
) -> dict:
    """Write full Region 8 and Ulster subsets as native-EPSG:26918 GeoParquet."""
    normalized = normalize_nysdot_small_culverts(
        source, source_dataset=source_dataset, source_crs=source_crs
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    region_path = output_dir / "nysdot_small_culverts_r8_2026_07_14.geoparquet"
    ulster_path = output_dir / "nysdot_small_culverts_r8_ulster_2026_07_14.geoparquet"
    metadata_path = output_dir / "nysdot_small_culverts_r8_2026_07_14_inventory.json"
    geoparquet_table = _parquet_safe(normalized)
    geoparquet_table.to_parquet(region_path, index=False)
    ulster = geoparquet_table.loc[geoparquet_table["county"].str.upper().eq("ULSTER")].copy()
    _validate_ulster_bounds(ulster)
    ulster.to_parquet(ulster_path, index=False)
    summary = inventory_summary(source)
    summary["source_dataset"] = source_dataset
    summary["ulster_transformed_wgs84_bounds"] = _wgs84_bounds(ulster)
    summary["invalid_transformations"] = 0
    summary["outputs"] = {"region_8": str(region_path), "ulster": str(ulster_path)}
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _latest_date_and_fields(
    parsed_dates: dict[str, pd.Series], columns: tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    dates = pd.DataFrame({column: parsed_dates[column] for column in columns})
    latest = dates.max(axis=1)
    fields = dates.apply(
        lambda row: "|".join(column for column in columns if pd.notna(latest[row.name]) and row[column] == latest[row.name]),
        axis=1,
    )
    return latest, fields.astype("string")


def _finite(values: pd.Series) -> pd.Series:
    return values.notna() & values.map(math.isfinite)


def _identifier(values: pd.Series) -> pd.Series:
    result = values.map(lambda value: "" if pd.isna(value) else str(value).strip())
    return result.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True).astype("string")


def _text(values: pd.Series) -> pd.Series:
    return values.map(lambda value: "" if pd.isna(value) else str(value).strip()).astype("string")


def _parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _parquet_safe(table: pd.DataFrame) -> pd.DataFrame:
    """Make heterogeneous Excel object columns representable without dropping values."""

    result = table.copy()
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(lambda value: None if pd.isna(value) else str(value))
    return result


def _validate_transformation(points: gpd.GeoDataFrame, valid_pair: pd.Series) -> None:
    """Reject a failed native-to-WGS84 transformation instead of masking it."""

    expected_count = int(valid_pair.sum())
    transformed = points.loc[valid_pair, ["longitude", "latitude"]]
    valid_wgs84 = (
        transformed["longitude"].map(math.isfinite)
        & transformed["latitude"].map(math.isfinite)
        & transformed["longitude"].between(-180, 180)
        & transformed["latitude"].between(-90, 90)
    )
    if int(valid_wgs84.sum()) != expected_count:
        raise ValueError(
            f"EPSG:26918 to EPSG:4326 transformation failed for {expected_count - int(valid_wgs84.sum())} records."
        )


def _validate_ulster_bounds(points: gpd.GeoDataFrame) -> None:
    bounds = ULSTER_HUDSON_VALLEY_BOUNDS
    plausible = (
        points["longitude"].between(bounds["min_longitude"], bounds["max_longitude"])
        & points["latitude"].between(bounds["min_latitude"], bounds["max_latitude"])
    )
    invalid = int((~plausible).sum())
    if invalid:
        raise ValueError(f"{invalid} Ulster records fall outside plausible Hudson Valley bounds after transformation.")


def _wgs84_bounds(points: pd.DataFrame) -> dict:
    return {
        "min_longitude": float(points["longitude"].min()),
        "min_latitude": float(points["latitude"].min()),
        "max_longitude": float(points["longitude"].max()),
        "max_latitude": float(points["latitude"].max()),
    }


def _quality_flags(**flags: bool) -> list[str]:
    return [name for name, enabled in flags.items() if enabled]


def _flag_count(flags: pd.Series, flag: str) -> int:
    return int(flags.map(lambda values: flag in values).sum())


def _date_string(value) -> str | None:
    return None if pd.isna(value) else value.isoformat()


def _require_columns(source: pd.DataFrame, required: object) -> None:
    missing = sorted(set(required).difference(source.columns))
    if missing:
        raise ValueError(f"NYSDOT workbook is missing required columns: {', '.join(missing)}")


def _field_meaning(column: str) -> str:
    meanings = {
        "2026 Inspector Name": "Name recorded for the 2026 inspection.",
        "2026 Inspection Date": "Timestamp recorded for the 2026 inspection.",
        "2026 Inspection Comments": "Free-text observations from the 2026 inspection.",
        "Photo URL": "Link or path to a related photo resource.",
        "User Updated": "User account recorded for an update.",
        "Date Updated": "Timestamp recorded for an update.",
        "last_edited_user": "User account recorded by a last-edit tracking field.",
        "last_edited_date": "Date recorded by a last-edit tracking field.",
        "x": "Source x coordinate in EPSG:26918 (NAD83 / UTM Zone 18N), meters.",
        "y": "Source y coordinate in EPSG:26918 (NAD83 / UTM Zone 18N), meters.",
    }
    return meanings.get(column, "Source field.")
