import json

import pandas as pd

from culvert_ai.nysdot_inventory import (
    SOURCE_CRS,
    ULSTER_HUDSON_VALLEY_BOUNDS,
    _validate_ulster_bounds,
    inventory_summary,
    last_ten_column_summary,
    normalize_nysdot_small_culverts,
)


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Small Culvert ID": [101, 101, None],
            "Legacy Culvert ID": ["OLD-1", "OLD-2", None],
            "Route": ["NY28", "NY28", "NY209"],
            "DOT Region": ["R8 - Poughkeepsie"] * 3,
            "County": ["ULSTER", "ULSTER", "DUTCHESS"],
            "2026 Inspector Name": ["A", None, None],
            "2026 Inspection Date": ["2026-07-14 09:00", None, None],
            "2026 Inspection Comments": ["Clear", None, None],
            "Photo URL": [None, None, None],
            "User Updated": ["user-a", "user-b", None],
            "Date Updated": ["2026-07-13 10:00", "2026-07-12 10:00", None],
            "last_edited_user": [None, "editor-b", None],
            "last_edited_date": [None, "2026-07-14", None],
            "x": [568000.0, 568000.0, "not-a-number"],
            "y": [4648000.0, 4648000.0, 4647000.0],
        }
    )


def test_normalizer_preserves_duplicate_source_records_and_provenance():
    normalized = normalize_nysdot_small_culverts(_source_rows())

    assert len(normalized) == 3
    assert normalized["source_row_id"].tolist() == [2, 3, 4]
    assert normalized["culvert_id"].tolist() == ["101", "101", ""]
    assert normalized["x"].tolist() == [568000.0, 568000.0, "not-a-number"]
    assert normalized["Small Culvert ID"].tolist()[:2] == [101.0, 101.0]
    first_flags = json.loads(normalized.loc[0, "source_record_quality_flags"])
    third_flags = json.loads(normalized.loc[2, "source_record_quality_flags"])
    assert {"duplicate_culvert_id", "duplicate_coordinate"}.issubset(first_flags)
    assert {"missing_culvert_id", "missing_x", "invalid_x"}.issubset(third_flags)
    assert normalized.crs.to_string() == SOURCE_CRS
    assert normalized.loc[0, "source_x_raw"] == normalized.loc[0, "x"]
    assert normalized.loc[0, "source_y_raw"] == normalized.loc[0, "y"]


def test_latest_dates_are_provenance_based_not_invented_statuses():
    normalized = normalize_nysdot_small_culverts(_source_rows())

    assert normalized.loc[0, "latest_update_date"] == pd.Timestamp("2026-07-13 10:00")
    assert normalized.loc[0, "latest_update_source_fields"] == "Date Updated"
    assert normalized.loc[0, "latest_source_activity_date"] == pd.Timestamp("2026-07-14 09:00")
    assert normalized.loc[0, "latest_source_activity_fields"] == "2026 Inspection Date"
    assert "status" not in normalized.columns


def test_epsg_26918_transformation_yields_valid_wgs84_ulster_coordinates():
    normalized = normalize_nysdot_small_culverts(_source_rows())
    ulster = normalized.loc[normalized["county"].eq("ULSTER")].copy()

    assert ulster["longitude"].between(-180, 180).all()
    assert ulster["latitude"].between(-90, 90).all()
    assert ulster["longitude"].between(
        ULSTER_HUDSON_VALLEY_BOUNDS["min_longitude"],
        ULSTER_HUDSON_VALLEY_BOUNDS["max_longitude"],
    ).all()
    assert ulster["latitude"].between(
        ULSTER_HUDSON_VALLEY_BOUNDS["min_latitude"],
        ULSTER_HUDSON_VALLEY_BOUNDS["max_latitude"],
    ).all()
    _validate_ulster_bounds(ulster)


def test_summary_and_last_ten_report_coordinate_and_date_completeness():
    source = _source_rows()
    summary = inventory_summary(source)
    by_field = {record["field"]: record for record in last_ten_column_summary(source)}

    assert summary["duplicate_culvert_id_rows"] == 2
    assert summary["duplicate_coordinate_rows"] == 2
    assert summary["invalid_x"] == 1
    assert summary["missing_x"] == 1
    assert summary["crs"]["confirmed"] is True
    assert summary["crs"]["value"] == SOURCE_CRS
    assert by_field["2026 Inspection Date"]["date_min"] == "2026-07-14T09:00:00"
    assert by_field["Date Updated"]["non_null_count"] == 2
