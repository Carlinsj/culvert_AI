import json

import pandas as pd

from culvert_ai.nysdot_estimated import (
    DATA_ROLE,
    estimated_inventory_summary,
    normalize_density_based_points,
    normalize_segment_density,
)


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LINEARID": [1, 1],
            "road_name": ["State Rte 28", "State Rte 28"],
            "rttyp": ["S", "S"],
            "county": ["Ulster County", "Ulster County"],
            "length_km": [2.0, 4.0],
            "n_small_culverts": [3, 4],
            "density_per_km": [1.5, 1.0],
        }
    )


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Culvert ID Number": [71, 72],
            "longitude": [-74.0, -74.0],
            "latitude": [42.0, 42.0],
            "county": ["Ulster", "Ulster"],
        }
    )


def test_segment_normalization_explicitly_marks_estimated_target_leakage():
    normalized = normalize_segment_density(_segments())

    assert normalized["data_role"].tolist() == [DATA_ROLE, DATA_ROLE]
    assert normalized["segment_id"].tolist() == ["1", "1"]
    assert normalized["estimated_small_culvert_count"].tolist() == [3, 4]
    assert "potential_target_leakage" in json.loads(normalized.loc[0, "source_record_quality_flags"])


def test_point_normalization_never_promotes_estimated_ids_to_culvert_ids():
    normalized = normalize_density_based_points(_points())

    assert "culvert_id" not in normalized.columns
    assert normalized["estimated_point_id"].tolist() == ["71", "72"]
    flags = json.loads(normalized.loc[0, "source_record_quality_flags"])
    assert {"estimated_data", "not_verified_positive", "not_official_culvert_label", "duplicate_coordinate"}.issubset(flags)


def test_inventory_summary_reports_density_formula_and_coverage():
    summary = estimated_inventory_summary(_segments(), _points())

    assert summary["data_role"] == DATA_ROLE
    assert summary["segment_density"]["density_formula_max_abs_difference"] == 0.0
    assert summary["segment_density"]["duplicate_segment_identifier_rows"] == 2
    assert summary["density_based_points"]["duplicate_coordinate_rows"] == 2
