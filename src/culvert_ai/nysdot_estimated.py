"""Preserve NYSDOT-provided estimated datasets outside the official-label flow."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


DATA_ROLE = "estimated"
SEGMENT_SOURCE_DATASET = "NYSDOT segment density R8"
POINT_SOURCE_DATASET = "NYSDOT Region 8 density-based points Ulster County"
SEGMENT_SHEET_NAME = "segment_density_r8_u16_frcnn"


def read_segment_density(path: str | Path) -> pd.DataFrame:
    """Read the estimated segment-density worksheet without interpreting it as labels."""

    return pd.read_excel(path, sheet_name=SEGMENT_SHEET_NAME)


def read_density_based_points(path: str | Path) -> pd.DataFrame:
    """Read estimated density-based points without treating their IDs as culvert IDs."""

    return pd.read_csv(path)


def normalize_segment_density(source: pd.DataFrame) -> pd.DataFrame:
    """Add explicit estimated-data provenance while retaining all source fields."""

    _require_columns(
        source,
        {"LINEARID", "road_name", "rttyp", "county", "length_km", "n_small_culverts", "density_per_km"},
    )
    result = source.copy()
    flags = _segment_flags(result)
    result.insert(0, "data_role", DATA_ROLE)
    result.insert(1, "source_dataset", SEGMENT_SOURCE_DATASET)
    result.insert(2, "source_row_id", range(2, len(result) + 2))
    result.insert(3, "segment_id", result["LINEARID"].astype("string"))
    result.insert(4, "route_label", _text(result["road_name"]))
    result.insert(5, "route_type", _text(result["rttyp"]))
    result.insert(6, "estimated_small_culvert_count", result["n_small_culverts"])
    result.insert(7, "estimated_density_per_km", result["density_per_km"])
    result.insert(8, "source_record_quality_flags", [json.dumps(value) for value in flags])
    return result


def normalize_density_based_points(source: pd.DataFrame) -> pd.DataFrame:
    """Add explicit estimated-data provenance while retaining all source fields.

    ``Culvert ID Number`` is deliberately mapped to ``estimated_point_id``. It
    must not be merged into the project's official ``culvert_id`` field.
    """

    _require_columns(source, {"Culvert ID Number", "longitude", "latitude", "county"})
    result = source.copy()
    longitude = pd.to_numeric(result["longitude"], errors="coerce")
    latitude = pd.to_numeric(result["latitude"], errors="coerce")
    valid_coordinates = _finite(longitude) & _finite(latitude)
    flags = [
        _point_flags(
            missing_longitude=not _finite(longitude).iloc[index],
            missing_latitude=not _finite(latitude).iloc[index],
            invalid_longitude=result["longitude"].notna().iloc[index]
            and not _finite(longitude).iloc[index],
            invalid_latitude=result["latitude"].notna().iloc[index]
            and not _finite(latitude).iloc[index],
            duplicate_coordinate=bool(
                valid_coordinates.iloc[index]
                and result.loc[valid_coordinates, ["longitude", "latitude"]]
                .duplicated(keep=False)
                .reindex(result.index, fill_value=False)
                .iloc[index]
            ),
        )
        for index in range(len(result))
    ]
    result.insert(0, "data_role", DATA_ROLE)
    result.insert(1, "source_dataset", POINT_SOURCE_DATASET)
    result.insert(2, "source_row_id", range(2, len(result) + 2))
    result.insert(3, "estimated_point_id", result["Culvert ID Number"].astype("string"))
    result.insert(4, "source_record_quality_flags", [json.dumps(value) for value in flags])
    return result


def write_estimated_nysdot_outputs(
    segment_source: pd.DataFrame, point_source: pd.DataFrame, output_dir: str | Path
) -> dict:
    """Write separate estimated-only Parquet tables and an audit metadata file."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = output_dir / "nysdot_segment_density_r8_estimated.parquet"
    point_path = output_dir / "nysdot_density_based_points_ulster_estimated.parquet"
    metadata_path = output_dir / "nysdot_estimated_inputs_inventory.json"
    segments = normalize_segment_density(segment_source)
    points = normalize_density_based_points(point_source)
    _parquet_safe(segments).to_parquet(segment_path, index=False)
    _parquet_safe(points).to_parquet(point_path, index=False)
    metadata = estimated_inventory_summary(segment_source, point_source)
    metadata["outputs"] = {"segment_density": str(segment_path), "density_based_points": str(point_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def estimated_inventory_summary(segment_source: pd.DataFrame, point_source: pd.DataFrame) -> dict:
    """Return source-only audit metadata for the two estimated datasets."""

    segments = normalize_segment_density(segment_source)
    points = normalize_density_based_points(point_source)
    density_expected = segment_source["n_small_culverts"] / segment_source["length_km"]
    point_coordinates = point_source[["longitude", "latitude"]]
    return {
        "data_role": DATA_ROLE,
        "segment_density": {
            "row_count": len(segments),
            "columns": list(segment_source.columns),
            "coordinate_fields": [],
            "segment_identifier": "LINEARID",
            "route_fields": ["road_name", "rttyp"],
            "density_fields": ["length_km", "n_small_culverts", "density_per_km"],
            "geographic_coverage": {
                "county_count": int(segment_source["county"].nunique()),
                "counties": sorted(segment_source["county"].unique().tolist()),
            },
            "missing_value_count": int(segment_source.isna().sum().sum()),
            "duplicate_segment_identifier_rows": int(segment_source["LINEARID"].duplicated(keep=False).sum()),
            "duplicate_full_rows": int(segment_source.duplicated(keep=False).sum()),
            "density_formula_max_abs_difference": float(
                (segment_source["density_per_km"] - density_expected).abs().max()
            ),
            "target_leakage_risk": "high: n_small_culverts and density_per_km are estimated culvert-count signals.",
        },
        "density_based_points": {
            "row_count": len(points),
            "columns": list(point_source.columns),
            "coordinate_fields": ["longitude", "latitude"],
            "segment_identifier": None,
            "route_fields": [],
            "density_fields": [],
            "geographic_coverage": {
                "county_values": sorted(point_source["county"].unique().tolist()),
                "longitude_range": [float(point_source.longitude.min()), float(point_source.longitude.max())],
                "latitude_range": [float(point_source.latitude.min()), float(point_source.latitude.max())],
            },
            "missing_value_count": int(point_source.isna().sum().sum()),
            "duplicate_estimated_point_id_rows": int(
                point_source["Culvert ID Number"].duplicated(keep=False).sum()
            ),
            "duplicate_coordinate_rows": int(point_coordinates.duplicated(keep=False).sum()),
            "duplicate_full_rows": int(point_source.duplicated(keep=False).sum()),
        },
    }


def _segment_flags(source: pd.DataFrame) -> list[list[str]]:
    duplicate = source["LINEARID"].duplicated(keep=False)
    return [
        [
            "estimated_data",
            "potential_target_leakage",
            *( ["duplicate_segment_identifier"] if duplicate.iloc[index] else []),
        ]
        for index in range(len(source))
    ]


def _point_flags(**flags: bool) -> list[str]:
    result = ["estimated_data", "not_verified_positive", "not_official_culvert_label"]
    result.extend(name for name, enabled in flags.items() if enabled)
    return result


def _finite(values: pd.Series) -> pd.Series:
    return values.notna() & values.map(math.isfinite)


def _text(values: pd.Series) -> pd.Series:
    return values.map(lambda value: "" if pd.isna(value) else str(value).strip()).astype("string")


def _require_columns(source: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Estimated NYSDOT source is missing required columns: {', '.join(missing)}")


def _parquet_safe(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(lambda value: None if pd.isna(value) else str(value))
    return result
