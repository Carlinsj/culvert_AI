from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from culvert_ai.io import read_vector, write_vector


def merge_confirmed_observations(
    observations_path: str | Path,
    output_path: str | Path,
    base_known_path: str | Path | None = None,
    csv_output: str | Path | None = None,
    confirmed_output_path: str | Path | None = None,
    denied_output_path: str | Path | None = None,
    denied_csv_output: str | Path | None = None,
    include_confirmed: bool = True,
) -> dict:
    observations = read_vector(observations_path)
    base = _read_optional_base(base_known_path)
    confirmed = (
        _confirmed_observations_as_known(observations, base.crs if base is not None else None)
        if include_confirmed
        else None
    )
    denied = _denied_observations_as_negative(observations, base.crs if base is not None else None)

    layers = [layer for layer in [base, confirmed] if layer is not None and not layer.empty]
    if not layers:
        combined = gpd.GeoDataFrame(
            columns=[
                "report_date",
                "route",
                "culvert_id",
                "source_file",
                "label",
                "label_confidence",
                "observation_id",
                "field_culvert_id",
                "layout_source",
                "notes",
                "geometry",
            ],
            geometry="geometry",
            crs=observations.crs,
        )
    else:
        crs = layers[0].crs
        aligned = [layer.to_crs(crs) if layer.crs != crs else layer for layer in layers]
        combined = gpd.GeoDataFrame(pd.concat(aligned, ignore_index=True), geometry="geometry", crs=crs)

    write_vector(combined, output_path)
    if csv_output:
        write_vector(combined, csv_output)
    if confirmed_output_path and confirmed is not None and not confirmed.empty:
        write_vector(confirmed, confirmed_output_path)
    if denied_output_path and denied is not None and not denied.empty:
        write_vector(denied, denied_output_path)
    if denied_csv_output and denied is not None and not denied.empty:
        write_vector(denied, denied_csv_output)

    return {
        "observations": Path(observations_path),
        "base_known": Path(base_known_path) if base_known_path else None,
        "output": Path(output_path),
        "rows": len(combined),
        "confirmed_added": len(confirmed) if confirmed is not None else 0,
        "denied_saved_for_review": len(denied),
        "confirmed_output": Path(confirmed_output_path) if confirmed_output_path else None,
        "denied_output": Path(denied_output_path) if denied_output_path else None,
    }


def _read_optional_base(path: str | Path | None) -> gpd.GeoDataFrame | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return read_vector(path)


def _confirmed_observations_as_known(
    observations: gpd.GeoDataFrame,
    output_crs,
) -> gpd.GeoDataFrame | None:
    if observations.empty or "status" not in observations:
        return None

    confirmed = observations[observations["status"] == "confirmed_culvert"].copy()
    if confirmed.empty:
        return None

    if output_crs is not None and confirmed.crs != output_crs:
        confirmed = confirmed.to_crs(output_crs)

    known = gpd.GeoDataFrame(
        {
            "report_date": _string_series(confirmed, "observed_at").map(_date_part),
            "route": _string_series(confirmed, "road_name"),
            "culvert_id": confirmed.apply(_observation_culvert_id, axis=1),
            "source_file": "field_observations.geojson",
            "label": "confirmed_field_observation",
            "label_confidence": 1.0,
            "observation_id": _string_series(confirmed, "observation_id"),
            "field_culvert_id": _string_series(confirmed, "field_culvert_id"),
            "layout_source": _string_series(confirmed, "layout_source"),
            "notes": _string_series(confirmed, "notes"),
        },
        geometry=confirmed.geometry,
        crs=confirmed.crs,
    )
    return known


def _denied_observations_as_negative(
    observations: gpd.GeoDataFrame,
    output_crs,
) -> gpd.GeoDataFrame:
    if observations.empty or "status" not in observations:
        return _empty_observation_labels(observations.crs)

    denied = observations[observations["status"] == "no_culvert"].copy()
    if denied.empty:
        return _empty_observation_labels(observations.crs)

    if output_crs is not None and denied.crs != output_crs:
        denied = denied.to_crs(output_crs)

    return gpd.GeoDataFrame(
        {
            "report_date": _string_series(denied, "observed_at").map(_date_part),
            "route": _string_series(denied, "road_name"),
            "culvert_id": denied.apply(_observation_culvert_id, axis=1),
            "source_file": "field_observations.geojson",
            "label": "no_culvert",
            "label_confidence": 1.0,
            "observation_id": _string_series(denied, "observation_id"),
            "field_culvert_id": _string_series(denied, "field_culvert_id"),
            "layout_source": _string_series(denied, "layout_source"),
            "candidate_id": _string_series(denied, "candidate_id"),
            "notes": _string_series(denied, "notes"),
        },
        geometry=denied.geometry,
        crs=denied.crs,
    )


def _empty_observation_labels(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "report_date",
            "route",
            "culvert_id",
            "source_file",
            "label",
            "label_confidence",
            "observation_id",
            "field_culvert_id",
            "layout_source",
            "candidate_id",
            "notes",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def _string_series(table: gpd.GeoDataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series([""] * len(table), index=table.index)
    return table[column].fillna("").astype(str)


def _date_part(value) -> str:
    text = str(value or "")
    return text[:10] if text else ""


def _observation_culvert_id(row) -> str:
    field_culvert_id = str(row.get("field_culvert_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    observation_id = str(row.get("observation_id") or "").strip()
    return field_culvert_id or candidate_id or observation_id or "field_observation"
