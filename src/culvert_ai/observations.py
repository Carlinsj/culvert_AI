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
) -> dict:
    observations = read_vector(observations_path)
    base = _read_optional_base(base_known_path)
    confirmed = _confirmed_observations_as_known(observations, base.crs if base is not None else None)

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

    denied = (
        observations[observations["status"] == "no_culvert"]
        if "status" in observations
        else observations.iloc[0:0]
    )
    return {
        "observations": Path(observations_path),
        "base_known": Path(base_known_path) if base_known_path else None,
        "output": Path(output_path),
        "rows": len(combined),
        "confirmed_added": len(confirmed) if confirmed is not None else 0,
        "denied_saved_for_review": len(denied),
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
