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
    miss_threshold_m: float = 10.0,
) -> dict:
    observations = read_vector(observations_path)
    base = _read_optional_base(base_known_path)
    confirmed = (
        _confirmed_observations_as_known(observations, base.crs if base is not None else None)
        if include_confirmed
        else None
    )
    denied = _field_negative_observations(
        observations,
        base.crs if base is not None else None,
        miss_threshold_m=miss_threshold_m,
    )

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


def _field_negative_observations(
    observations: gpd.GeoDataFrame,
    output_crs,
    miss_threshold_m: float,
) -> gpd.GeoDataFrame:
    if observations.empty or "status" not in observations:
        return _empty_observation_labels(observations.crs)

    denied = _denied_observations_as_negative(observations)
    missed = _missed_predictions_as_negative(observations, miss_threshold_m=miss_threshold_m)
    negatives = [layer for layer in [denied, missed] if not layer.empty]
    if not negatives:
        return _empty_observation_labels(observations.crs)

    combined = gpd.GeoDataFrame(
        pd.concat(negatives, ignore_index=True),
        geometry="geometry",
        crs=negatives[0].crs,
    )
    if output_crs is not None and combined.crs != output_crs:
        combined = combined.to_crs(output_crs)
    return combined


def _denied_observations_as_negative(observations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    denied = observations[observations["status"] == "no_culvert"].copy()
    if denied.empty:
        return _empty_observation_labels(observations.crs)

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
            "miss_distance_m": pd.Series([pd.NA] * len(denied), index=denied.index),
            "notes": _string_series(denied, "notes"),
        },
        geometry=denied.geometry,
        crs=denied.crs,
    )


def _missed_predictions_as_negative(
    observations: gpd.GeoDataFrame,
    miss_threshold_m: float,
) -> gpd.GeoDataFrame:
    if observations.empty or "status" not in observations.columns:
        return _empty_observation_labels(observations.crs)

    confirmed = observations[observations["status"] == "confirmed_culvert"].copy()
    if confirmed.empty:
        return _empty_observation_labels(observations.crs)

    candidate_id = _first_non_empty_series(
        confirmed,
        ["missed_candidate_id", "nearest_candidate_id"],
    ).str.strip()
    distance_m = pd.to_numeric(
        _first_non_empty_series(
            confirmed,
            ["missed_candidate_distance_m", "nearest_candidate_distance_m"],
        ),
        errors="coerce",
    )
    confirmed["miss_candidate_id"] = candidate_id
    confirmed["miss_distance_m"] = distance_m
    missed = confirmed[
        (confirmed["miss_candidate_id"] != "")
        & (confirmed["miss_distance_m"] > float(miss_threshold_m))
    ].copy()
    if missed.empty:
        return _empty_observation_labels(observations.crs)

    notes = _string_series(missed, "notes")
    miss_notes = missed["miss_distance_m"].map(
        lambda value: f"confirmed culvert was {float(value):.1f} m from this prediction"
    )
    notes = notes.where(notes == "", notes + "; ") + miss_notes

    return gpd.GeoDataFrame(
        {
            "report_date": _string_series(missed, "observed_at").map(_date_part),
            "route": _string_series(missed, "road_name"),
            "culvert_id": _string_series(missed, "miss_candidate_id"),
            "source_file": "field_observations.geojson",
            "label": "missed_prediction",
            "label_confidence": 1.0,
            "observation_id": _string_series(missed, "observation_id"),
            "field_culvert_id": _string_series(missed, "field_culvert_id"),
            "layout_source": _string_series(missed, "layout_source"),
            "candidate_id": _string_series(missed, "miss_candidate_id"),
            "miss_distance_m": missed["miss_distance_m"],
            "notes": notes,
        },
        geometry=missed.geometry,
        crs=missed.crs,
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
            "miss_distance_m",
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


def _first_non_empty_series(table: gpd.GeoDataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series([""] * len(table), index=table.index)
    for column in columns:
        values = _string_series(table, column).str.strip()
        result = result.mask(result == "", values)
    return result


def _date_part(value) -> str:
    text = str(value or "")
    return text[:10] if text else ""


def _observation_culvert_id(row) -> str:
    field_culvert_id = str(row.get("field_culvert_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    observation_id = str(row.get("observation_id") or "").strip()
    return field_culvert_id or candidate_id or observation_id or "field_observation"
