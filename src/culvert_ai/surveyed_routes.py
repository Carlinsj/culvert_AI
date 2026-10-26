"""Fail-closed helpers for complete surveyed-route supervision.

Candidate generation is intentionally absent from this module.  It resolves
documented survey intervals, joins inventory records after generation, and
creates three-way labels without treating unvisited roads as negatives.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


CRS = "EPSG:26918"
POSITIVE_TOLERANCE_M = 20.0
UNCERTAINTY_BUFFER_M = 40.0
MAX_SUMMER_ROUTE_DISTANCE_M = 100.0


@dataclass(frozen=True)
class SurveySegmentSpec:
    segment_id: str
    route: str
    route_id: str
    start_marker: str
    end_marker: str
    coverage_status: str
    source_document: str
    source_date: str
    evidence_note: str


def canonical_route(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    if text.isdigit():
        return "NY" + text
    if text.startswith("ROUTE"):
        return "NY" + text[5:]
    return text


def canonical_culvert_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    text = re.sub(r"\.0$", "", text)
    if text.startswith("SC"):
        text = text[2:]
    return text if text.isdigit() else ""


def canonical_marker(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").upper())


def marker_in_interval(value: object, start: object, end: object) -> bool:
    marker = canonical_marker(value)
    lower = canonical_marker(start)
    upper = canonical_marker(end)
    if not marker or len(marker) < 4 or len(lower) < 4 or len(upper) < 4:
        return False
    if marker[:-3] != lower[:-3] or lower[:-3] != upper[:-3]:
        return False
    try:
        return int(lower[-3:]) <= int(marker[-3:]) <= int(upper[-3:])
    except ValueError:
        return False


def read_survey_manifest(path: str | Path) -> list[SurveySegmentSpec]:
    table = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = set(SurveySegmentSpec.__dataclass_fields__)
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Survey manifest missing columns: {sorted(missing)}")
    if table.segment_id.duplicated().any():
        raise ValueError("Survey segment IDs must be unique")
    invalid = table.loc[~table.coverage_status.eq("confirmed_complete"), "segment_id"].tolist()
    if invalid:
        raise ValueError(f"Non-confirmed survey extents cannot create negatives: {invalid}")
    return [SurveySegmentSpec(**row) for row in table[list(required)].to_dict("records")]


def _interpolate_vertex(left: list[float], right: list[float], measure: float) -> tuple[float, float]:
    m0, m1 = float(left[-1]), float(right[-1])
    if np.isclose(m0, m1):
        return float(left[0]), float(left[1])
    fraction = (measure - m0) / (m1 - m0)
    return (
        float(left[0]) + fraction * (float(right[0]) - float(left[0])),
        float(left[1]) + fraction * (float(right[1]) - float(left[1])),
    )


def clip_measured_paths(features: list[dict], route_id: str, start: float, end: float):
    """Clip ArcGIS paths using their M coordinate, never inventory locations."""
    lower, upper = sorted((float(start), float(end)))
    pieces = []
    for feature in features:
        if str(feature.get("attributes", {}).get("Route_ID")) != str(route_id):
            continue
        for path in feature.get("geometry", {}).get("paths", []):
            for left, right in zip(path[:-1], path[1:]):
                m0, m1 = float(left[-1]), float(right[-1])
                edge_low, edge_high = sorted((m0, m1))
                use_low, use_high = max(lower, edge_low), min(upper, edge_high)
                if use_high <= use_low:
                    continue
                if m0 <= m1:
                    a = _interpolate_vertex(left, right, use_low)
                    b = _interpolate_vertex(left, right, use_high)
                else:
                    a = _interpolate_vertex(left, right, use_high)
                    b = _interpolate_vertex(left, right, use_low)
                if a != b:
                    pieces.append(LineString([a, b]))
    if not pieces:
        raise ValueError(f"No measured route geometry for {route_id} from {lower} to {upper}")
    merged = linemerge(unary_union(pieces))
    if merged.is_empty:
        raise ValueError(f"Empty clipped route geometry for {route_id}")
    return merged


def build_survey_segments(
    specs: list[SurveySegmentSpec], marker_features: list[dict], roadway_features: list[dict]
) -> gpd.GeoDataFrame:
    markers = {
        canonical_marker(feature["properties"]["REFERENCE_MARKER_PANEL"]): feature
        for feature in marker_features
    }
    rows = []
    for spec in specs:
        start = markers.get(canonical_marker(spec.start_marker))
        end = markers.get(canonical_marker(spec.end_marker))
        if start is None or end is None:
            raise ValueError(f"Missing official reference marker for {spec.segment_id}")
        for marker in (start, end):
            properties = marker["properties"]
            if str(properties["ROUTE_ID"]) != spec.route_id or properties.get("LOCERROR") != "NO ERROR":
                raise ValueError(f"Invalid official marker resolution for {spec.segment_id}")
        start_measure = float(start["properties"]["MEASURE"])
        end_measure = float(end["properties"]["MEASURE"])
        geometry = clip_measured_paths(
            roadway_features, spec.route_id, start_measure, end_measure
        )
        rows.append(
            {
                **spec.__dict__,
                "id": spec.segment_id,
                "name": spec.route,
                "start_measure": start_measure,
                "end_measure": end_measure,
                "length_m": float(geometry.length),
                "road_bridge": False,
                "geometry": geometry,
            }
        )
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    if (result.length_m <= 0).any():
        raise ValueError("Survey segment geometry must have positive length")
    return result


def deduplicate_inventory(records: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge only stable-ID or sub-metre duplicates; keep nearby culverts distinct."""
    if records.crs is None:
        raise ValueError("Inventory CRS is required")
    frame = records.to_crs(CRS).copy().reset_index(drop=True)
    frame["route"] = frame.route.map(canonical_route)
    frame["canonical_culvert_id"] = frame.culvert_id.map(canonical_culvert_id)
    frame["_source_order"] = frame.source.map({"official": 0, "summer": 1}).fillna(2)
    frame = frame.sort_values(["_source_order", "record_id"], kind="stable").reset_index(drop=True)
    groups: list[list[int]] = []
    representatives: list[int] = []
    id_lookup: dict[tuple[str, str], list[int]] = {}
    for index, row in frame.iterrows():
        key = (row.route, row.canonical_culvert_id)
        group_index = None
        if key[1]:
            compatible = [
                candidate_group
                for candidate_group in id_lookup.get(key, [])
                if row.geometry.distance(frame.iloc[representatives[candidate_group]].geometry)
                <= 100.0
            ]
            if compatible:
                group_index = min(
                    compatible,
                    key=lambda candidate_group: row.geometry.distance(
                        frame.iloc[representatives[candidate_group]].geometry
                    ),
                )
        if group_index is None:
            for candidate_group, representative in enumerate(representatives):
                other = frame.iloc[representative]
                group_sources = set(frame.iloc[groups[candidate_group]].source.astype(str))
                if (
                    row.route == other.route
                    and str(row.source) not in group_sources
                    and row.geometry.distance(other.geometry) <= 2.0
                ):
                    group_index = candidate_group
                    break
        if group_index is None:
            group_index = len(groups)
            groups.append([])
            representatives.append(index)
        groups[group_index].append(index)
        if key[1]:
            id_lookup.setdefault(key, [])
            if group_index not in id_lookup[key]:
                id_lookup[key].append(group_index)
    output = []
    for number, indices in enumerate(groups, 1):
        group = frame.iloc[indices]
        representative = group.iloc[0].copy()
        ids = [value for value in group.canonical_culvert_id if value]
        representative["canonical_culvert_id"] = ids[0] if ids else ""
        representative["actual_id"] = f"actual_{number:05d}"
        representative["member_record_ids"] = json.dumps(group.record_id.astype(str).tolist())
        representative["inventory_sources"] = ",".join(sorted(set(group.source.astype(str))))
        representative["source_record_count"] = len(group)
        markers = [str(value) for value in group.reference_marker if str(value).strip()]
        representative["reference_marker"] = markers[0] if markers else ""
        output.append(representative)
    return gpd.GeoDataFrame(output, geometry="geometry", crs=CRS).drop(columns="_source_order")


def associate_inventory_to_segments(
    inventory: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    max_distance_m: float = MAX_SUMMER_ROUTE_DISTANCE_M,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    associated, excluded = [], []
    for _, row in inventory.to_crs(segments.crs).iterrows():
        options = segments.loc[segments.route.eq(row.route)]
        marker_matches = [
            index
            for index, segment in options.iterrows()
            if marker_in_interval(row.reference_marker, segment.start_marker, segment.end_marker)
        ]
        distances = options.geometry.distance(row.geometry) if len(options) else pd.Series(dtype=float)
        spatial_matches = distances.loc[distances <= max_distance_m].sort_values().index.tolist()
        marker = canonical_marker(row.reference_marker)
        recognized_marker_family = bool(
            marker
            and marker[-3:].isdigit()
            and any(
                marker[:-3] == canonical_marker(segment.start_marker)[:-3]
                for segment in options.itertuples()
            )
        )
        # A valid marker outside every documented interval is evidence that the
        # record is out of scope. Never override it merely because a noisy or
        # inconsistent coordinate happens to fall near a segment.
        matches = marker_matches or ([] if recognized_marker_family else spatial_matches)
        if not matches:
            excluded.append(
                {
                    "record_id": row.actual_id,
                    "route": row.route,
                    "reason": (
                        "reference_marker_outside_documented_extents"
                        if recognized_marker_family
                        else "outside_documented_survey_extents_or_ambiguous_route"
                    ),
                    "nearest_documented_segment_m": float(distances.min()) if len(distances) else None,
                }
            )
            continue
        chosen = matches[0]
        record = row.copy()
        record["segment_id"] = segments.loc[chosen, "segment_id"]
        record["road_distance_m"] = float(segments.loc[chosen].geometry.distance(row.geometry))
        record["road_association_ambiguous"] = (
            len(matches) > 1
            or (not marker_matches and bool(marker))
            or float(segments.loc[chosen].geometry.distance(row.geometry)) > 40.0
        )
        record["association_basis"] = "reference_marker" if marker_matches else "route_and_geometry"
        associated.append(record)
    if associated:
        result = gpd.GeoDataFrame(associated, geometry="geometry", crs=segments.crs)
    else:
        # Preserve a valid geometry column and schema when every nearby record
        # is rejected by the fail-closed extent rules.
        result = inventory.to_crs(segments.crs).iloc[0:0].copy()
        result["segment_id"] = pd.Series(dtype=str)
        result["road_distance_m"] = pd.Series(dtype=float)
        result["road_association_ambiguous"] = pd.Series(dtype=bool)
        result["association_basis"] = pd.Series(dtype=str)
    exclusions = pd.DataFrame(
        excluded,
        columns=["record_id", "route", "reason", "nearest_documented_segment_m"],
    )
    return result.reset_index(drop=True), exclusions


def label_physical_sites(
    sites: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    positive_tolerance_m: float = POSITIVE_TOLERANCE_M,
    uncertainty_buffer_m: float = UNCERTAINTY_BUFFER_M,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Assign one positive site per actual; nearby extra sites remain uncertain."""
    frame = sites.to_crs(truth.crs).copy().reset_index(drop=True)
    pairs = []
    for segment_id, candidates in frame.groupby("segment_id", sort=True):
        actuals = truth.loc[truth.segment_id.eq(segment_id)]
        for candidate_index, candidate in candidates.iterrows():
            for actual_index, actual in actuals.iterrows():
                distance = float(candidate.geometry.distance(actual.geometry))
                if distance <= uncertainty_buffer_m:
                    pairs.append((distance, candidate_index, actual_index))
    positive_candidates: dict[int, tuple[int, float]] = {}
    used_actuals: set[int] = set()
    for distance, candidate_index, actual_index in sorted(pairs):
        if distance > positive_tolerance_m:
            break
        if "road_association_ambiguous" in truth:
            ambiguity = truth.loc[actual_index, "road_association_ambiguous"]
            if pd.notna(ambiguity) and bool(ambiguity):
                # Ambiguous inventory association can protect nearby sites from
                # becoming negatives, but cannot create a positive training row.
                continue
        if candidate_index in positive_candidates or actual_index in used_actuals:
            continue
        positive_candidates[candidate_index] = (actual_index, distance)
        used_actuals.add(actual_index)
    nearest = {}
    for distance, candidate_index, actual_index in pairs:
        nearest.setdefault(candidate_index, (actual_index, distance))
        if distance < nearest[candidate_index][1]:
            nearest[candidate_index] = (actual_index, distance)
    labels, actual_ids, distances = [], [], []
    for index in frame.index:
        if index in positive_candidates:
            actual_index, distance = positive_candidates[index]
            labels.append("positive")
            actual_ids.append(truth.loc[actual_index, "actual_id"])
            distances.append(distance)
        elif index in nearest:
            actual_index, distance = nearest[index]
            labels.append("uncertain")
            actual_ids.append(truth.loc[actual_index, "actual_id"])
            distances.append(distance)
        else:
            labels.append("negative")
            actual_ids.append("")
            distances.append(np.nan)
    frame["supervision_label"] = labels
    frame["target"] = frame.supervision_label.map({"negative": 0, "positive": 1})
    frame["matched_actual_id"] = actual_ids
    frame["nearest_actual_distance_m"] = distances
    frame["hard_negative"] = frame.supervision_label.eq("negative")
    matches = pd.DataFrame(
        [
            {
                "candidate_id": frame.loc[candidate_index, "candidate_id"],
                "actual_id": truth.loc[actual_index, "actual_id"],
                "distance_m": distance,
            }
            for candidate_index, (actual_index, distance) in positive_candidates.items()
        ]
    )
    return frame, matches
