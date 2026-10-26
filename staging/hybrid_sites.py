"""Label-free high-recall proposals and bounded, score-ordered physical sites.

This module is staging only. Evidence classes describe available observations,
not calibrated confidence and never alter eligibility or model priority.
"""

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from culvert_ai.candidates import CandidateSettings, generate_candidates, merge_candidate_layers
from culvert_ai.inference import RouteZConfig, dynamic_route_z
from staging.hydrology_sites import WARNING, identifier, line_parts, metric, search_candidates

VERSION = "hybrid-sites-v1"
EVIDENCE = ("crossing_supported", "stream_supported", "terrain_supported", "road_corridor_fallback")


def mapped_proposals(roads, streams):
    """Recover the historical mapped crossing / <=20m approach definition."""
    roads = metric(roads).sort_values("id", kind="stable").reset_index(drop=True)
    streams = metric(streams)
    if "id" in streams:
        streams = streams.sort_values("id", kind="stable").reset_index(drop=True)
    result = generate_candidates(roads, streams, CandidateSettings(20.0, 20.0))
    if result.empty:
        return result
    result["candidate_id"] = [
        "mapped_"
        + identifier(r.road_id, r.stream_id, r.source, f"{r.geometry.x:.4f}", f"{r.geometry.y:.4f}")
        for r in result.itertuples()
    ]
    result["proposal_kind"] = result.source
    result["evidence_source"] = result.source
    return attach_road_topology(result, roads)


def attach_road_topology(points, roads):
    result = metric(points)
    lookup = {str(r.id): r for r in metric(roads).itertuples()}
    records = []
    for row in result.itertuples():
        road = lookup.get(str(row.road_id))
        if road is None:
            raise ValueError(f"Unknown road geometry for proposal {row.candidate_id}")
        parts = line_parts(road.geometry)
        part = min(range(len(parts)), key=lambda i: (parts[i].distance(row.geometry), i))
        records.append(
            (
                part,
                parts[part].project(row.geometry),
                str(getattr(road, "route_id", road.id)),
                getattr(road, "name", str(road.id)),
            )
        )
    result[["road_part_index", "road_chainage_m", "route_id", "road_name"]] = pd.DataFrame(
        records, index=result.index, columns=["part", "chainage", "route", "name"]
    )
    result["road_id"] = result.road_id.astype(str)
    return result


def historical_candidates(roads, streams):
    """Exact legacy geometry recipe restricted to the supplied roads, no labels.

    Historical route selection came partly from field-report route names. Here
    every supplied road is eligible, including unknown IDs/names. The historical
    geometry functions (10m half-offset probes,20m mapped approaches,5m merge)
    remain unchanged. Return both deduplicated output and upstream proposals.
    """
    roads = metric(roads).sort_values("id", kind="stable").reset_index(drop=True)
    probes = attach_road_topology(search_candidates(roads), roads)
    mapped = mapped_proposals(roads, streams)
    for layer in (probes, mapped):
        layer["origin_candidate_id"] = layer.candidate_id
    result = merge_candidate_layers([mapped, probes], min_spacing_m=5.0)
    result["candidate_id"] = result.origin_candidate_id
    return attach_road_topology(result, roads), probes, mapped


def validate(points):
    if points.crs is None or points.crs.to_epsg() != 32618:
        raise ValueError("Site clustering requires EPSG:32618 metre geometry.")
    if not points.candidate_id.is_unique:
        raise ValueError("Candidate IDs must be unique.")
    if not points.geometry.geom_type.eq("Point").all() or points.geometry.is_empty.any():
        raise ValueError("Finite point geometry is required.")
    if not np.isfinite(np.c_[points.geometry.x, points.geometry.y]).all():
        raise ValueError("Finite point geometry is required.")


def classify_evidence(measured):
    result = measured.copy()
    kind = result.proposal_kind.fillna("")
    # The legacy generator can return a centroid of collinear line overlap.
    # Preserve eligibility, but do not mislabel parallel water as a crossing.
    angle = pd.to_numeric(
        result.get("crossing_angle_degrees", pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    mapped_contact = kind.eq("exact_road_stream_intersection")
    crossing = kind.isin(["mapped_crossing", "dem_crossing"]) | (mapped_contact & angle.gt(1e-6))
    stream = kind.eq("nearest_road_stream_approach") | mapped_contact | crossing
    if "distance_to_nearest_stream_m" in result:
        stream |= result.distance_to_nearest_stream_m.le(20)
    terrain = pd.Series(False, index=result.index)
    if "route_low_point_score" in result:
        terrain = result.route_low_point_score.gt(0) & result.route_elevation_dip_60m.ge(0.15)
    result["crossing_supported"] = crossing
    result["stream_supported"] = stream
    result["terrain_supported"] = terrain
    result["road_corridor_fallback"] = ~(crossing | stream | terrain)
    result["representative_evidence_type"] = np.select(
        [crossing, stream, terrain], EVIDENCE[:3], default=EVIDENCE[3]
    )
    return result


def cluster_sites(points, radius_m=20.0, priority_column=None):
    """Bounded spatial+road-chainage clustering, never single-link chaining.

    Same road and part only; distinct nonempty channel identities stay separate.
    Deterministic representative is highest fixed priority (then raw score/ID),
    or candidate-ID ascending for score-free geometry coverage. Never relocate
    a representative or boost a crossing. All member provenance is preserved.
    """
    validate(points)
    frame = points.copy().reset_index(drop=True)
    columns, ascending = ["candidate_id"], [True]
    if priority_column:
        if not np.isfinite(frame[priority_column].to_numpy(float)).all():
            raise ValueError("Clustering priority must be finite.")
        columns, ascending = [priority_column, "model_score", "candidate_id"], [False, False, True]
    ordered = frame.sort_values(columns, ascending=ascending, kind="stable").index.to_numpy()
    xy = np.c_[frame.geometry.x, frame.geometry.y]
    tree = cKDTree(xy)
    road = frame.road_id.astype(str).to_numpy()
    part = frame.road_part_index.to_numpy(int)
    chain = frame.road_chainage_m.to_numpy(float)
    if not np.isfinite(chain).all():
        raise ValueError("Finite road chainage is required.")
    channel = (
        frame.get("channel_key", pd.Series("", index=frame.index)).fillna("").astype(str).to_numpy()
    )
    available = np.ones(len(frame), dtype=bool)
    position = np.empty(len(frame), dtype=int)
    position[ordered] = np.arange(len(frame))
    representatives, members = [], []
    for i in ordered:
        if not available[i]:
            continue
        nearby = sorted(tree.query_ball_point(xy[i], radius_m), key=lambda j: position[j])
        selected, channels = [], set()
        for j in nearby:
            if (
                not available[j]
                or road[j] != road[i]
                or part[j] != part[i]
                or abs(chain[j] - chain[i]) > radius_m + 1e-8
            ):
                continue
            if channel[j] and channels and channel[j] not in channels:
                continue
            if channel[j]:
                channels.add(channel[j])
            selected.append(j)
        available[selected] = False
        representatives.append(i)
        members.append(selected)
    result = frame.iloc[representatives].copy().reset_index(drop=True)
    origins = frame.get(
        "member_candidate_ids", frame.candidate_id.map(lambda v: json.dumps([v]))
    ).to_numpy()
    sources = frame.get("proposal_sources", frame.proposal_kind.fillna("")).to_numpy()
    result["member_candidate_ids"] = [
        json.dumps(sorted(set(x for j in use for x in json.loads(origins[j])))) for use in members
    ]
    result["member_count"] = result.member_candidate_ids.map(lambda v: len(json.loads(v)))
    result["proposal_sources"] = [
        ",".join(sorted(set(x for j in use for x in str(sources[j]).split(",") if x)))
        for use in members
    ]
    for evidence in EVIDENCE:
        if evidence in frame:
            values = frame[evidence].fillna(False).to_numpy(bool)
            result[evidence] = [bool(values[use].any()) for use in members]
    if "crossing_supported" in result:
        result["evidence_type"] = np.select(
            [result[c] for c in EVIDENCE[:3]], EVIDENCE[:3], default=EVIDENCE[3]
        )
    result["physical_site_id"] = [
        "hybrid_" + identifier(radius_m, r.road_id, r.road_part_index, r.candidate_id)
        for r in result.itertuples()
    ]
    result["field_verification_notice"] = WARNING
    return result


def overlapping_proposals(points):
    # Coordinate-scale merging only, not20m site selection. No stream eligibility
    # gate and no label-based exclusions. Same table is used by both RF policies.
    return cluster_sites(points, radius_m=1.0).drop(columns=["physical_site_id"])


def rank_sites(points, policy="raw_rf", reference=None):
    if policy not in {"raw_rf", "dynamic_route_z"}:
        raise ValueError("Unknown fixed ranking policy")
    frame = points.copy().reset_index(drop=True)
    if not np.isfinite(frame.model_score.to_numpy(float)).all():
        raise ValueError("Ranking scores must be finite.")
    if policy == "dynamic_route_z":
        z = dynamic_route_z(
            frame.rename(columns={"model_score": "flow_c_score"}),
            config=reference or RouteZConfig(),
        )
        frame["priority"] = z.route_z_score.to_numpy()
        frame["route_z_method"] = z.route_z_method.to_numpy()
    else:
        frame["priority"] = frame.model_score
        frame["route_z_method"] = "not_used"
    result = cluster_sites(frame, priority_column="priority")
    result = result.sort_values(
        ["priority", "model_score", "candidate_id"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["ranking_policy"] = policy
    result["score_interpretation"] = "Uncalibrated model ranking score; not a probability."
    return result
