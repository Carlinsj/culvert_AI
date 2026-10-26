"""Conservative evidence-identity grouping; no inventory or fitted thresholds.

Only coincident representations or a shared explicit drainage-channel identity
can merge. Distance bounds constrain such merges; proximity alone is insufficient.
Road samples and separate measured extrema are not silently called one culvert.
"""

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from culvert_ai.inference import dynamic_route_z
from staging.hybrid_sites import validate
from staging.hydrology_sites import WARNING, identifier

VERSION = "evidence-identity-v1"
RADII = (5, 10, 15, 20, 30, 40)


def identity_groups(points, radius_m=20.0):
    """Complete-link bounds, same road/part, stable existing-point medoid.

    Exact duplicate means <=1e-6 metres, not rounded lat/lon. Conflicting explicit
    channels never merge, even at identical coordinates. A shared channel can
    recur across a winding road; every pair must meet both spatial and chainage
    bounds. No chaining, road/intersection bridging, score-based relocation or
    suppression of distinct local extrema. Missing identity stays separate.
    """
    validate(points)
    if not np.isfinite(radius_m) or not 0 < radius_m <= 100:
        raise ValueError("Grouping distance must be in (0,100] metres.")
    frame = points.sort_values("candidate_id", kind="stable").reset_index(drop=True).copy()
    chain = frame.road_chainage_m.to_numpy(float)
    if not np.isfinite(chain).all():
        raise ValueError("Finite road chainage is required.")
    xy = np.c_[frame.geometry.x, frame.geometry.y]
    channels = frame.get("channel_key", pd.Series("", index=frame.index)).fillna("").astype(str)
    keys = list(zip(frame.road_id.astype(str), frame.road_part_index.astype(int)))
    tree = cKDTree(xy)
    used = np.zeros(len(frame), dtype=bool)
    groups = []
    for i in range(len(frame)):
        if used[i]:
            continue
        members = [i]
        for j in sorted(tree.query_ball_point(xy[i], radius_m)):
            if j == i or used[j] or keys[j] != keys[i]:
                continue
            if channels[i] and channels[j] and channels[i] != channels[j]:
                continue
            same_channel = bool(channels[i] and channels[i] == channels[j])
            coincident = np.linalg.norm(xy[i] - xy[j]) <= 1e-6
            if not (same_channel or coincident):
                continue
            # A coincident member with unknown channel must not bridge identities.
            if any(channels[k] and channels[j] and channels[k] != channels[j] for k in members):
                continue
            if any(
                np.linalg.norm(xy[k] - xy[j]) > radius_m + 1e-8
                or abs(chain[k] - chain[j]) > radius_m + 1e-8
                for k in members
            ):
                continue
            members.append(j)
        used[members] = True
        groups.append(members)
    reps, evidence, ids, sources, reasons = [], [], [], [], []
    for members in groups:
        # Choose an actual observed location, independent of scores and inventory.
        costs = np.linalg.norm(xy[members, None, :] - xy[None, members, :], axis=2).sum(axis=1)
        rep = members[int(np.argmin(costs))]  # members are ID-sorted: deterministic tie.
        reps.append(rep)
        origin_ids = set()
        member_evidence = []
        for j in members:
            row = frame.iloc[j]
            origin_ids.update(
                json.loads(row.get("member_candidate_ids", json.dumps([row.candidate_id])))
            )
            member_evidence.append(
                dict(
                    candidate_id=row.candidate_id,
                    road_id=str(row.road_id),
                    road_part_index=int(row.road_part_index),
                    chainage_m=float(row.road_chainage_m),
                    x=float(row.geometry.x),
                    y=float(row.geometry.y),
                    crs="EPSG:32618",
                    source=str(row.get("evidence_source", row.proposal_kind)),
                    proposal_kind=row.proposal_kind,
                    channel_key=str(channels[j]),
                    model_score=float(row.model_score)
                    if pd.notna(row.get("model_score"))
                    else None,
                    measurements={
                        k: (float(row[k]) if pd.notna(row.get(k)) else None)
                        for k in (
                            "upstream_area_m2",
                            "flow_accumulation_cells",
                            "route_elevation_dip_60m",
                            "negative_tpi_9x9_m",
                            "distance_to_nearest_stream_m",
                        )
                    },
                )
            )
        ids.append(json.dumps(sorted(origin_ids)))
        sources.append(",".join(sorted({str(frame.iloc[j].proposal_kind) for j in members})))
        evidence.append(json.dumps(member_evidence, allow_nan=False))
        reasons.append(
            "Existing-coordinate medoid of shared drainage identity or coincident representations; stable candidate-ID tie; scores and inventory unused."
        )
    result = frame.iloc[reps].copy().reset_index(drop=True)
    result["member_candidate_ids"] = ids
    result["member_count"] = [len(json.loads(v)) for v in ids]
    result["member_evidence"] = evidence
    result["proposal_sources"] = sources
    result["representative_selection"] = reasons
    result["grouping_policy"] = VERSION
    result["grouping_distance_m"] = radius_m
    result["physical_site_id"] = ["identity_" + identifier(radius_m, v) for v in ids]
    result["field_verification_notice"] = WARNING
    if "localization_evidence" in frame:
        result["member_localization_evidence"] = [
            json.dumps(
                sorted(
                    {e for j in members for e in json.loads(frame.iloc[j].localization_evidence)}
                )
            )
            for members in groups
        ]
    return result


def ranked_identity(points, policy="dynamic_route_z", radius_m=20.0):
    """Freeze physical groups first; only then order existing representatives."""
    if policy not in {"raw_rf", "dynamic_route_z"}:
        raise ValueError("Unknown fixed ranking policy")
    frame = points.copy()
    if not np.isfinite(frame.model_score.to_numpy(float)).all():
        raise ValueError("Finite saved compatible scores are required.")
    if policy == "dynamic_route_z" and len(frame):
        z = dynamic_route_z(frame.rename(columns={"model_score": "flow_c_score"}))
        frame["priority"] = z.route_z_score.to_numpy()
        frame["route_z_method"] = z.route_z_method.to_numpy()
    else:
        frame["priority"] = frame.model_score
        frame["route_z_method"] = "not_used"
    result = identity_groups(frame, radius_m)
    result = result.sort_values(
        ["priority", "model_score", "candidate_id"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["ranking_policy"] = policy
    result["score_interpretation"] = "Uncalibrated ranking score, not a probability."
    return result
