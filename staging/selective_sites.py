"""Opt-in research admission and ordering. No labels, inventory or model fitting.

The original representatives are immutable anchors. Missing physical evidence
never becomes support. Unknown nearby extrema are not assumed to be one pipe.
"""

import heapq
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from staging.hybrid_sites import validate
from staging.retrieval_sites import identity_groups

VERSION = "selective-admission-v1"
RULES = ("corroborated", "balanced", "drainage")


def numeric(frame, name):
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce")


def flag(frame, name):
    return frame.get(name, pd.Series(False, index=frame.index)).fillna(False).astype(bool)


def channel_key(value):
    return "" if pd.isna(value) else str(value)


def support(frame):
    """Bounded physical corroboration, not a calibrated probability."""
    profile = flag(frame, "road_profile_available") & numeric(frame, "route_elevation_dip_60m").ge(
        0.15
    )
    flow = numeric(frame, "upstream_area_m2").ge(1000) & numeric(frame, "negative_tpi_9x9_m").gt(0)
    transverse = frame.proposal_kind.isin(["mapped_crossing", "dem_crossing", "small_dem_crossing"])
    return (profile.astype(float) + flow.astype(float) + transverse.astype(float)) / 3


def admission(frame, rule="corroborated"):
    if rule not in RULES:
        raise ValueError("Unknown selective admission rule")
    profile = flag(frame, "adaptive_profile") & flag(frame, "road_profile_available")
    flow = flag(frame, "adaptive_flow")
    small = frame.proposal_kind.eq("small_dem_crossing")
    area = numeric(frame, "upstream_area_at_link_m2")
    if rule == "drainage":
        return (small & area.ge(100000)) | (profile & flow)
    corroborated = (
        profile
        & numeric(frame, "upstream_area_m2").ge(1000)
        & numeric(frame, "negative_tpi_9x9_m").gt(0)
    ) | (
        flow
        & flag(frame, "road_profile_available")
        & numeric(frame, "route_elevation_dip_60m").ge(0.15)
    )
    if rule == "balanced":
        corroborated |= (
            profile
            & numeric(frame, "route_elevation_dip_60m").ge(0.5)
            & numeric(frame, "valley_depth_9x9_m").ge(0.5)
        )
    return corroborated | (small & area.ge(10000 if rule == "balanced" else 100000))


def population(baseline, hybrid, rule="corroborated"):
    """Preserve the baseline; select only measured, previously generated extras."""
    validate(baseline)
    validate(hybrid)
    lookup = hybrid.set_index("candidate_id", drop=False)
    if not set(baseline.candidate_id).issubset(lookup.index):
        raise ValueError("Baseline representatives must have original measured features")
    # Preserve full baseline provenance and existing representative geometry,
    # joining missing measurements in one block instead of fragmenting a frame.
    measured = lookup.loc[
        baseline.candidate_id, [c for c in hybrid if c not in baseline]
    ].reset_index(drop=True)
    base = gpd.GeoDataFrame(
        pd.concat([baseline.reset_index(drop=True), measured], axis=1), crs=baseline.crs
    )
    represented = set()
    for row in baseline.itertuples():
        represented.update(
            json.loads(getattr(row, "member_candidate_ids", json.dumps([row.candidate_id])))
        )
    extra = hybrid.loc[~hybrid.candidate_id.isin(represented) & admission(hybrid, rule)].copy()
    base["baseline_anchor"] = True
    extra["baseline_anchor"] = False
    for name in ("member_candidate_ids", "member_evidence", "proposal_sources"):
        if name == "member_candidate_ids":
            extra[name] = extra.candidate_id.map(lambda v: json.dumps([v]))
        elif name == "member_evidence":
            extra[name] = "[]"
        else:
            extra[name] = extra.proposal_kind
    extra["member_count"] = 1
    result = gpd.GeoDataFrame(pd.concat([base, extra], ignore_index=True), crs=baseline.crs)
    result["selective_rule"] = rule
    result["selection_policy"] = VERSION
    assert result.candidate_id.is_unique
    return result


def anchored_groups(points):
    """Strict channel/coincidence groups; never move or merge baseline anchors."""
    validate(points)
    base = points.loc[points.baseline_anchor].sort_values("candidate_id", kind="stable").copy()
    extra = points.loc[~points.baseline_anchor].copy()
    if not len(extra):
        return base.reset_index(drop=True)
    extra = identity_groups(extra, 20).sort_values("candidate_id", kind="stable")
    bxy = np.c_[base.geometry.x, base.geometry.y]
    tree = cKDTree(bxy) if len(base) else None
    attached = {j: [] for j in range(len(base))}
    keep = []
    base = base.reset_index(drop=True)
    for index, row in extra.iterrows():
        members = json.loads(row.member_evidence)
        candidates = (
            []
            if tree is None
            else sorted(tree.query_ball_point([row.geometry.x, row.geometry.y], 20))
        )
        target = None
        channel = channel_key(row.get("channel_key", ""))
        for j in candidates:
            anchor = base.iloc[j]
            key = channel_key(anchor.get("channel_key", ""))
            if (
                str(anchor.road_id) != str(row.road_id)
                or anchor.road_part_index != row.road_part_index
            ):
                continue
            if channel and key and channel != key:
                continue
            coincident = row.geometry.distance(anchor.geometry) <= 1e-6
            if not (coincident or (channel and channel == key)):
                continue
            existing = [
                dict(x=anchor.geometry.x, y=anchor.geometry.y, chainage_m=anchor.road_chainage_m),
                *attached[j],
            ]
            if any(
                np.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 20 + 1e-8
                or abs(a["chainage_m"] - b["chainage_m"]) > 20 + 1e-8
                for a in existing
                for b in members
            ):
                continue
            target = j
            break
        if target is None:
            keep.append(index)
        else:
            attached[target].extend(members)
    for j, members in attached.items():
        if not members:
            continue
        original = json.loads(base.loc[j, "member_candidate_ids"])
        new_ids = [v["candidate_id"] for v in members]
        base.loc[j, "member_candidate_ids"] = json.dumps(sorted(set(original + new_ids)))
        base.loc[j, "member_count"] = len(set(original + new_ids))
        prior = base.loc[j, "member_evidence"]
        prior = json.loads(prior) if isinstance(prior, str) else []
        base.loc[j, "member_evidence"] = json.dumps(prior + members, allow_nan=False)
    result = gpd.GeoDataFrame(pd.concat([base, extra.loc[keep]], ignore_index=True), crs=points.crs)
    result["grouping_policy"] = "anchored-evidence-identity-v1"
    result["representative_selection"] = (
        "Original baseline anchor, otherwise existing-coordinate channel medoid; no inventory or score relocation."
    )
    assert set(base.candidate_id).issubset(result.candidate_id)
    return result


def order(points, score="score_RF", weight=0.0, penalty=0.0, baseline_first=False):
    """Soft same-road/path neighbourhood penalty, no route quotas or deletions."""
    validate(points)
    if not np.isfinite(points.road_chainage_m.to_numpy(float)).all():
        raise ValueError("Finite actual-road chainage required")
    if weight not in (0.0, 0.1, 0.25) or penalty not in (0.0, 0.1):
        raise ValueError("Use a declared fixed research ordering")
    result = points.sort_values("candidate_id", kind="stable").reset_index(drop=True).copy()
    values = numeric(result, score).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Finite compatible ranking scores required")
    values = values + weight * support(result).to_numpy(float)
    if baseline_first:
        values += 2 * flag(result, "baseline_anchor").to_numpy(float)
    keys = list(zip(result.road_id.astype(str), result.road_part_index))
    heap = [(-float(v), str(cid), i) for i, (v, cid) in enumerate(zip(values, result.candidate_id))]
    heapq.heapify(heap)
    selected = np.zeros(len(result), bool)
    penalized = np.zeros(len(result), bool)
    tree = cKDTree(np.c_[result.geometry.x, result.geometry.y]) if penalty and len(result) else None
    chosen, priorities = [], []
    while heap:
        negative, _, i = heapq.heappop(heap)
        current = values[i] - penalty * penalized[i]
        if selected[i] or not np.isclose(-negative, current, rtol=0, atol=1e-12):
            continue
        selected[i] = True
        chosen.append(i)
        priorities.append(float(current))
        if tree is not None:
            for j in sorted(
                tree.query_ball_point([result.geometry.iloc[i].x, result.geometry.iloc[i].y], 40)
            ):
                if selected[j] or penalized[j] or keys[j] != keys[i]:
                    continue
                if abs(result.road_chainage_m.iloc[j] - result.road_chainage_m.iloc[i]) > 40:
                    continue
                penalized[j] = True
                heapq.heappush(heap, (-(values[j] - penalty), str(result.candidate_id.iloc[j]), j))
    result = result.iloc[chosen].reset_index(drop=True)
    result["research_priority"] = priorities
    result["global_rank"] = np.arange(1, len(result) + 1)
    result["score_interpretation"] = "Uncalibrated ranking signal; not a probability."
    return result
