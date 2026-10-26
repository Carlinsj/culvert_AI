"""Read-only live feature localization. No inventory, training or production fallback."""

import json
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box
from threadpoolctl import threadpool_limits

from staging.feature_sites import (
    VERSION,
    EVIDENCE_COLUMNS,
    check_model,
    proposals,
    ranked,
    spacing_links,
)
from staging.hydrology_service import ROOT, HYDRO, datasets, geojson, load_model
from staging.hydrology_sites import (
    WARNING,
    apply_feature_contract,
    detect_events,
    physical_measurements,
)
from staging.hybrid_sites import (
    historical_candidates,
    attach_road_topology,
    classify_evidence,
    overlapping_proposals,
    cluster_sites,
)
from staging.retrieval_sites import identity_groups, ranked_identity

META = [
    "candidate_id",
    "physical_site_id",
    "road_id",
    "road_name",
    "route_id",
    "road_part_index",
    "road_chainage_m",
    "proposal_kind",
    "proposal_sources",
    "global_rank",
    "model_score",
    "priority",
    "ranking_policy",
    "evidence_type",
    "member_count",
    "member_candidate_ids",
    "localization_evidence",
    "localization_policy",
    "originated_at_sampling_anchor",
    "profile_local_peak",
    "flow_valley_local_peak",
    "ml_local_peak",
    "physical_crossing",
    "evidence_admitted",
    "missing_feature_count",
    *EVIDENCE_COLUMNS,
]
META.append("member_localization_evidence")
META.extend(["site_type", "legacy_evidence_type", "selection_policy"])
META.extend(
    ["grouping_policy", "grouping_distance_m", "representative_selection", "member_evidence"]
)
META.extend(["x_m", "y_m"])


def display_metadata(frame, site_type):
    """Presentation provenance only;no eligibility,score,geometry or rank edits."""
    result = frame.copy()
    result["x_m"] = result.geometry.x
    result["y_m"] = result.geometry.y
    result["site_type"] = site_type
    result["legacy_evidence_type"] = result.get(
        "evidence_type", result.get("representative_evidence_type")
    )
    result["evidence_type"] = np.select(
        [
            result.physical_crossing,
            result.profile_local_peak & result.flow_valley_local_peak,
            result.profile_local_peak,
            result.flow_valley_local_peak,
        ],
        [
            "crossing-supported",
            "profile-and-flow-supported",
            "road-profile-supported",
            "flow-and-concavity-supported",
        ],
        default="limited-physical-evidence",
    )
    return result


@lru_cache(maxsize=4)
def analyze(
    latitude,
    longitude,
    radius_m=1500,
    policy="dynamic_route_z",
    mode="feature",
    grouping="legacy20",
):
    lat, lon, radius = float(latitude), float(longitude), float(radius_m)
    if (
        not np.isfinite([lat, lon, radius]).all()
        or not -90 <= lat <= 90
        or not -180 <= lon <= 180
        or not 200 <= radius <= 4000
    ):
        raise ValueError("Use valid coordinates and a radius between200 and4000 metres.")
    if policy not in {"raw_rf", "dynamic_route_z"} or mode not in {"feature", "high_recall"}:
        raise ValueError("Unknown fixed staging mode or ranking policy.")
    if grouping not in {"legacy20", "identity20"}:
        raise ValueError("Unknown staging grouping policy.")
    center = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32618).iloc[0]
    area = center.buffer(radius)
    with rasterio.open(HYDRO / "dem_projected.tif") as dem:
        if not box(*dem.bounds).covers(area):
            raise ValueError("Outside supported coverage. Explore the supported Beaver Kill area.")
    all_roads, streams, networks = datasets()
    roads = (
        all_roads.iloc[sorted(all_roads.sindex.query(area, predicate="intersects"))]
        .copy()
        .reset_index(drop=True)
    )
    if roads.empty:
        raise ValueError("No supported roads here. Explore the supported area.")
    roads["route_id"] = roads.name.fillna(roads.id.astype(str)).astype(str)
    _, anchors, mapped = historical_candidates(roads, streams)
    events, _ = detect_events(roads, streams, networks)
    points = gpd.GeoDataFrame(pd.concat([anchors, mapped, events], ignore_index=True), crs=32618)
    # 60m halo permits true local comparisons at the display boundary. No site is moved.
    points = (
        points.loc[points.geometry.distance(center) <= radius + 60].copy().reset_index(drop=True)
    )
    points = attach_road_topology(points, roads)
    if points.empty:
        raise ValueError("No usable road geometry here.")
    fitted = load_model("RF")
    schema = json.loads((ROOT / "models/flow_c_frozen_v1.joblib.schema.json").read_text())[
        "feature_columns"
    ]
    columns = check_model(fitted, schema)
    points["source"] = "route_interval_sample"
    measured = physical_measurements(points, all_roads, streams, HYDRO)
    inside = points.geometry.distance(center) <= radius
    context = measured.loc[inside & measured.proposal_kind.eq("internal_road_probe")]
    transformed = apply_feature_contract(
        measured, fitted["contract"], normalization_context=context
    )
    numeric = (
        transformed[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    with threadpool_limits(limits=2):
        scores = fitted["model"].predict_proba(numeric.fillna(fitted["medians"]))[:, 1]
    enriched = points.copy()
    for col in EVIDENCE_COLUMNS:
        enriched[col] = transformed[col].to_numpy()
    enriched["model_score"] = scores
    enriched["missing_feature_count"] = numeric.isna().sum(axis=1).to_numpy()
    enriched = proposals(classify_evidence(enriched))
    inner = enriched.loc[inside].copy().reset_index(drop=True)
    selected = inner.loc[inner.evidence_admitted].copy()
    # Do not pre-suppress raw evidence before the conservative grouping option.
    population = (
        selected
        if mode == "feature"
        else (inner if grouping == "identity20" else overlapping_proposals(inner))
    )
    sites = (
        ranked_identity(population, policy)
        if grouping == "identity20"
        else ranked(population, policy)
    )
    scorefree = (
        identity_groups(population) if grouping == "identity20" else cluster_sites(population)
    )
    if grouping == "legacy20":
        for frame in (sites, scorefree):
            frame["grouping_policy"] = "legacy-bounded-road-nms20"
            frame["grouping_distance_m"] = 20
            frame["representative_selection"] = (
                "Existing 20 m road-aware suppression; score-ordered representative for recommendations, stable candidate ID for score-free sites. Proximity is not proof of one culvert."
            )
    # Full, unmodified member evidence is available for either policy, including
    # pre-suppressed IDs. Metadata only: never refit, relocate or rescore a site.
    raw_lookup = {str(r.candidate_id): r for r in inner.itertuples()}
    ll_lookup = inner.to_crs(4326).set_index("candidate_id")
    for frame in (sites, scorefree):
        if "member_evidence" in frame:
            continue
        frame["member_evidence"] = [
            json.dumps(
                [
                    dict(
                        candidate_id=member,
                        latitude=float(ll_lookup.loc[member].geometry.y),
                        longitude=float(ll_lookup.loc[member].geometry.x),
                        source=str(raw_lookup[member].proposal_kind),
                        model_score=float(raw_lookup[member].model_score),
                        channel_key=str(getattr(raw_lookup[member], "channel_key", "")),
                    )
                    for member in json.loads(value)
                    if member in raw_lookup
                ]
            )
            for value in frame.member_candidate_ids
        ]
    sites = display_metadata(sites, "Ranked experimental inspection recommendation")
    scorefree = display_metadata(scorefree, "Consolidated potential physical site")
    selected = display_metadata(selected, "Feature-driven experimental proposal")
    inner = display_metadata(inner, "Raw computational anchor or geometric contact")
    sites["selection_policy"] = VERSION if mode == "feature" else "high-recall-all-candidates-v1"
    scorefree["selection_policy"] = (
        sites.selection_policy.iloc[0]
        if len(sites)
        else (VERSION if mode == "feature" else "high-recall-all-candidates-v1")
    )
    selected["selection_policy"] = VERSION
    inner["selection_policy"] = "raw-computation-no-admission"
    local_water = streams.iloc[streams.sindex.query(area, predicate="intersects")]
    derived = networks["0p1km2"]
    derived = derived.iloc[derived.sindex.query(area, predicate="intersects")]
    return dict(
        experimental=True,
        mode=mode,
        grouping=grouping,
        grouping_notice="Grouping is a label-free hypothesis, not verified culvert identity. Legacy 20 m remains default; evidence identity is an explicit conservative alternative.",
        field_verification_notice=WARNING,
        model=dict(
            version=fitted["model_version"],
            feature_contract="hydrology-sites-v2",
            original_frozen_compatible=False,
            ranking_policy=policy,
            calibrated=False,
            independent_validation=False,
        ),
        localization_policy=VERSION,
        analysis=dict(
            latitude=lat, longitude=lon, radius_m=radius, roads=geojson(roads, ["id", "name"])
        ),
        site_count=len(sites),
        sites=geojson(sites, META),
        anchors=geojson(inner.loc[inner.proposal_kind.eq("internal_road_probe")], META),
        raw_candidates=geojson(inner, META),
        proposals=geojson(selected, META),
        consolidated=geojson(scorefree, META),
        mapped_crossings=geojson(inner.loc[inner.proposal_kind.eq("mapped_crossing")], META),
        dem_crossings=geojson(inner.loc[inner.proposal_kind.eq("dem_crossing")], META),
        spacing={
            key: geojson(
                spacing_links(frame, roads, label),
                ["road_id", "road_part_index", "from_id", "to_id", "distance_m", "measurement"],
            )
            for key, frame, label in [
                (
                    "anchors",
                    inner.loc[inner.proposal_kind.eq("internal_road_probe")],
                    "sampling-anchor",
                ),
                ("proposals", selected, "feature-proposal"),
            ]
        },
        hydrology=dict(
            mapped=geojson(local_water, ["name"]), derived=geojson(derived, ["threshold_km2"])
        ),
        diagnostics=dict(
            raw_candidates=len(inner),
            sampling_anchors=int(inner.proposal_kind.eq("internal_road_probe").sum()),
            admitted_proposals=len(selected),
            physical_sites=len(sites),
            no_proposal_roads=sorted(set(roads.id.astype(str)) - set(selected.road_id.astype(str))),
        ),
        limitations=[
            "Experimental v2 score,not an independently validated probability.",
            "Frozenv1 feature parity unresolved in35/48fields; it is not used.",
            "Cross-road elevation difference is not an existing compatible measurement; unavailable.",
            "Edge-originating flowpath value is raw path accumulation,not a risk probability.",
            "Local terrain or score extrema do not establish a culvert; absent evidence is not absence.",
        ],
    )
