"""Separate research-only physical-discovery path; no training or inventory reads."""

from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box
from threadpoolctl import threadpool_limits

from staging.discovery_sites import (
    VERSION,
    adaptive_proposals,
    small_drainage,
    physical_groups,
    ordered,
)
from staging.feature_sites import proposals, spacing_links
from staging.feature_service import META
from staging.hydrology_service import datasets, load_model, geojson, HYDRO
from staging.hydrology_sites import (
    WARNING,
    physical_measurements,
    apply_feature_contract,
    detect_events,
)
from staging.hybrid_sites import historical_candidates, attach_road_topology, classify_evidence


def compatible_scores(measured, context, fitted):
    """Saved family-specific transform and imputation, no changed representation."""
    columns = fitted["feature_columns"]
    if (
        not fitted.get("experimental")
        or fitted["contract"]["version"] != "hydrology-sites-v2"
        or len(columns) != 48
        or columns != fitted["contract"]["feature_columns"]
        or list(fitted["medians"].index) != columns
    ):
        raise ValueError("An exact saved v2 feature contract is required")
    transformed = apply_feature_contract(
        measured, fitted["contract"], normalization_context=context
    )
    values = (
        transformed[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    if getattr(fitted["model"], "n_features_in_", 48) != 48:
        raise ValueError("Incompatible estimator feature dimension")
    with threadpool_limits(limits=2):
        scores = fitted["model"].predict_proba(values.fillna(fitted["medians"]))[:, 1]
    return transformed, scores, values.isna().sum(axis=1).to_numpy()


def prepare(measured, center, radius, small_measured=None, families=("RF", "HGB")):
    """Exact dense baseline reference context reused by every tested variant."""
    original_inside = measured.geometry.distance(center).le(radius)
    context = measured.loc[original_inside & measured.proposal_kind.eq("internal_road_probe")]
    combined = (
        measured
        if small_measured is None or small_measured.empty
        else gpd.GeoDataFrame(pd.concat([measured, small_measured], ignore_index=True), crs=32618)
    )
    assert combined.candidate_id.is_unique
    output = None
    transforms = {}
    for family in families:
        transformed, scores, missing = compatible_scores(combined, context, load_model(family))
        transforms[family] = transformed
        if output is None:
            output = transformed.copy()
        output[f"score_{family}"] = scores
        output[f"missing_{family}"] = missing
    output["model_score"] = output["score_RF"]
    output["missing_feature_count"] = output["missing_RF"]
    output = proposals(classify_evidence(output))
    output = adaptive_proposals(output)
    # Preserve baseline-admitted 60m features as well as new short-profile events.
    output["discovery_admitted"] |= output.evidence_admitted
    output["profile_local_peak"] |= output.adaptive_profile
    output["flow_valley_local_peak"] |= output.adaptive_flow
    output["inside_query"] = output.geometry.distance(center).le(radius)
    return output, transforms


@lru_cache(maxsize=4)
def analyze(latitude, longitude, radius_m=1500, policy="dynamic_route_z"):
    lat, lon, radius = float(latitude), float(longitude), float(radius_m)
    if (
        not np.isfinite([lat, lon, radius]).all()
        or not -90 <= lat <= 90
        or not -180 <= lon <= 180
        or not 200 <= radius <= 4000
    ):
        raise ValueError("Use valid coordinates and a radius between 200 and 4000 metres.")
    if policy not in {"raw_rf", "dynamic_route_z"}:
        raise ValueError("Unknown fixed ranking policy")
    center = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32618).iloc[0]
    area = center.buffer(radius)
    with rasterio.open(HYDRO / "dem_projected.tif") as dem:
        if not box(*dem.bounds).covers(area):
            raise ValueError("Outside supported coverage. Explore a supported road area.")
    all_roads, streams, networks = datasets()
    roads = (
        all_roads.iloc[sorted(all_roads.sindex.query(area, predicate="intersects"))]
        .copy()
        .reset_index(drop=True)
    )
    if roads.empty:
        raise ValueError("No supported roads here. Explore another road area.")
    roads["route_id"] = roads.name.fillna(roads.id.astype(str)).astype(str)
    _, anchors, mapped = historical_candidates(roads, streams)
    events, _ = detect_events(roads, streams, networks)
    points = gpd.GeoDataFrame(pd.concat([anchors, mapped, events], ignore_index=True), crs=32618)
    points = attach_road_topology(
        points.loc[points.geometry.distance(center).le(radius + 60)].reset_index(drop=True), roads
    )
    if points.empty:
        raise ValueError("No usable road geometry here.")
    points["source"] = "route_interval_sample"
    measured = physical_measurements(points, all_roads, streams, HYDRO)
    small, links = small_drainage(roads, HYDRO, center.buffer(radius + 80).bounds)
    small = small.loc[small.geometry.distance(center).le(radius + 60)].copy().reset_index(drop=True)
    small_measured = physical_measurements(small, all_roads, streams, HYDRO) if len(small) else None
    raw, _ = prepare(measured, center, radius, small_measured, families=("RF",))
    inner = raw.loc[raw.inside_query].copy()
    selected = inner.loc[inner.discovery_admitted].copy()
    groups = physical_groups(selected)
    sites = ordered(groups, policy=policy)
    for frame in [inner, selected, groups, sites]:
        frame["site_type"] = "Experimental physical-site hypothesis"
        frame["evidence_type"] = np.where(
            frame.physical_crossing, "crossing-supported", "terrain-supported"
        )
    meta = list(
        dict.fromkeys(
            META
            + [
                "adaptive_profile",
                "adaptive_flow",
                "discovery_admitted",
                "upstream_area_at_link_m2",
                "evidence_source",
                "channel_key",
            ]
        )
    )
    local_water = streams.iloc[streams.sindex.query(area, predicate="intersects")]
    return dict(
        experimental=True,
        mode="physical_discovery",
        grouping="physical_identity20",
        grouping_notice="Shared drainage identity or coincident geometry only; separate close channels remain separate. Unverified site identity.",
        field_verification_notice=WARNING,
        localization_policy=VERSION,
        model=dict(
            version=load_model("RF")["model_version"],
            feature_contract="hydrology-sites-v2",
            original_frozen_compatible=False,
            ranking_policy=policy,
            calibrated=False,
            independent_validation=False,
        ),
        analysis=dict(
            latitude=lat, longitude=lon, radius_m=radius, roads=geojson(roads, ["id", "name"])
        ),
        site_count=len(sites),
        sites=geojson(sites, meta),
        anchors=geojson(inner.loc[inner.proposal_kind.eq("internal_road_probe")], meta),
        raw_candidates=geojson(inner, meta),
        proposals=geojson(selected, meta),
        consolidated=geojson(groups, meta),
        mapped_crossings=geojson(inner.loc[inner.proposal_kind.eq("mapped_crossing")], meta),
        dem_crossings=geojson(
            inner.loc[inner.proposal_kind.isin(["dem_crossing", "small_dem_crossing"])], meta
        ),
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
                ("proposals", selected, "physical-proposal"),
            ]
        },
        hydrology=dict(
            mapped=geojson(local_water, ["name"]),
            derived=geojson(links, ["link_key", "upstream_area_m2"]),
        ),
        diagnostics=dict(
            raw_candidates=len(inner),
            sampling_anchors=int(inner.proposal_kind.eq("internal_road_probe").sum()),
            admitted_proposals=len(selected),
            physical_sites=len(sites),
            no_proposal_roads=sorted(set(roads.id.astype(str)) - set(selected.road_id.astype(str))),
        ),
        limitations=[
            "Retrospective experimental candidate comparison; no independent accuracy claim.",
            "One-hectare D8 paths on a conditioned 10m DEM may be artificial or represent bridges/open channels.",
            "Road-profile extrema do not prove a ditch or pipe; across-road embankment measurements remain unavailable.",
            "Scores are uncalibrated. Original frozen v1 is not compatible and is not used.",
        ],
    )
