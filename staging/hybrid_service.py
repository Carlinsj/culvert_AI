"""Live hybrid research inference, independent of inventory and production."""

from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box
from threadpoolctl import threadpool_limits

from staging.hydrology_service import HYDRO, datasets, geojson, load_model
from staging.hydrology_sites import (
    WARNING,
    apply_feature_contract,
    detect_events,
    physical_measurements,
)
from staging.hybrid_sites import (
    EVIDENCE,
    VERSION,
    classify_evidence,
    historical_candidates,
    overlapping_proposals,
    rank_sites,
)


@lru_cache(maxsize=4)
def analyze(latitude, longitude, radius_m=1500.0, policy="dynamic_route_z"):
    latitude, longitude, radius_m = float(latitude), float(longitude), float(radius_m)
    if not (
        np.isfinite([latitude, longitude, radius_m]).all()
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and 200 <= radius_m <= 4000
    ):
        raise ValueError("Use valid coordinates and a radius between200 and4000 metres.")
    if policy not in {"raw_rf", "dynamic_route_z"}:
        raise ValueError("Choose raw_rf or dynamic_route_z.")
    center = gpd.GeoSeries([Point(longitude, latitude)], crs=4326).to_crs(32618).iloc[0]
    area = center.buffer(radius_m)
    with rasterio.open(HYDRO / "dem_projected.tif") as dem:
        if dem.crs.to_epsg() != 32618 or not box(*dem.bounds).covers(area):
            raise ValueError(
                "This area is outside supported terrain coverage. Explore a supported area."
            )
    fitted = load_model("RF")  # Explicit v2 experimental RF guard, never frozen v1.
    all_roads, streams, networks = datasets()
    roads = (
        all_roads.iloc[sorted(all_roads.sindex.query(area, predicate="intersects"))]
        .copy()
        .reset_index(drop=True)
    )
    if roads.empty:
        raise ValueError("No supported roads here. Explore a supported area.")
    roads["route_id"] = roads.name.fillna(roads.id.astype(str)).astype(str)
    _, probes, mapped = historical_candidates(roads, streams)
    events, _ = detect_events(roads, streams, networks)
    # Complete road geometry preserves stable chainage and measured shoulders;
    # only request-window proposals become scored/displayed sites.
    points = gpd.GeoDataFrame(pd.concat([probes, mapped, events], ignore_index=True), crs=32618)
    points = points.loc[points.geometry.distance(center) <= radius_m].reset_index(drop=True)
    if points.empty:
        raise ValueError("No usable road proposals here.")
    from staging.hybrid_sites import attach_road_topology

    points = attach_road_topology(points, roads)
    points["source"] = "route_interval_sample"
    measured = physical_measurements(points, all_roads, streams, HYDRO)
    transformed = apply_feature_contract(
        measured,
        fitted["contract"],
        normalization_context=measured.loc[measured.proposal_kind.eq("internal_road_probe")],
    )
    columns = fitted["feature_columns"]
    numeric = (
        transformed[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    with threadpool_limits(limits=2):
        scores = fitted["model"].predict_proba(numeric.fillna(fitted["medians"]))[:, 1]
    enriched = points.copy()
    evidence = [
        "distance_to_nearest_stream_m",
        "route_elevation_dip_60m",
        "route_low_point_score",
        "road_profile_available",
        "flow_accumulation_cells",
        "upstream_area_m2",
        "slope_degrees",
        "hydrology_edge_risk",
    ]
    for column in evidence:
        enriched[column] = transformed[column].to_numpy()
    enriched["model_score"], enriched["missing_feature_count"] = (
        scores,
        numeric.isna().sum(axis=1).to_numpy(),
    )
    enriched = classify_evidence(enriched)
    candidates = overlapping_proposals(enriched)
    sites = rank_sites(candidates, policy)
    sites["model_version"] = fitted["model_version"]
    sites["selection_reason"] = (
        "High-recall road/drainage proposal; representative chosen only by fixed RF/ROUTE-Z priority within20m road/chainage groups. Evidence classes are descriptive, not ranking boosts. Fallback is not a confirmed drainage structure."
    )
    mapped_water = streams.iloc[streams.sindex.query(area, predicate="intersects")]
    drainage = networks["0p1km2"]
    drainage = drainage.iloc[drainage.sindex.query(area, predicate="intersects")]
    visible_events = events.loc[events.geometry.distance(center) <= radius_m]
    return dict(
        experimental=True,
        field_verification_notice=WARNING,
        model=dict(
            version=fitted["model_version"],
            candidate_version=VERSION,
            feature_contract=fitted["contract"]["version"],
            ranking_policy=policy,
            calibrated=False,
            independent_validation=False,
        ),
        analysis=dict(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            roads=geojson(roads, ["id", "name"]),
        ),
        site_count=len(sites),
        sites=geojson(
            sites,
            [
                "candidate_id",
                "physical_site_id",
                "road_id",
                "road_name",
                "route_id",
                "global_rank",
                "model_score",
                "priority",
                "ranking_policy",
                "evidence_type",
                "representative_evidence_type",
                *EVIDENCE,
                "member_count",
                "member_candidate_ids",
                "proposal_sources",
                "model_version",
                "selection_reason",
                "missing_feature_count",
                *evidence,
            ],
        ),
        candidates=geojson(
            enriched, ["candidate_id", "proposal_kind", "representative_evidence_type"]
        ),
        hydrology=dict(
            mapped=geojson(mapped_water, ["name"]),
            derived=geojson(drainage, ["threshold_km2"]),
            intersections=geojson(visible_events, ["candidate_id", "evidence_source"]),
        ),
        diagnostics=dict(
            raw_candidates=len(enriched),
            merged_candidates=len(candidates),
            physical_sites=len(sites),
            crossing_proposals=len(visible_events),
            evidence_counts=sites.evidence_type.value_counts().to_dict(),
            default_candidates_visible=False,
        ),
    )
