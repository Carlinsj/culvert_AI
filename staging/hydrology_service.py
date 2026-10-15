"""Opt-in experimental API. No inventory reads, production writes or model fallback."""

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box
from threadpoolctl import threadpool_limits

from staging.hydrology_sites import (
    VERSION,
    WARNING,
    THRESHOLDS,
    apply_feature_contract,
    consolidate_events,
    detect_events,
    physical_measurements,
    rank_physical_sites,
    search_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/hydrology_driven_sites_20261015"
HYDRO = ROOT / "data/processed/hydrology/usgs_3dep_10m"


@lru_cache(maxsize=2)
def load_model(family="RF"):
    if family not in {"RF", "HGB"}:
        raise ValueError("Model must be RF or HGB.")
    fitted = joblib.load(OUT / f"{VERSION}_{family}_experimental.joblib")
    if not fitted.get("experimental") or fitted.get("contract", {}).get("version") != VERSION:
        raise ValueError("Incompatible model: repaired features require an experimental v2 model.")
    if fitted["feature_columns"] != fitted["contract"]["feature_columns"]:
        raise ValueError("Model/feature contract mismatch.")
    return fitted


@lru_cache(maxsize=1)
def datasets():
    return (
        gpd.read_file(ROOT / "data/raw/roads.gpkg").to_crs(32618),
        gpd.read_file(ROOT / "data/raw/streams.gpkg").to_crs(32618),
        {key: gpd.read_parquet(HYDRO / f"drainage_{key}.parquet") for key in THRESHOLDS},
    )


def geojson(frame, columns):
    # pandas JSON converts unavailable measurements to null, never to zero.
    return __import__("json").loads(
        frame[[c for c in columns if c in frame] + ["geometry"]]
        .to_crs(4326)
        .to_json(drop_id=True, na="null")
    )


@lru_cache(maxsize=4)
def analyze(latitude, longitude, radius_m=1500.0, family="RF"):
    latitude, longitude, radius_m = float(latitude), float(longitude), float(radius_m)
    if not (
        np.isfinite([latitude, longitude, radius_m]).all()
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and 200 <= radius_m <= 4000
    ):
        raise ValueError("Use valid coordinates and a radius between 200 and 4000 metres.")
    center = gpd.GeoSeries([Point(longitude, latitude)], crs=4326).to_crs(32618).iloc[0]
    area = center.buffer(radius_m)
    with rasterio.open(HYDRO / "dem_projected.tif") as raster:
        if raster.crs.to_epsg() != 32618 or not box(*raster.bounds).covers(area):
            raise ValueError(
                "This area is outside the supported terrain coverage. Explore a supported area."
            )
    fitted = load_model(family)
    all_roads, streams, networks = datasets()
    indices = all_roads.sindex.query(area, predicate="intersects")
    roads = all_roads.iloc[indices].copy().reset_index(drop=True)
    if roads.empty:
        raise ValueError(
            "No supported road data here. Choose a point within the Ulster data coverage."
        )
    # Complete road parts preserve stable IDs, topology and profile shoulders.
    # Analysis is spatially restricted after measurement, not by moving sites.
    roads["route_id"] = roads["name"].fillna(roads.id.astype(str)).astype(str)
    probes = search_candidates(roads)
    probes = probes.loc[probes.geometry.distance(center) <= radius_m + 75].reset_index(drop=True)
    events, rejected = detect_events(roads, streams, networks)
    events = events.loc[events.geometry.distance(center) <= radius_m].reset_index(drop=True)
    points = gpd.GeoDataFrame(pd.concat([probes, events], ignore_index=True), crs=32618)
    if points.empty:
        raise ValueError("No usable road geometry in the selected area.")
    measured = physical_measurements(points, all_roads, streams, HYDRO)
    transformed = apply_feature_contract(
        measured,
        fitted["contract"],
        normalization_context=measured.loc[measured.proposal_kind.eq("internal_road_probe")],
    )
    columns = fitted["feature_columns"]
    values = (
        transformed[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    with threadpool_limits(limits=2):
        scores = fitted["model"].predict_proba(values.fillna(fitted["medians"]))[:, 1]
    transformed["model_score"] = scores
    transformed["missing_feature_count"] = values.isna().sum(axis=1)
    measured_events = transformed.set_index("candidate_id").reindex(events.candidate_id)
    enriched = events.copy()
    evidence_columns = [
        "model_score",
        "flow_accumulation_cells",
        "upstream_area_m2",
        "slope_degrees",
        "distance_to_nearest_stream_m",
        "route_elevation_dip_60m",
        "hydrology_edge_risk",
        "road_profile_available",
        "missing_feature_count",
    ]
    for column in evidence_columns:
        if column in measured_events:
            enriched[column] = measured_events[column].to_numpy()
    sites = consolidate_events(enriched, enriched.model_score.to_numpy() if len(enriched) else None)
    sites = rank_physical_sites(sites) if len(sites) else sites
    sites["model_version"] = fitted["model_version"]
    sites["selection_reason"] = (
        "Transverse, two-sided drainage-road intersection; ranked by uncalibrated model score. DEM crossings represent concentrated flow, not verified culverts."
    )
    stream_subset = streams.iloc[streams.sindex.query(area, predicate="intersects")]
    drainage = networks["0p1km2"]
    drainage_subset = drainage.iloc[drainage.sindex.query(area, predicate="intersects")]
    return dict(
        experimental=True,
        field_verification_notice=WARNING,
        model=dict(
            version=fitted["model_version"],
            feature_contract=VERSION,
            family=family,
            calibrated=False,
            independent_validation=False,
        ),
        analysis=dict(
            latitude=latitude,
            longitude=longitude,
            center=dict(latitude=latitude, longitude=longitude),
            source="current_location",
            analysis_crs="EPSG:32618",
            coverage="available",
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
                "global_rank",
                "proposal_kind",
                "evidence_source",
                "support_sources",
                "member_count",
                "member_candidate_ids",
                "maximum_member_model_score",
                "crossing_angle_degrees",
                "bilateral_drainage",
                "bridge_context",
                "model_version",
                "selection_reason",
                *evidence_columns,
            ],
        ),
        candidates=geojson(transformed, ["candidate_id", "proposal_kind", "model_score"]),
        hydrology=dict(
            mapped=geojson(stream_subset, ["name"]),
            derived=geojson(drainage_subset, ["threshold_km2"]),
            intersections=geojson(events, ["candidate_id", "evidence_source"]),
        ),
        diagnostics=dict(
            internal_probe_count=len(probes),
            crossing_proposals=len(events),
            physical_sites=len(sites),
            rejected_contacts=len(rejected),
            default_layer="recommended_sites",
            all_candidates_default_visible=False,
        ),
    )
