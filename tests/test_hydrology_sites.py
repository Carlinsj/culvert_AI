import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from staging.hydrology_sites import (
    VERSION,
    apply_feature_contract,
    consolidate_events,
    detect_events,
    profile_values,
    rank_physical_sites,
    raster_sample,
    repair_profiles,
    search_candidates,
)


def frame(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=32618)


def road(identifier="never-seen-road"):
    return frame(
        [
            dict(
                id=identifier,
                name="Unseen Road",
                route_id=identifier,
                geometry=LineString([(0, 0), (200, 0)]),
            )
        ]
    )


def event(name, x, **kwargs):
    return dict(
        candidate_id=name,
        road_id="r",
        road_part_index=0,
        channel_key="channel",
        mapped_water_id="",
        evidence_source="dem",
        proposal_kind="dem_crossing",
        road_chainage_m=x,
        geometry=Point(x, 0),
        **kwargs,
    )


def test_crossings_not_parallel_proximity_and_not_one_sided():
    streams = frame(
        [
            dict(geometry=LineString([(50, -50), (50, 50)])),
            dict(geometry=LineString([(0, 5), (200, 5)])),
            dict(geometry=LineString([(100, 0), (100, 50)])),
        ]
    )
    events, rejected = detect_events(road(), streams, {})
    assert len(events) == 1
    assert events.iloc[0].geometry.equals(Point(50, 0))
    assert events.iloc[0].crossing_angle_degrees == pytest.approx(90)
    assert len(rejected) == 1


def test_shallow_angle_rejected_even_when_intersecting():
    streams = frame([dict(geometry=LineString([(0, -10), (200, 10)]))])
    events, _ = detect_events(road(), streams, {})
    assert events.empty


def test_unseen_road_ids_and_deterministic_probes():
    a = search_candidates(road())
    b = search_candidates(road())
    assert a.candidate_id.tolist() == b.candidate_id.tolist()
    assert a.candidate_id.is_unique
    assert set(a.route_id) == {"never-seen-road"}
    assert len(a) > 10


def test_profile_independent_of_75m_proposals():
    distances = np.arange(0, 201, 10)
    z = np.abs(distances - 100) / 10
    dip, rank, score = profile_values(distances, z, [25, 100, 175], [7.5, 0, 7.5])
    assert dip[1] > 0 and rank[1] > 0 and score[1] > 0
    assert np.isfinite(dip).all()


def test_profile_missing_not_valid_zero():
    x = np.arange(0, 201, 10)
    dip, _, score = profile_values(x, np.zeros(len(x)), [0, 100], [0, 0])
    assert np.isnan(dip[0]) and np.isnan(score[0])
    assert dip[1] == 0 and score[1] == 0
    assert np.isnan(profile_values(x, np.full(len(x), np.nan), [100], [np.nan])[0][0])


def test_missing_road_profile_is_unavailable():
    points = frame([dict(road_id="missing", geometry=Point(0, 0))])
    fixed = repair_profiles(points, road(), "not-opened.tif")
    assert fixed.route_elevation_dip_60m.isna().all()
    assert not fixed.road_profile_available.any()


def test_consolidation_bounded_not_transitive_and_preserves_geometry_scores():
    events = frame([event("a", 0), event("b", 15), event("c", 30)])
    result = consolidate_events(events, [0.2, 0.9, 0.8])
    assert len(result) == 2
    first = result.loc[result.candidate_id.eq("a")].iloc[0]
    assert first.geometry.equals(Point(0, 0))
    assert first.model_score == 0.2 and first.maximum_member_model_score == 0.9
    assert first.member_count == 2


@pytest.mark.parametrize(
    "column,value",
    [
        ("road_id", "parallel-road"),
        ("road_part_index", 1),
        ("channel_key", "second-channel"),
        ("road_chainage_m", 100),
    ],
)
def test_nearby_separate_topology_not_merged(column, value):
    rows = [event("a", 0), event("b", 5)]
    rows[1][column] = value
    assert len(consolidate_events(frame(rows), [0.1, 0.9])) == 2


def test_distinct_mapped_waters_not_merged():
    rows = [event("a", 0), event("b", 5)]
    for i, row in enumerate(rows):
        row.update(proposal_kind="mapped_crossing", mapped_water_id=str(i))
    assert len(consolidate_events(frame(rows), [0.1, 0.9])) == 2


def test_order_and_representatives_are_deterministic():
    events = frame([event("c", 25), event("a", 0), event("b", 10)])
    a = rank_physical_sites(consolidate_events(events, [0.5, 0.5, 0.5]))
    b = rank_physical_sites(consolidate_events(events.iloc[::-1], [0.5, 0.5, 0.5]))
    assert a.physical_site_id.tolist() == b.physical_site_id.tolist()
    assert a.candidate_id.tolist() == sorted(a.candidate_id)


def test_old_contract_is_rejected():
    with pytest.raises(ValueError, match="v2"):
        apply_feature_contract(pd.DataFrame(), {"version": "frozen-v1"})


def test_raster_zero_nodata_and_outside_are_distinct(tmp_path):
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs=32618,
        transform=from_origin(0, 20, 10, 10),
        nodata=-9999,
    ) as raster:
        raster.write(np.array([[0, -9999], [2, 3]], dtype="float32"), 1)
    values = raster_sample(path, [(5, 15), (15, 15), (100, 100)])
    assert values[0] == 0
    assert np.isnan(values[1:]).all()


def test_model_loader_requires_explicit_new_contract(monkeypatch):
    from staging import hydrology_service as service

    service.load_model.cache_clear()
    monkeypatch.setattr(
        service.joblib,
        "load",
        lambda _: {
            "experimental": True,
            "contract": {"version": VERSION, "feature_columns": ["a"]},
            "feature_columns": ["b"],
        },
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        service.load_model("RF")
    service.load_model.cache_clear()


@pytest.mark.parametrize(
    "latitude,longitude,radius",
    [(91, 0, 1000), (42, -74, 0), (42, -74, 5000), (float("nan"), 0, 1000)],
)
def test_invalid_analysis_input_fails_before_model_load(latitude, longitude, radius):
    from staging.hydrology_service import analyze

    with pytest.raises(ValueError, match="coordinates"):
        analyze(latitude, longitude, radius)


def test_outside_terrain_fails_closed_without_model_substitution():
    from staging.hydrology_service import analyze

    with pytest.raises(ValueError, match="outside"):
        analyze(0, 0, 1000)
