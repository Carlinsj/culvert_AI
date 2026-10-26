import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from staging.discovery_sites import (
    drainage_links,
    crossing_links,
    adaptive_proposals,
    physical_groups,
    ordered,
)
from scripts.diagnose_summer_discovery import primary_failure
from scripts.validate_inventory_heldout_roads import match_one_to_one


def roads():
    return gpd.GeoDataFrame(
        [
            dict(
                id="new-unseen-road-XYZ",
                name="Unknown road",
                route_id="opaque",
                geometry=LineString([(0, 15), (40, 15)]),
            )
        ],
        crs=32618,
    )


def test_whitebox_pointer_encoding_and_stable_global_cell_ids():
    p = np.zeros((4, 4))
    p[1, 1] = 8
    a = np.full((4, 4), 20000)
    links = drainage_links(p, a, from_origin(0, 40, 10, 10), 100, 200)
    assert links.iloc[0].geometry.equals(LineString([(15, 25), (15, 15)]))
    assert links.iloc[0].link_key == "d8:101:201:102:201"
    p[1, 1] = 1
    assert list(drainage_links(p, a, from_origin(0, 40, 10, 10)).iloc[0].geometry.coords)[1] == (
        25,
        35,
    )


def test_only_actual_transverse_drainage_intersections_are_proposed():
    links = gpd.GeoDataFrame(
        [
            dict(link_key="a", upstream_area_m2=15000, geometry=LineString([(20, 10), (20, 20)])),
            dict(
                link_key="parallel",
                upstream_area_m2=15000,
                geometry=LineString([(0, 15), (40, 15)]),
            ),
            dict(
                link_key="touch", upstream_area_m2=15000, geometry=LineString([(30, 15), (30, 20)])
            ),
        ],
        crs=32618,
    )
    events = crossing_links(roads(), links)
    assert len(events) == 1 and events.iloc[0].geometry.equals(Point(20, 15))
    assert events.iloc[0].road_id == "new-unseen-road-XYZ"
    assert events.iloc[0].bilateral_drainage
    shuffled = crossing_links(roads(), links.sample(frac=1, random_state=4))
    assert events.candidate_id.tolist() == shuffled.candidate_id.tolist()


def probes():
    return gpd.GeoDataFrame(
        [
            dict(
                candidate_id=f"p{i}",
                road_id="r",
                road_part_index=0,
                road_chainage_m=i * 10,
                route_id="r",
                proposal_kind="internal_road_probe",
                elevation_m=z,
                upstream_area_m2=100,
                negative_tpi_9x9_m=0,
                model_score=0.5,
                geometry=Point(i * 10, 0),
            )
            for i, z in enumerate([2, 1, 0, 1, 2, 1, 0, 1, 2])
        ],
        crs=32618,
    )


def test_short_profile_extrema_not_every_computational_anchor():
    frame = probes()
    out = adaptive_proposals(frame)
    assert out.loc[out.discovery_admitted].candidate_id.tolist() == ["p2", "p6"]
    frame["elevation_m"] = 1
    assert not adaptive_proposals(frame).discovery_admitted.any()
    frame["elevation_m"] = np.nan
    assert not adaptive_proposals(frame).discovery_admitted.any()


def test_candidate_generation_is_inventory_score_and_input_order_independent():
    frame = probes()
    before = adaptive_proposals(frame).set_index("candidate_id").discovery_admitted.sort_index()
    frame["target"] = 1
    frame["distance_to_inventory"] = 0
    frame["model_score"] = np.arange(len(frame))
    after = (
        adaptive_proposals(frame.sample(frac=1, random_state=3))
        .set_index("candidate_id")
        .discovery_admitted.sort_index()
    )
    pd.testing.assert_series_equal(before, after)


def test_grouping_preserves_close_distinct_channels_and_road_identity():
    frame = probes().iloc[:3].copy()
    frame["channel_key"] = ["a", "b", "b"]
    frame["proposal_kind"] = "dem_crossing"
    result = physical_groups(frame)
    assert len(result) == 2
    assert set(sum([json.loads(v) for v in result.member_candidate_ids], [])) == set(
        frame.candidate_id
    )
    frame.loc[frame.index[2], "road_id"] = "other"
    assert len(physical_groups(frame)) == 3


def test_one_prediction_cannot_claim_two_close_recorded_culverts():
    frame = probes().iloc[[2]]
    matches = match_one_to_one(frame, np.array([[20, 0], [21, 0]]), 20)
    assert len(matches) == 1
    assert len(match_one_to_one(frame.iloc[:0], np.array([[20, 0]]), 20)) == 0


def test_ordering_uses_own_score_and_ids_and_preserves_coordinates():
    frame = physical_groups(probes().iloc[[2, 6]])
    a = ordered(frame)
    b = ordered(frame.sample(frac=1, random_state=1))
    assert a.candidate_id.tolist() == b.candidate_id.tolist() == ["p2", "p6"]
    assert a.geometry.equals(b.geometry)
    assert a.model_score.tolist() == [0.5, 0.5]
    frame["model_score"] = np.nan
    with pytest.raises(ValueError):
        ordered(frame)


def diagnostic(**kwargs):
    values = dict(
        Latitude=42,
        Longitude=-74,
        query_success=True,
        inside_terrain=True,
        full_list_matched20=False,
        road_distance_m=5,
        nearest_raw_owned_m=5,
        nearest_raw_query_m=5,
        nearest_proposals_owned_m=50,
        nearest_proposals_query_m=50,
        nearest_sites_owned_m=60,
        nearest_sites_query_m=60,
    )
    return SimpleNamespace(**(values | kwargs))


def test_failure_accounting_exclusive_and_never_blames_top25_for_full_list_loss():
    assert primary_failure(diagnostic())[1] == "physical_evidence_admission"
    assert (
        primary_failure(diagnostic(query_success=False, inside_terrain=False))[1]
        == "outside_loaded_terrain"
    )
    assert primary_failure(diagnostic(query_success=False))[0] == "No road coverage"
    assert (
        primary_failure(diagnostic(nearest_proposals_owned_m=5))[1]
        == "score_ordered_representative"
    )
    assert (
        primary_failure(diagnostic(nearest_proposals_owned_m=5, nearest_sites_owned_m=5))[1]
        == "one_to_one_capacity_competition"
    )
    assert (
        primary_failure(diagnostic(nearest_raw_owned_m=30, nearest_raw_query_m=30))[1]
        == "raw_candidate_outside20m"
    )
    assert primary_failure(diagnostic(full_list_matched20=True))[0] == "Matched"


def test_compatibility_guard_rejects_old_or_incomplete_features_before_predict():
    from staging.discovery_service import compatible_scores

    with pytest.raises(ValueError, match="exact saved v2"):
        compatible_scores(
            pd.DataFrame(),
            pd.DataFrame(),
            dict(feature_columns=[], experimental=False, contract={}),
        )


def test_duplicate_ids_are_rejected_not_silently_merged():
    frame = probes().iloc[:2].copy()
    frame["candidate_id"] = "duplicate"
    with pytest.raises(ValueError, match="unique"):
        physical_groups(frame)


def test_adaptive_computational_windows_respect_actual_road_and_chainage():
    from scripts.adaptive_discovery_control import adaptive

    fine = probes()
    coarse = fine.iloc[[0, 8]].copy()
    seeds = fine.iloc[[2]].copy()
    selected = adaptive(coarse, fine, seeds, 15)
    assert selected.candidate_id.tolist() == ["p0", "p1", "p2", "p3", "p8"]
    fine.loc[fine.index[1], "road_id"] = "other-road"
    fine.loc[fine.index[3], "road_chainage_m"] = 500
    assert adaptive(coarse, fine, seeds, 15).candidate_id.tolist() == ["p0", "p2", "p8"]


def test_research_mode_is_explicit_and_inventory_overlay_not_in_inference(monkeypatch):
    from io import BytesIO
    import staging.discovery_service as service
    from scripts.serve_feature_staging import Handler

    calls = []
    monkeypatch.setattr(service, "analyze", lambda *args: calls.append(args) or {"site_count": 0})
    h = object.__new__(Handler)
    h.path = "/api/analysis/nearby"
    body = json.dumps(dict(latitude=42, longitude=-74, mode="physical_discovery")).encode()
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = BytesIO(body)
    responses = []
    h.respond = lambda *args: responses.append(args)
    h.do_POST()
    assert responses[-1][0] == 200 and calls == [(42, -74, 1500, "dynamic_route_z")]
    import inspect

    source = inspect.getsource(service)
    assert "inventory_coverage" not in source and ".fit(" not in source
