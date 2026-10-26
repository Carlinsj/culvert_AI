import json
from io import BytesIO
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point
from staging.hybrid_sites import (
    attach_road_topology,
    classify_evidence,
    cluster_sites,
    historical_candidates,
    overlapping_proposals,
    rank_sites,
)


def frame(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=32618)


def point(name, x, score=0.5, **kw):
    return dict(
        candidate_id=name,
        road_id="unseen",
        road_part_index=0,
        road_chainage_m=x,
        route_id="unseen",
        proposal_kind="internal_road_probe",
        model_score=score,
        geometry=Point(x, 0),
        **kw,
    )


def test_historical_geometry_on_unseen_nonnumbered_network_no_streams():
    roads = frame(
        [
            dict(
                id="new-opaque-id",
                name="Never Named Highway",
                route_id="x",
                geometry=LineString([(0, 0), (100, 0)]),
            )
        ]
    )
    streams = frame([dict(id="parallel", geometry=LineString([(0, 100), (100, 100)]))])
    sites, probes, mapped = historical_candidates(roads, streams)
    assert len(probes) == 10 and len(sites) == 10 and mapped.empty
    assert list(probes.geometry.x) == pytest.approx(list(np.arange(5, 100, 10)))
    assert sites.route_id.eq("x").all()


def test_historical_mapped_approach_not_mislabeled_as_crossing():
    roads = frame([dict(id="r", name="r", route_id="r", geometry=LineString([(0, 0), (100, 0)]))])
    streams = frame([dict(id="s", geometry=LineString([(20, 10), (80, 10)]))])
    _, _, mapped = historical_candidates(roads, streams)
    result = classify_evidence(mapped)
    assert (
        len(result) == 1 and result.stream_supported.all() and not result.crossing_supported.any()
    )


def test_collinear_mapped_overlap_stays_eligible_but_not_crossing_supported():
    roads = frame([dict(id="r", name="r", route_id="r", geometry=LineString([(0, 0), (100, 0)]))])
    streams = frame([dict(id="s", geometry=LineString([(20, 0), (80, 0)]))])
    _, _, mapped = historical_candidates(roads, streams)
    result = classify_evidence(mapped)
    assert len(result) == 1 and result.stream_supported.all()
    assert not result.crossing_supported.any()


def test_fallback_eligibility_and_no_crossing_boost():
    rows = [point("fallback", 0, 0.9), point("crossing", 100, 0.1)]
    rows[1]["proposal_kind"] = "dem_crossing"
    sites = rank_sites(classify_evidence(frame(rows)), "raw_rf")
    assert sites.iloc[0].candidate_id == "fallback"
    assert sites.iloc[0].evidence_type == "road_corridor_fallback"
    assert len(sites) == 2


def test_bounded_clustering_not_transitive():
    sites = cluster_sites(frame([point("a", 0), point("b", 15), point("c", 30)]))
    assert len(sites) == 2
    assert json.loads(sites.iloc[0].member_candidate_ids) == ["a", "b"]
    assert sites.iloc[0].geometry.equals(Point(0, 0))


def test_20m_boundary_is_inclusive_without_expanding_it():
    points = frame([point("a", 0), point("b", 20), point("c", 20.0001)])
    sites = cluster_sites(points)
    assert len(sites) == 2
    assert json.loads(sites.iloc[0].member_candidate_ids) == ["a", "b"]


@pytest.mark.parametrize(
    "column,value", [("road_id", "parallel"), ("road_part_index", 1), ("road_chainage_m", 100)]
)
def test_topology_distinguishes_close_roads_or_hairpins(column, value):
    rows = [point("a", 0), point("b", 5)]
    rows[1][column] = value
    assert len(cluster_sites(frame(rows))) == 2


def test_known_separate_channels_not_merged():
    assert (
        len(
            cluster_sites(
                frame([point("a", 0, channel_key="one"), point("b", 5, channel_key="two")])
            )
        )
        == 2
    )


def test_overlap_then20m_keeps_all_source_provenance_and_actual_score():
    rows = [point("a", 0, 0.8), point("b", 0.5, 0.9), point("c", 10, 0.7)]
    rows[1]["proposal_kind"] = "dem_crossing"
    first = overlapping_proposals(classify_evidence(frame(rows)))
    assert len(first) == 2
    sites = rank_sites(first)
    assert len(sites) == 1 and sites.iloc[0].candidate_id == "a"
    assert sites.iloc[0].model_score == 0.8
    assert sites.iloc[0].geometry.equals(Point(0, 0))
    assert set(json.loads(sites.iloc[0].member_candidate_ids)) == {"a", "b", "c"}
    assert sites.iloc[0].crossing_supported
    assert sites.iloc[0].representative_evidence_type == "road_corridor_fallback"


@pytest.mark.parametrize("policy", ["raw_rf", "dynamic_route_z"])
def test_score_ties_and_input_order_deterministic(policy):
    points = classify_evidence(frame([point("b", 5), point("a", 0), point("c", 100)]))
    a, b = rank_sites(points, policy), rank_sites(points.iloc[::-1], policy)
    assert a.candidate_id.tolist() == b.candidate_id.tolist()
    assert a.physical_site_id.tolist() == b.physical_site_id.tolist()


def test_missing_terrain_is_fallback_not_fake_support():
    rows = frame([point("a", 0), point("b", 30)])
    rows["route_elevation_dip_60m"] = [np.nan, 0.2]
    rows["route_low_point_score"] = [np.nan, 0.5]
    result = classify_evidence(rows)
    assert result.iloc[0].representative_evidence_type == "road_corridor_fallback"
    assert result.iloc[1].representative_evidence_type == "terrain_supported"


def test_unknown_policy_and_invalid_units_fail_closed():
    points = frame([point("a", 0)])
    with pytest.raises(ValueError, match="policy"):
        rank_sites(points, "invented")
    with pytest.raises(ValueError, match="EPSG"):
        cluster_sites(points.to_crs(4326))


def test_invalid_geometry_or_score_fails():
    points = frame([point("a", 0, float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        rank_sites(points)


def test_topology_projection_preserves_proposal_geometry():
    roads = frame(
        [dict(id="unseen", name="road", route_id="route", geometry=LineString([(0, 0), (100, 0)]))]
    )
    points = frame([point("a", 10)])
    points.geometry = [Point(10, 1)]
    fixed = attach_road_topology(points, roads)
    assert fixed.geometry.iloc[0].equals(Point(10, 1)) and fixed.road_chainage_m.iloc[0] == 10


def test_both_policies_preserve_identical_eligible_population():
    rows = [point(str(i), i * 10, score=(i + 1) / 12) for i in range(10)]
    for i, row in enumerate(rows):
        row["route_id"] = "first" if i < 5 else "second"
        row["road_id"] = row["route_id"]
    points = classify_evidence(frame(rows))
    a, b = rank_sites(points, "raw_rf"), rank_sites(points, "dynamic_route_z")

    def members(result):
        return set(x for v in result.member_candidate_ids for x in json.loads(v))

    assert members(a) == members(b) == set(points.candidate_id)
    assert set(a.physical_site_id) == set(b.physical_site_id)


@pytest.mark.parametrize(
    "lat,lon,radius,policy",
    [(91, 0, 1000, "raw_rf"), (0, 0, 5000, "raw_rf"), (42, -74, 1000, "invented")],
)
def test_live_invalid_input_fails_before_model_load(lat, lon, radius, policy):
    from staging.hybrid_service import analyze

    with pytest.raises(ValueError):
        analyze(lat, lon, radius, policy)


def request_handler(payload):
    from scripts.serve_hybrid_staging import Handler

    body = json.dumps(payload).encode()
    handler = object.__new__(Handler)
    handler.path = "/api/analysis/nearby"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    responses = []
    handler.respond = lambda status, result, *args: responses.append((status, result))
    handler.log_error = lambda *args: None
    return handler, responses


def test_api_preserves_fixed_policy_and_response(monkeypatch):
    from scripts import serve_hybrid_staging as server

    received = []

    def analysis(*args):
        received.append(args)
        return {"site_count": 0, "experimental": True}

    monkeypatch.setattr(server, "analyze", analysis)
    handler, result = request_handler(
        dict(location=dict(latitude=42, longitude=-74), radius_m=1000, policy="raw_rf")
    )
    handler.do_POST()
    assert received == [(42, -74, 1000, "raw_rf")]
    assert result == [(200, {"site_count": 0, "experimental": True})]


@pytest.mark.parametrize(
    "exception,status",
    [
        (ValueError("outside coverage"), 400),
        (FileNotFoundError("private model path"), 503),
        (RuntimeError("private diagnostics"), 500),
    ],
)
def test_api_fails_closed_and_does_not_expose_internal_paths(monkeypatch, exception, status):
    from scripts import serve_hybrid_staging as server

    def fail(*args):
        raise exception

    monkeypatch.setattr(server, "analyze", fail)
    handler, result = request_handler(dict(latitude=42, longitude=-74))
    handler.do_POST()
    assert result[0][0] == status
    assert "private" not in json.dumps(result)


def test_static_server_disallows_inventory_and_traversal():
    handler, responses = request_handler({})
    handler.path = "/../data/incoming_nysdot/inventory.xlsx"
    handler.do_GET()
    assert responses[0][0] == 404
