import copy
import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, LineString

from staging.feature_sites import proposals, local_peaks, ranked, check_model, spacing_links


def anchors(values=None):
    values = values or [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    return gpd.GeoDataFrame(
        [
            dict(
                candidate_id=f"p{i:03d}",
                road_id="unseen-road",
                route_id="opaque",
                road_part_index=0,
                road_chainage_m=i * 10.0,
                proposal_kind="internal_road_probe",
                route_elevation_dip_60m=float(v),
                road_profile_available=True,
                route_low_point_score=0.8 if v else 0,
                upstream_area_m2=0.0,
                negative_tpi_9x9_m=0.0,
                model_score=0.5,
                geometry=Point(i * 10, 0),
            )
            for i, v in enumerate(values)
        ],
        crs=32618,
    )


def test_constant_sampling_is_not_detection_and_can_return_zero():
    d = anchors([0] * 20)
    p = proposals(d)
    assert not p.evidence_admitted.any() and not p.ml_local_peak.any()
    assert ranked(p.loc[p.evidence_admitted]).empty


def test_local_profile_peak_is_localized_not_all_positive_anchors():
    d = anchors([0.2] * 7 + [1] + [0.2] * 7)
    p = proposals(d)
    assert p.loc[p.evidence_admitted, "candidate_id"].tolist() == ["p007"]
    assert p.loc[p.evidence_admitted].geometry.equals(d.iloc[[7]].geometry)


def test_ground_truth_columns_do_not_affect_admission_or_move_geometry():
    d = anchors()
    a = proposals(d)
    d["official_x"] = np.arange(len(d))
    d["target"] = 1
    d["distance_to_known"] = 0
    b = proposals(d)
    np.testing.assert_array_equal(a.evidence_admitted, b.evidence_admitted)
    assert a.geometry.equals(b.geometry)


def test_missing_and_window_edge_do_not_create_peaks():
    d = anchors()
    d.loc[4, "route_elevation_dip_60m"] = np.nan
    assert not local_peaks(d, "route_elevation_dip_60m").any()
    short = anchors([0, 1, 0])
    assert not local_peaks(short, "route_elevation_dip_60m").any()


def test_positive_flow_peak_requires_actual_concavity():
    d = anchors([0] * 15)
    d.loc[7, "upstream_area_m2"] = 10000
    assert not proposals(d).evidence_admitted.any()
    d.loc[7, "negative_tpi_9x9_m"] = 0.1
    assert proposals(d).loc[7, "flow_valley_local_peak"]


def test_ml_peak_alone_is_diagnostic_not_a_physical_admission():
    d = anchors([0] * 15)
    d.loc[7, "model_score"] = 0.9
    p = proposals(d)
    assert p.loc[7, "ml_local_peak"] and not p.evidence_admitted.any()


def test_crossings_admitted_but_nearest_approach_is_not():
    d = anchors([0] * 15)
    d.loc[0, "proposal_kind"] = "mapped_crossing"
    d.loc[1, "proposal_kind"] = "dem_crossing"
    d.loc[2, "proposal_kind"] = "nearest_road_stream_approach"
    assert proposals(d).loc[lambda x: x.evidence_admitted, "candidate_id"].tolist() == [
        "p000",
        "p001",
    ]


def test_extrema_never_bridge_road_parts_or_other_roads():
    d = anchors()
    d.loc[8:, "road_part_index"] = 1
    assert not local_peaks(d, "route_elevation_dip_60m").any()


def test_plateau_tie_and_shuffle_are_deterministic():
    d = anchors([0] * 7 + [1, 1] + [0] * 7)
    a = proposals(d).set_index("candidate_id").evidence_admitted.sort_index()
    b = (
        proposals(d.sample(frac=1, random_state=5))
        .set_index("candidate_id")
        .evidence_admitted.sort_index()
    )
    pd.testing.assert_series_equal(a, b)
    assert a.sum() == 1


def test_consolidation_is_deterministic_keeps_original_coordinate_and_score():
    d = anchors()
    d.loc[[6, 7, 8], "proposal_kind"] = "dem_crossing"
    d.loc[[6, 7, 8], "model_score"] = [0.8, 0.9, 0.7]
    p = proposals(d)
    p = p.loc[p.evidence_admitted]
    a = ranked(p, "raw_rf")
    b = ranked(p.sample(frac=1, random_state=1), "raw_rf")
    assert a.candidate_id.tolist() == b.candidate_id.tolist()
    assert a.iloc[0].geometry.equals(Point(70, 0)) and a.iloc[0].model_score == 0.9
    assert "physical_crossing" in json.loads(a.iloc[0].member_localization_evidence)


def test_model_guard_rejects_frozen_and_schema_reorder():
    columns = [f"f{i}" for i in range(48)]
    bundle = dict(
        experimental=True,
        contract=dict(version="hydrology-sites-v2", feature_columns=columns),
        feature_columns=columns,
        medians=pd.Series(0.0, index=columns),
        model=SimpleNamespace(n_estimators=800),
    )
    assert check_model(bundle, columns) == columns
    bad = copy.deepcopy(bundle)
    bad["experimental"] = False
    with pytest.raises(ValueError):
        check_model(bad, columns)
    with pytest.raises(ValueError):
        check_model(bundle, columns[::-1])


def test_spacing_follows_actual_line_not_nearest_neighbours_or_other_branches():
    roads = gpd.GeoDataFrame(
        [dict(id="unseen-road", geometry=LineString([(0, 0), (50, 0), (50, 100)]))], crs=32618
    )
    d = anchors()
    d = d.iloc[[2, 8, 12]].copy()
    d["geometry"] = [Point(20, 0), Point(50, 30), Point(50, 70)]
    links = spacing_links(d, roads, "sampling-anchor")
    assert links.distance_m.tolist() == [60, 40] and links.iloc[0].geometry.length == 60
    d.loc[d.index[-1], "road_part_index"] = 1
    assert len(spacing_links(d, roads, "sampling-anchor")) == 1


def test_server_paths_and_invalid_payload_fail_closed(monkeypatch):
    from io import BytesIO
    from scripts.serve_feature_staging import Handler

    h = object.__new__(Handler)
    responses = []
    h.respond = lambda *args: responses.append(args)
    h.path = "/../../models/flow_c_frozen_v1.joblib"
    h.do_GET()
    assert responses[-1][0] == 404
    for body in [
        b"[]",
        b"{}",
        b'{"latitude":"bad","longitude":0}',
        b'{"latitude":0,"longitude":0,"mode":"frozen"}',
    ]:
        h.path = "/api/analysis/nearby"
        h.headers = {"Content-Length": str(len(body))}
        h.rfile = BytesIO(body)
        h.do_POST()
        assert responses[-1][0] == 400


def test_display_provenance_distinguishes_new_flow_support_from_legacy_fallback():
    from staging.feature_service import display_metadata

    d = anchors([0] * 15)
    d.loc[7, "upstream_area_m2"] = 10000
    d.loc[7, "negative_tpi_9x9_m"] = 0.2
    p = proposals(d)
    p["evidence_type"] = "road_corridor_fallback"
    original_xy = p.geometry.copy()
    original_scores = p.model_score.copy()
    view = display_metadata(p, "Experimental proposal")
    assert view.loc[7, "evidence_type"] == "flow-and-concavity-supported"
    assert view.loc[7, "legacy_evidence_type"] == "road_corridor_fallback"
    assert view.geometry.equals(original_xy)
    pd.testing.assert_series_equal(view.model_score, original_scores)
    assert display_metadata(p.iloc[:0], "Empty proposals").empty


@pytest.mark.parametrize("control", ["global_chainage", "road_part", "road_part_block2km"])
def test_random_phase_controls_preserve_budgets_and_geometry_strata(control):
    from scripts.validate_feature_spatial_null import plan, draw
    import shapely

    roads = gpd.GeoDataFrame(
        [
            dict(segment_id="a", geometry=LineString([(0, 0), (4000, 0)])),
            dict(segment_id="b", geometry=LineString([(0, 100), (800, 100)])),
        ],
        crs=32618,
    )
    frame = gpd.GeoDataFrame(
        [
            dict(road_id=r, road_part_index=0, road_chainage_m=x, geometry=Point(x, y))
            for r, x, y in [
                ("a", 100, 0),
                ("a", 500, 0),
                ("a", 2100, 0),
                ("a", 2500, 0),
                ("b", 200, 100),
            ]
        ],
        crs=32618,
    )
    prepared = plan(roads, frame, control)
    first = draw(prepared, np.random.default_rng(42), control)
    again = draw(prepared, np.random.default_rng(42), control)
    assert len(first) == len(frame) and np.all(shapely.equals(first, again))
    coords = np.c_[shapely.get_x(first), shapely.get_y(first)]
    if control != "global_chainage":
        assert (coords[:, 1] == 0).sum() == 4 and (coords[:, 1] == 100).sum() == 1
    if control == "road_part_block2km":
        assert ((coords[:, 1] == 0) & (coords[:, 0] < 2000)).sum() == 2
        assert ((coords[:, 1] == 0) & (coords[:, 0] >= 2000)).sum() == 2
    frame["target"] = 1
    other = draw(plan(roads, frame, control), np.random.default_rng(42), control)
    assert np.all(shapely.equals(first, other))
