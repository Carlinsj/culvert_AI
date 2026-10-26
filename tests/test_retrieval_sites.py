import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from staging.retrieval_sites import identity_groups, ranked_identity


def candidates(xs=(0, 5, 10), channels=("a", "a", "a")):
    return gpd.GeoDataFrame(
        [
            dict(
                candidate_id=f"candidate-{i}",
                road_id="unseen-road",
                road_part_index=0,
                route_id="unknown-route",
                road_chainage_m=float(x),
                channel_key=channel,
                proposal_kind="dem_crossing",
                model_score=0.2 + i * 0.1,
                localization_evidence='["physical_crossing"]',
                geometry=Point(500000 + x, 4650000),
            )
            for i, (x, channel) in enumerate(zip(xs, channels))
        ],
        crs=32618,
    )


def test_duplicate_coordinates_and_multiple_source_records_preserve_all_evidence():
    d = candidates((0, 0), ("a", "a"))
    d["team"] = ["first team", "second team"]
    d["proposal_kind"] = ["mapped_crossing", "dem_crossing"]
    result = identity_groups(d)
    assert len(result) == 1 and result.iloc[0].member_count == 2
    assert set(json.loads(result.iloc[0].member_candidate_ids)) == set(d.candidate_id)
    evidence = json.loads(result.iloc[0].member_evidence)
    assert {p["proposal_kind"] for p in evidence} == {"mapped_crossing", "dem_crossing"}
    assert [p["model_score"] for p in evidence] == d.model_score.tolist()


def test_two_distinct_close_culverts_or_conflicting_channels_never_merge():
    d = candidates((0, 1), ("first-channel", "second-channel"))
    assert len(identity_groups(d, 40)) == 2
    d["geometry"] = [Point(500000, 4650000)] * 2
    assert len(identity_groups(d, 40)) == 2


@pytest.mark.parametrize(
    "field,value", [("road_id", "intersection-other-road"), ("road_part_index", 1)]
)
def test_never_merge_across_roads_intersections_or_parts(field, value):
    d = candidates((0, 0), ("a", "a"))
    d.loc[1, field] = value
    assert len(identity_groups(d)) == 2


def test_nearby_unknown_anchors_or_distinct_local_extrema_are_not_duplicates():
    d = candidates((0, 5), ("", ""))
    d["proposal_kind"] = "internal_road_probe"
    d["localization_evidence"] = '["profile_local_peak"]'
    assert len(identity_groups(d, 40)) == 2


def test_missing_channel_cannot_bridge_two_conflicting_identities():
    d = candidates((0, 0, 0), ("", "a", "b"))
    assert len(identity_groups(d)) == 2


def test_complete_link_bound_prevents_dbscan_chaining_and_meander_shortcuts():
    d = candidates((0, 15, 30), ("a", "a", "a"))
    result = identity_groups(d, 20)
    assert sorted(result.member_count) == [1, 2]
    d = candidates((0, 1), ("a", "a"))
    d.loc[1, "road_chainage_m"] = 500
    assert len(identity_groups(d, 20)) == 2


def test_medoid_is_an_existing_point_not_inventory_or_score_selected():
    d = candidates()
    d["model_score"] = [0.99, 0.1, 0.9]
    a = identity_groups(d)
    assert a.iloc[0].geometry.equals(d.iloc[1].geometry)
    assert a.iloc[0].model_score == 0.1
    d["target"] = 1
    d["distance_to_known"] = [0, 100, 0]
    b = identity_groups(d.sample(frac=1, random_state=7))
    assert a.physical_site_id.tolist() == b.physical_site_id.tolist()
    assert a.geometry.equals(b.geometry)


def test_fixed_rank_policies_share_same_identity_groups_and_own_scores():
    d = candidates((0, 5, 100), ("a", "a", "b"))
    a = ranked_identity(d, "raw_rf").set_index("physical_site_id").sort_index()
    b = ranked_identity(d, "dynamic_route_z").set_index("physical_site_id").sort_index()
    assert a.index.tolist() == b.index.tolist() and a.geometry.equals(b.geometry)
    pd.testing.assert_series_equal(a.model_score, b.model_score)
    np.testing.assert_array_equal(sorted(a.global_rank), np.arange(1, len(a) + 1))


def test_crs_and_distance_errors_fail_closed_and_empty_is_valid():
    d = candidates()
    with pytest.raises(ValueError, match="EPSG"):
        identity_groups(d.to_crs(4326))
    with pytest.raises(ValueError):
        identity_groups(d, -1)
    with pytest.raises(ValueError):
        identity_groups(d, np.nan)
    with pytest.raises(ValueError):
        identity_groups(pd.concat([d, d]))
    assert identity_groups(d.iloc[:0]).empty


def test_matching_does_not_count_one_group_as_multiple_inventory_records():
    from scripts.validate_inventory_heldout_roads import match_one_to_one

    d = identity_groups(candidates())
    assert len(match_one_to_one(d, np.array([[500004, 4650000], [500006, 4650000]]))) == 1


def test_geojson_keeps_exact_group_ids_member_evidence_and_coordinates():
    from staging.hydrology_service import geojson
    from staging.feature_service import META

    d = ranked_identity(candidates())
    payload = geojson(d, META)
    assert len(payload["features"]) == len(d)
    feature = payload["features"][0]
    assert feature["properties"]["physical_site_id"] == d.iloc[0].physical_site_id
    assert feature["properties"]["member_count"] == 3
    actual = (
        gpd.GeoSeries([Point(*feature["geometry"]["coordinates"])], crs=4326).to_crs(32618).iloc[0]
    )
    assert actual.distance(d.iloc[0].geometry) < 1e-6
    assert len(json.loads(feature["properties"]["member_evidence"])) == 3


def test_request_routes_grouping_option_without_mutating_production(monkeypatch):
    from io import BytesIO
    import scripts.serve_feature_staging as server

    captured = []
    monkeypatch.setattr(server, "analyze", lambda *args: captured.append(args) or {"ok": True})
    h = object.__new__(server.Handler)
    responses = []
    h.respond = lambda *args: responses.append(args)
    body = b'{"latitude":42,"longitude":-74,"grouping":"identity20"}'
    h.path = "/api/analysis/nearby"
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = BytesIO(body)
    h.do_POST()
    assert responses[-1][0] == 200 and captured[0][-1] == "identity20"
