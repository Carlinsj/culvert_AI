import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from staging import selective_sites as s
from staging import site_ranker


def points():
    records = []
    for i, x in enumerate([0.0, 5.0, 10.0, 30.0, 100.0]):
        records.append(
            dict(
                candidate_id=f"p{i}",
                road_id="arbitrary-unseen-road",
                road_name="road",
                route_id="route",
                road_part_index=0,
                road_chainage_m=x,
                channel_key="",
                geometry=Point(500000 + x, 4600000),
                model_score=0.8 - i * 0.01,
                score_RF=0.8 - i * 0.01,
                score_HGB=0.2,
                proposal_kind="internal_road_probe",
                adaptive_profile=True,
                adaptive_flow=False,
                road_profile_available=True,
                upstream_area_m2=2000.0,
                negative_tpi_9x9_m=1.0,
                route_elevation_dip_60m=0.5,
                route_low_point_score=0.5,
                valley_depth_9x9_m=1.0,
                member_candidate_ids=json.dumps([f"p{i}"]),
                member_evidence="[]",
                member_count=1,
            )
        )
    return gpd.GeoDataFrame(records, crs=32618)


def test_selective_preserves_baseline_geometry_scores_and_members():
    hybrid = points()
    base = hybrid.iloc[[0]].copy()
    base["member_candidate_ids"] = '["p0", "p1"]'
    result = s.population(base, hybrid)
    assert set(result.candidate_id) == {"p0", "p2", "p3", "p4"}
    assert result.iloc[0].geometry.equals(base.iloc[0].geometry)
    assert result.iloc[0].score_RF == base.iloc[0].score_RF
    assert json.loads(result.iloc[0].member_candidate_ids) == ["p0", "p1"]


def test_no_stream_requirement_but_required_missing_features_reject():
    frame = points()
    frame["distance_to_nearest_stream_m"] = 100000
    assert s.admission(frame).all()
    frame["upstream_area_m2"] = np.nan
    assert not s.admission(frame).any()
    frame["proposal_kind"] = "small_dem_crossing"
    frame["upstream_area_at_link_m2"] = 10000
    assert s.admission(frame, "balanced").all()
    assert not s.admission(frame, "corroborated").any()


def test_candidate_admission_ignores_scores_labels_inventory():
    frame = points()
    expected = s.admission(frame)
    frame["score_RF"] = 0
    frame["score_HGB"] = 1
    frame["target"] = 0
    frame["nearest_inventory_m"] = np.arange(len(frame))
    pd.testing.assert_series_equal(expected, s.admission(frame))


def test_unknown_rule_rejected():
    with pytest.raises(ValueError):
        s.admission(points(), "look-at-labels")


def test_close_distinct_culverts_unknown_and_conflicting_channels_survive():
    frame = points()
    frame["channel_key"] = ["a", "b", "", "c", ""]
    result = s.anchored_groups(s.population(frame.iloc[[0]], frame))
    assert len(result) == len(frame)


def test_nan_channel_identity_is_unknown_not_a_shared_drainage_path():
    frame = points()
    frame["channel_key"] = np.nan
    result = s.anchored_groups(s.population(frame.iloc[[0]], frame))
    assert len(result) == len(frame)


def test_baseline_anchor_not_moved_and_member_sources_preserved():
    frame = points()
    frame["channel_key"] = "same"
    result = s.anchored_groups(s.population(frame.iloc[[0]], frame))
    assert len(result) == 3
    anchor = result.loc[result.candidate_id.eq("p0")].iloc[0]
    assert anchor.geometry.equals(frame.iloc[0].geometry)
    assert set(json.loads(anchor.member_candidate_ids)) == {"p0", "p1", "p2"}
    assert anchor.score_RF == frame.iloc[0].score_RF
    assert json.loads(anchor.member_evidence)


def test_never_merge_two_baseline_sites_even_with_same_channel():
    frame = points()
    frame["channel_key"] = "same"
    result = s.anchored_groups(s.population(frame.iloc[:2], frame))
    assert {"p0", "p1"}.issubset(result.candidate_id)


def test_road_part_and_complete_link_bounds_prevent_chaining():
    frame = points()
    frame["channel_key"] = "same"
    frame.loc[1, "road_id"] = "other"
    frame.loc[2, "road_part_index"] = 1
    result = s.anchored_groups(s.population(frame.iloc[[0]], frame))
    assert len(result) == 5


def test_shuffle_determinism_and_soft_diversity_keeps_every_candidate():
    frame = points()
    a = s.order(frame, penalty=0.1)
    b = s.order(frame.sample(frac=1, random_state=42), penalty=0.1)
    assert a.candidate_id.tolist() == b.candidate_id.tolist()
    assert set(a.candidate_id) == set(frame.candidate_id)
    assert a.candidate_id.iloc[0] == "p0"
    assert a.candidate_id.iloc[1] == "p4"
    # A nearby distinct pipe is deferred, never deleted/relocated.
    assert (
        a.set_index("candidate_id").geometry.to_dict()
        == frame.set_index("candidate_id").geometry.to_dict()
    )


def test_baseline_first_is_preservation_control_not_a_new_detection():
    frame = s.population(points().iloc[:2], points())
    assert s.order(frame, baseline_first=True).candidate_id.iloc[:2].tolist() == ["p0", "p1"]


def test_unsupported_crs_and_invalid_scores_rejected():
    with pytest.raises(ValueError):
        s.population(points().to_crs(4326), points())
    frame = points()
    frame.loc[0, "score_RF"] = np.nan
    with pytest.raises(ValueError):
        s.order(frame)


def supervised():
    rows = []
    for target in (0, 1):
        for i in range(16):
            row = {column: float(i + target * 3 + 1) for column in site_ranker.FEATURES}
            row.update(
                site_key=f"site-{target}-{i}",
                target=target,
                label_source="field_verified_negative"
                if target == 0
                else "authoritative_inventory_positive_20m",
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_site_model_excludes_metadata_coordinates_and_exposed_scores():
    frame = supervised()
    frame["nearest_inventory_m"] = frame.target
    frame["score_RF"] = frame.target
    frame["x_m"] = frame.target
    model = site_ranker.train(frame)
    assert set(model["features"]) == set(site_ranker.FEATURES)
    first = site_ranker.score(model, frame)
    frame["score_RF"] = 999
    frame["nearest_inventory_m"] = -1
    np.testing.assert_allclose(site_ranker.score(model, frame), first)


def test_unvisited_negatives_and_duplicate_site_labels_cannot_train():
    frame = supervised()
    frame.loc[0, "label_source"] = "unvisited"
    with pytest.raises(ValueError):
        site_ranker.train(frame)
    frame = supervised()
    frame.loc[0, "site_key"] = frame.loc[1, "site_key"]
    with pytest.raises(ValueError):
        site_ranker.train(frame)


def test_label_source_missingness_feature_is_excluded_not_used_as_signal():
    frame = supervised()
    frame.loc[frame.target.eq(0), "route_elevation_dip_60m"] = np.nan
    bundle = site_ranker.train(frame)
    assert "route_elevation_dip_60m" not in bundle["features"]


def test_deterministic_fixed_site_model():
    frame = supervised()
    a, b = site_ranker.train(frame), site_ranker.train(frame)
    np.testing.assert_array_equal(site_ranker.score(a, frame), site_ranker.score(b, frame))


def test_spatial_purge_excludes_siblings_and_far_examples_on_whole_corridor():
    sites = gpd.GeoDataFrame(
        dict(
            site_key=["near", "sibling", "far_same_road", "safe"],
            geometry=[
                Point(500010, 4600000),
                Point(502000, 4600000),
                Point(505000, 4600000),
                Point(506000, 4600000),
            ],
        ),
        crs=32618,
    )
    siblings = sites.copy()
    siblings["route"] = ["other", "other", "heldout", "safe-road"]
    siblings.loc[1, "geometry"] = Point(500100, 4600000)
    road = gpd.GeoDataFrame(
        dict(
            heldout_corridor=["heldout"],
            geometry=[LineString([(500000, 4599900), (500000, 4600100)])],
        ),
        crs=32618,
    )
    keep, distance, purged = site_ranker.training_exclusion(sites, siblings, road)
    assert keep.tolist() == [False, False, False, True]
    assert distance[-1] == 6000
    assert purged == {"near", "sibling", "far_same_road"}
    with pytest.raises(ValueError):
        site_ranker.training_exclusion(sites, siblings, road, purge_m=10)
