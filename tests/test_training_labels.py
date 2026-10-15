import geopandas as gpd
from shapely.geometry import Point

from culvert_ai.features import build_feature_table
from culvert_ai.labels import (
    UNLABELED_CANDIDATE,
    VERIFIED_NEGATIVE,
    VERIFIED_POSITIVE,
    WEAK_NEGATIVE,
)
from culvert_ai.model import select_training_rows


def _candidates(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32618")


def test_confirmed_culvert_is_verified_positive():
    candidates = _candidates(
        [{"candidate_id": "candidate", "source": "route_interval_sample", "geometry": Point(0, 0)}]
    )
    known = _candidates([{"label": "confirmed_field_observation", "geometry": Point(0, 0)}])

    features = build_feature_table(candidates, known_culverts=known).set_index("candidate_id")

    assert features.loc["candidate", "training_label_class"] == VERIFIED_POSITIVE
    assert features.loc["candidate", "is_culvert"] == 1


def test_field_confirmed_no_culvert_is_verified_negative():
    candidates = _candidates(
        [
            {
                "candidate_id": "field-negative",
                "source": "field_observed_non_culvert",
                "field_observation_label": "no_culvert",
                "geometry": Point(0, 0),
            }
        ]
    )
    negatives = _candidates(
        [
            {
                "observation_id": "obs-negative",
                "label": "no_culvert",
                "training_label_class": VERIFIED_NEGATIVE,
                "geometry": Point(0, 0),
            }
        ]
    )

    features = build_feature_table(candidates, negative_culverts=negatives).set_index(
        "candidate_id"
    )

    assert features.loc["field-negative", "training_label_class"] == VERIFIED_NEGATIVE
    assert features.loc["field-negative", "is_culvert"] == 0


def test_generated_candidate_is_weak_or_unlabeled_not_verified_negative():
    candidates = _candidates(
        [
            {"candidate_id": "nearby", "source": "route_interval_sample", "geometry": Point(5, 0)},
            {
                "candidate_id": "ordinary",
                "source": "exact_road_stream_intersection",
                "geometry": Point(100, 0),
            },
        ]
    )
    negatives = _candidates(
        [{"observation_id": "obs-negative", "label": "no_culvert", "geometry": Point(0, 0)}]
    )

    features = build_feature_table(
        candidates,
        negative_culverts=negatives,
        negative_radius_m=10,
    ).set_index("candidate_id")

    assert features.loc["nearby", "training_label_class"] == WEAK_NEGATIVE
    assert features.loc["ordinary", "training_label_class"] == UNLABELED_CANDIDATE


def test_weak_negative_sampling_is_explicit_and_bounded():
    rows = [
        {"candidate_id": "positive", "is_culvert": 1, "training_label_class": VERIFIED_POSITIVE},
        {"candidate_id": "negative", "is_culvert": 0, "training_label_class": VERIFIED_NEGATIVE},
        *[
            {
                "candidate_id": f"weak-{index}",
                "is_culvert": 0,
                "training_label_class": WEAK_NEGATIVE,
            }
            for index in range(4)
        ],
        {"candidate_id": "unlabeled", "is_culvert": 0, "training_label_class": UNLABELED_CANDIDATE},
    ]
    table = _candidates([{**row, "geometry": Point(index, 0)} for index, row in enumerate(rows)])

    verified_only, default_summary = select_training_rows(table)
    sampled, sampled_summary = select_training_rows(
        table, weak_negative_sample_size=2, random_state=7
    )

    assert set(verified_only["candidate_id"]) == {"positive", "negative"}
    assert default_summary["weak_negative_rows_sampled"] == 0
    assert len(sampled) == 4
    assert sampled_summary["weak_negative_rows_sampled"] == 2
    assert "unlabeled" not in set(sampled["candidate_id"])


def test_verified_positive_takes_precedence_over_negative_proximity():
    candidates = _candidates(
        [{"candidate_id": "same-place", "source": "route_interval_sample", "geometry": Point(0, 0)}]
    )
    known = _candidates([{"label": "confirmed_field_observation", "geometry": Point(0, 0)}])
    negatives = _candidates(
        [{"observation_id": "obs-negative", "label": "no_culvert", "geometry": Point(0, 0)}]
    )

    features = build_feature_table(
        candidates,
        known_culverts=known,
        negative_culverts=negatives,
    ).set_index("candidate_id")

    assert features.loc["same-place", "training_label_class"] == VERIFIED_POSITIVE
    assert features.loc["same-place", "is_culvert"] == 1
    assert features.loc["same-place", "field_denied"] == 0
