"""Checks for fold-label isolation and deterministic physical-site inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location(
    "test_challenger", Path(__file__).resolve().parents[1] / "scripts/train_site_challenger.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_training_excludes_unknowns_and_outer_labels_and_spatial_duplicates():
    pool = pd.DataFrame({"candidate_id": ["positive", "unknown", "junction", "outer"],
                         "route": ["train", "train", "train", "heldout"],
                         "x_m": [0., 100., 200., 210.], "y_m": [0.] * 4})
    real = pd.DataFrame({"route": ["train", "train", "heldout"], "x": [0., 200., 210.],
                         "y": [0.] * 3, "real_key": ["p1", "p2", "outer-label"]})
    negatives = pd.DataFrame({"candidate_id": ["verified"], "route": ["train"],
                              "x_m": [-100.], "y_m": [0.]})
    first = RUNNER.training_rows(pool, negatives, real, {"train"}, np.array([[210., 0.]]))
    real.loc[real.route.eq("heldout"), "x"] = -9999.
    second = RUNNER.training_rows(pool, negatives, real, {"train"}, np.array([[210., 0.]]))
    pd.testing.assert_frame_equal(first, second)
    assert set(first.candidate_id) == {"positive", "verified"}
    assert first.loc[first.target.eq(0), "label_source"].eq("field_verified_negative").all()


def test_arbitrary_road_names_preserve_normalization_and_sites():
    frame = pd.DataFrame({"candidate_id": [f"c{i}" for i in range(12)],
                          "route": ["road1"] * 8 + ["road2"] * 4,
                          "raw_score": np.arange(12.) / 12.,
                          "x_m": np.arange(12.) * 15., "y_m": [0.] * 12})
    renamed = frame.copy()
    renamed["route"] = renamed.route.map({"road1": "never-seen-X", "road2": "new-Y"})
    for policy in RUNNER.POLICIES:
        np.testing.assert_array_equal(RUNNER.normalize(frame, policy, (.4, .2)),
                                      RUNNER.normalize(renamed, policy, (.4, .2)))
    frame["priority"] = frame.raw_score
    sites = RUNNER.saved_priority_sites(frame, "priority")
    assert sites.candidate_id.iloc[0] == "c11"
    x = sites.x_m.to_numpy()
    assert (np.abs(np.subtract.outer(x, x)) + np.eye(len(sites)) * 100).min() > 20.
