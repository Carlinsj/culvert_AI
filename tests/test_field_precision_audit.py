"""Distinct-site matching and threshold tie handling protect precision claims."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


spec = importlib.util.spec_from_file_location(
    "field_precision_audit", Path(__file__).resolve().parents[1] / "scripts/audit_field_precision.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_prefix_matching_reassigns_instead_of_double_counting():
    assert audit.prefix_distinct_counts([[0, 1], [0], [0], [], [2]]).tolist() == [1, 2, 2, 2, 3]
    rng = np.random.default_rng(42)
    for _ in range(30):
        matrix = (rng.random((18, 9)) < .18).astype(np.int8)
        neighbors = [np.flatnonzero(row).tolist() for row in matrix]
        actual = audit.prefix_distinct_counts(neighbors)
        for k in range(1, 19):
            expected = np.sum(maximum_bipartite_matching(csr_matrix(matrix[:k]), perm_type="column") >= 0)
            assert actual[k - 1] == expected


def test_scalar_threshold_cannot_cherry_pick_inside_a_score_tie():
    sites = pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "score": [1., 1., .5],
        "x_m": [0., 100., 200.], "y_m": [0., 0., 0.],
    })
    curve = audit.threshold_curve(sites, "score", np.asarray([[0., 0.]]))
    assert curve["recommendations"].tolist() == [2, 3]
    assert curve["precision"].tolist() == [0.5, 1 / 3]
    assert audit.summary_row(curve, .7, "score")["maximum_k"] == 0
