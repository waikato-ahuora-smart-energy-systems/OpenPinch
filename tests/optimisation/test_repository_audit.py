"""Normalisation invariants found during repository audit."""

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from OpenPinch.optimisation.candidates import _cluster_candidates


@example(width=1e-8)
@settings(max_examples=30, deadline=None)
@given(
    width=st.floats(min_value=1e-8, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_fixed_coordinate_does_not_change_candidate_clusters(width):
    xs = np.asarray([[0.0], [width]])
    objectives = np.asarray([1.0, 2.0])
    expected = _cluster_candidates(
        xs, objectives, np.asarray([0.0]), np.asarray([width]), 0.01
    )
    actual = _cluster_candidates(
        np.column_stack((np.zeros(2), xs)),
        objectives,
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, width]),
        0.01,
    )
    assert expected == [0, 1]
    assert actual == expected
