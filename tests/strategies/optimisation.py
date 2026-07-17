"""Domain-specific Hypothesis strategies for scalar optimisation."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st


@st.composite
def finite_candidate_clouds(draw):
    """Generate finite two-dimensional candidates and scalar objectives."""
    count = draw(st.integers(min_value=1, max_value=12))
    coordinates = st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    objectives = st.floats(
        min_value=-1_000.0,
        max_value=1_000.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    points = draw(st.lists(st.tuples(coordinates, coordinates), min_size=count, max_size=count))
    values = draw(st.lists(objectives, min_size=count, max_size=count))
    return np.asarray(points, dtype=float), np.asarray(values, dtype=float)


__all__ = ["finite_candidate_clouds"]
