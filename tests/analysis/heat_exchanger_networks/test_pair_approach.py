"""The minimum approach temperature of one match, from its two contributions.

Pinch design holds a match to the sum of its source and sink contributions,
each already scaled by the task's dt_cont multiplier. The thermal derivative
and evolution steps hold a share of that sum. Either way a match is never
held to less than 0.1 K: that floor is on the pair, so a stream with a zero
contribution adds nothing to a partner that already supplies an approach.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    MINIMUM_PAIR_APPROACH,
    pair_minimum_approach,
)


@pytest.mark.parametrize(
    ("source", "sink", "expected"),
    [
        (5.0, 5.0, 10.0),
        (5.0, 1.0, 6.0),  # a process stream on a utility
        (5.0, 0.0, 5.0),  # a zero contribution adds nothing
        (0.03, 0.02, 0.1),  # a sum below the floor
        (0.0, 0.0, 0.1),
    ],
)
def test_pinch_design_holds_the_sum_of_the_contributions(source, sink, expected):
    assert pair_minimum_approach(source, sink) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("source", "sink", "expected"),
    [
        (5.0, 5.0, 2.5),
        (5.0, 1.0, 1.5),
        (5.0, 0.0, 1.25),
        (0.03, 0.02, 0.1),
        (0.0, 0.0, 0.1),
    ],
)
def test_the_later_steps_hold_a_share_of_the_sum(source, sink, expected):
    assert pair_minimum_approach(source, sink, share=0.25) == pytest.approx(
        expected, abs=1e-12
    )


def test_the_multiplier_scales_the_contributions_and_not_the_floor():
    multiplier = 0.75
    assert pair_minimum_approach(5.0 * multiplier, 0.0 * multiplier) == pytest.approx(
        3.75, abs=1e-12
    )
    assert pair_minimum_approach(0.0 * multiplier, 0.0 * multiplier) == pytest.approx(
        MINIMUM_PAIR_APPROACH, abs=1e-12
    )


def test_the_floor_is_a_tenth_of_a_kelvin():
    assert MINIMUM_PAIR_APPROACH == pytest.approx(0.1, abs=1e-12)
