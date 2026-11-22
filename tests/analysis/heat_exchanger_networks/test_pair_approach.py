"""The minimum approach temperature of one match, from its two contributions.

Pinch design holds a match to the sum of its source and sink contributions,
each already scaled by the task's dt_cont multiplier. The thermal derivative
and evolution steps hold a share of that sum. Either way a match is never
held to less than 0.1 K: that floor is on the pair, so a stream with a zero
contribution adds nothing to a partner that already supplies an approach.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base.approach import (
    _cold_utility_inlet_approach_temperature,
    _cold_utility_outlet_approach_temperature,
    _hot_utility_inlet_approach_temperature,
    _hot_utility_outlet_approach_temperature,
    _set_minimum_approach_temperatures,
)
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


def _model(*, hot, cold, hot_utilities, cold_utilities):
    """The contribution rows a model reads, one period, already scaled."""
    return SimpleNamespace(
        N_periods=1,
        dTmin=10.0,
        T_h_cont_period=np.array([hot], dtype=float),
        T_c_cont_period=np.array([cold], dtype=float),
        T_hu_cont_period=np.array([hot_utilities], dtype=float),
        T_cu_cont_period=np.array([cold_utilities], dtype=float),
    )


def test_a_models_approach_grids_floor_each_pair_and_not_each_stream():
    model = _model(
        hot=[5.0, 0.0], cold=[5.0, 0.0], hot_utilities=[1.0, 0.0], cold_utilities=[0.0]
    )

    _set_minimum_approach_temperatures(model)

    # [hot][cold]: only the pair of two zero contributions meets the floor.
    np.testing.assert_allclose(
        model.dT_r_period[0], [[10.0, 5.0], [5.0, 0.1]], rtol=0, atol=1e-12
    )
    # Heaters, [utility][cold].
    np.testing.assert_allclose(
        model.dT_hu_period[0], [[6.0, 1.0], [5.0, 0.1]], rtol=0, atol=1e-12
    )
    # Coolers, [hot][utility].
    np.testing.assert_allclose(
        model.dT_cu_period[0], [[5.0], [0.1]], rtol=0, atol=1e-12
    )
    assert model.dT_hu_min_period[0][1][1] == pytest.approx(0.1, abs=1e-12)
    assert model.dT_cu_min_period[0][1][0] == pytest.approx(0.1, abs=1e-12)


def test_a_heaters_and_a_coolers_end_approaches_floor_the_pair():
    model = _model(
        hot=[5.0, 0.0], cold=[5.0, 0.0], hot_utilities=[1.0, 0.0], cold_utilities=[0.0]
    )

    assert _hot_utility_inlet_approach_temperature(model, 0, 1, 0) == pytest.approx(
        5.0, abs=1e-12
    )
    assert _hot_utility_inlet_approach_temperature(model, 1, 1, 0) == pytest.approx(
        0.1, abs=1e-12
    )
    assert _hot_utility_outlet_approach_temperature(model, 1, 1, 0) == pytest.approx(
        0.1, abs=1e-12
    )
    assert _cold_utility_inlet_approach_temperature(model, 0, 0, 0) == pytest.approx(
        5.0, abs=1e-12
    )
    assert _cold_utility_inlet_approach_temperature(model, 1, 0, 0) == pytest.approx(
        0.1, abs=1e-12
    )
    assert _cold_utility_outlet_approach_temperature(model, 1, 0, 0) == pytest.approx(
        0.1, abs=1e-12
    )
