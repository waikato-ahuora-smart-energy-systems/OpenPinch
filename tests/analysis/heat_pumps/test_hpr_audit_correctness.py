"""Regression tests for the HPR correctness fixes (audit area 3a)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_pumps.common.layout import clip_to_bounds
from OpenPinch.analysis.heat_pumps.common.shared import (
    cap_stage_condensing_temperatures,
    floor_stage_evaporating_temperatures,
    is_negligible_useful_duty,
)
from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)

# 3.2 Subcritical caps per stage


def test_coolprop_stages_may_condense_above_their_critical_temperature():
    # Above its critical point a CoolProp stage runs transcritical.
    args = SimpleNamespace(
        refrigerant_ls=["water", "R134a"], simulation_backend="coolprop"
    )

    capped = cap_stage_condensing_temperatures(np.array([150.0, 140.0]), args)

    np.testing.assert_array_equal(capped, [150.0, 140.0])


def test_each_tespy_stage_is_capped_below_its_own_critical_temperature():
    pytest.importorskip("CoolProp")
    args = SimpleNamespace(
        refrigerant_ls=["water", "R134a"], simulation_backend="tespy"
    )

    capped = cap_stage_condensing_temperatures(np.array([150.0, 140.0]), args)

    # Water (Tcrit 374 degC) is untouched; R134a (Tcrit 101 degC) is capped
    # 2 K below its critical point.
    assert capped[0] == pytest.approx(150.0)
    assert capped[1] == pytest.approx(99.06, abs=0.1)


def test_capping_keeps_each_temperature_with_its_own_refrigerant():
    pytest.importorskip("CoolProp")
    # Unsorted refrigerants: R134a (Tcrit 101 degC) above water.
    args = SimpleNamespace(
        refrigerant_ls=["R134a", "water"], simulation_backend="tespy"
    )

    capped = cap_stage_condensing_temperatures(np.array([200.0, 150.0]), args)

    # Re-sorting would hand 150 degC to R134a; instead R134a's stage is capped
    # and the colder water stage follows it down to stay in descending order.
    assert capped[0] == pytest.approx(99.06, abs=0.1)
    assert capped[1] == pytest.approx(capped[0])


def test_tespy_stages_are_capped_like_coolprop():
    args = SimpleNamespace(refrigerant_ls=["R134a"], simulation_backend="tespy")

    capped = cap_stage_condensing_temperatures(np.array([140.0]), args)

    assert capped[0] == pytest.approx(99.06, abs=0.1)


# 3.3 Near-zero duty is no heat pump


@pytest.mark.parametrize("duty, negligible", [(0.0, True), (0.9, True), (1.1, False)])
def test_useful_duty_below_a_tenth_of_a_percent_is_no_heat_pump(duty, negligible):
    args = SimpleNamespace(Q_hpr_target=1000.0)

    assert is_negligible_useful_duty(duty, args) is negligible


# 3.4 Cascade sign and seed clipping


@pytest.mark.parametrize("t_cond, penalty", [(60.0, 3.0), (63.0, 0.0), (70.0, 0.0)])
def test_cascade_condensers_must_sit_dt_cascade_hx_above_evaporators(t_cond, penalty):
    cycle = SimpleNamespace(_dt_cascade_hx=5.0)

    violation = CascadeVapourCompressionCycle._validate_T_cond_and_evap(
        cycle, np.array([t_cond]), np.array([58.0])
    )

    assert violation == pytest.approx(penalty)


@pytest.mark.parametrize(
    "t_cond, t_evap, penalty",
    [
        ([90.0, 80.0, 70.0], [40.0, 30.0], 0.0),
        # Only the first pair is out of order; the old telescoping sum missed it.
        ([50.0, 90.0, 80.0], [40.0, 30.0], 40.0),
        ([90.0, 80.0, 70.0], [20.0, 30.0, 25.0], 10.0),
    ],
)
def test_cascade_stage_order_checks_every_adjacent_pair(t_cond, t_evap, penalty):
    cycle = SimpleNamespace(_dt_cascade_hx=5.0)

    violation = CascadeVapourCompressionCycle._validate_T_cond_and_evap(
        cycle, np.array(t_cond), np.array(t_evap)
    )

    assert violation == pytest.approx(penalty)


def test_encoded_seed_is_clipped_into_its_bounds():
    seed = clip_to_bounds([-1e-17, 1.0 + 1e-16, 0.5], [(0.0, 1.0)] * 3)

    np.testing.assert_array_equal(seed, [0.0, 1.0, 0.5])


# Evaporating temperature limit


def test_cycle_rejects_evaporation_below_the_fluid_minimum_temperature():
    pytest.importorskip("CoolProp")
    cycle = VapourCompressionCycle()

    # CoolProp extrapolates water saturation below its triple point (0.01 degC)
    # without error, so the cycle has to refuse the state itself.
    with pytest.raises(ValueError, match="minimum temperature"):
        cycle.solve(
            T_evap=-0.1, T_cond=85.0, dtcont=5.0, eta_comp=0.7, refrigerant="water"
        )


def test_each_evaporator_is_floored_at_its_own_refrigerant_minimum():
    pytest.importorskip("CoolProp")
    args = SimpleNamespace(
        refrigerant_ls=["water", "ammonia"], simulation_backend="coolprop"
    )

    floored = floor_stage_evaporating_temperatures(np.array([-5.0, -20.0]), args)

    # Water stops at its triple point (0.01 degC); ammonia (Tmin -77.7 degC) does not.
    assert floored[0] == pytest.approx(0.01, abs=1e-5)
    assert floored[1] == pytest.approx(-20.0)
