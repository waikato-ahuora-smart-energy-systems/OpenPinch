"""Regression tests for the scaled HPR feasibility penalty (audit area 3b)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_pumps.common.shared import (
    _cycle_penalty,
    hpr_penalty_cost_scale,
)


def _args(**overrides) -> SimpleNamespace:
    values = {
        "Q_hpr_target": 1000.0,
        "rho_penalty": 10.0,
        "eta_penalty": 0.01,
        "is_heat_pumping": True,
        "heat_to_power_ratio": 1.0,
        "refrigeration_to_power_ratio": 0.5,
        "ele_price": 100.0,
        "annual_op_time": 8000.0,
        "hot_utility_capital_cost": 750.0,
        "refrigeration_capital_cost": 1500.0,
        "discount_rate": 0.07,
        "serv_life": 20.0,
    }
    return SimpleNamespace(**(values | overrides))


def test_penalty_is_exact_in_the_relative_violation():
    args = _args()

    tenth = _cycle_penalty(args=args, cycle_penalty_terms=[100.0])
    one_megawatt = _cycle_penalty(args=args, cycle_penalty_terms=[1000.0])
    thirty_megawatts = _cycle_penalty(args=args, cycle_penalty_terms=[30000.0])

    # rho * (r + r^2): the linear part keeps small shortfalls from being cheap.
    assert tenth == pytest.approx(10.0 * (0.1 + 0.01))
    assert one_megawatt == pytest.approx(10.0 * (1.0 + 1.0))
    assert thirty_megawatts == pytest.approx(10.0 * (30.0 + 900.0))


def test_penalty_is_the_same_for_any_problem_size_at_the_same_relative_shortfall():
    small = _cycle_penalty(args=_args(Q_hpr_target=10.0), cycle_penalty_terms=[1.0])
    large = _cycle_penalty(
        args=_args(Q_hpr_target=10000.0), cycle_penalty_terms=[1000.0]
    )

    assert small == pytest.approx(large)


def test_penalty_cost_does_not_vanish_with_a_zero_electricity_price():
    priced = hpr_penalty_cost_scale(_args())
    free_power = hpr_penalty_cost_scale(_args(ele_price=0.0))
    no_hours = hpr_penalty_cost_scale(_args(annual_op_time=0.0))

    assert free_power > 1.0
    assert no_hours > 1.0
    assert priced > free_power


def test_refrigeration_penalty_is_priced_at_default_refrigeration():
    heating = hpr_penalty_cost_scale(_args(ele_price=0.0))
    cooling = hpr_penalty_cost_scale(_args(ele_price=0.0, is_heat_pumping=False))

    assert cooling == pytest.approx(2.0 * heating)  # 1500 vs 750 $/kW capital
