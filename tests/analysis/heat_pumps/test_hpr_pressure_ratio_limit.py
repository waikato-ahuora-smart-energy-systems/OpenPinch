"""A compressor's pressure ratio is limited when a design is decoded."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import CoolProp.CoolProp as CoolProp
import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
import OpenPinch.analysis.heat_pumps.targeting.parallel_vapour_compression as hp_parallel
import OpenPinch.analysis.heat_pumps.targeting.vapour_compression_mvr as hp_mvr
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.common.shared import (
    limit_cascade_pressure_ratios,
    limit_parallel_pressure_ratios,
)
from OpenPinch.contracts.hpr import INFEASIBLE_REFRIGERATION_REASON, HPRTargetingError
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS

from .helpers import _base_args


def _ratio(fluid: str, t_cond: float, t_evap: float) -> float:
    p_hi = CoolProp.PropsSI("P", "T", t_cond + 273.15, "Q", 0, fluid)
    p_lo = CoolProp.PropsSI("P", "T", t_evap + 273.15, "Q", 1, fluid)
    return p_hi / p_lo


def _args(*fluids: str, limit: float | None = 15.0, dt: float = 0.0):
    return SimpleNamespace(
        refrigerant_ls=list(fluids),
        simulation_backend="coolprop",
        max_pressure_ratio=limit,
        dt_cascade_hx=dt,
    )


def test_the_default_limit_is_15():
    assert CONFIG_FIELD_SPECS["HPR_MAX_PRESSURE_RATIO"].default == 15.0


def test_a_machine_over_the_limit_evaporates_warmer():
    T_evap = limit_parallel_pressure_ratios(
        np.array([205.0]), np.array([20.0]), _args("water")
    )

    assert _ratio("water", 205.0, T_evap[0]) == pytest.approx(15.0, rel=1e-6)


def test_a_machine_within_the_limit_is_unchanged():
    T_evap = limit_parallel_pressure_ratios(
        np.array([120.0]), np.array([90.0]), _args("water")
    )

    assert T_evap[0] == 90.0


def test_parallel_machines_use_their_own_fluid_and_stay_in_order():
    T_cond = np.array([205.0, 100.0])
    T_evap = limit_parallel_pressure_ratios(
        T_cond, np.array([60.0, 20.0]), _args("water", "ammonia")
    )

    assert _ratio("water", T_cond[0], T_evap[0]) <= 15.0 * (1 + 1e-6)
    assert _ratio("ammonia", T_cond[1], T_evap[1]) <= 15.0 * (1 + 1e-6)
    assert T_evap[0] >= T_evap[1]


def test_no_limit_leaves_the_temperatures_alone():
    T_evap = limit_parallel_pressure_ratios(
        np.array([205.0]), np.array([20.0]), _args("water", limit=None)
    )

    assert T_evap[0] == 20.0


def test_a_tespy_target_is_not_limited():
    args = _args("water")
    args.simulation_backend = "tespy"

    T_evap = limit_parallel_pressure_ratios(np.array([205.0]), np.array([20.0]), args)

    assert T_evap[0] == 20.0


def test_a_mixture_is_limited_between_its_bubble_and_dew_pressures():
    fluid = "R134a[0.5]&R245fa[0.5]"

    T_evap = limit_parallel_pressure_ratios(
        np.array([80.0]), np.array([-40.0]), _args(fluid)
    )

    state = CoolProp.AbstractState("HEOS", "R134a&R245fa")
    state.set_mole_fractions([0.5, 0.5])
    state.update(CoolProp.QT_INPUTS, 0.0, 80.0 + 273.15)
    p_hi = state.p()
    state.update(CoolProp.QT_INPUTS, 1.0, float(T_evap[0]) + 273.15)
    assert p_hi / state.p() == pytest.approx(15.0, rel=1e-4)


def test_each_cascade_stage_is_limited_from_the_top_down():
    args = _args("water", "ammonia", dt=5.0)

    T_cond, T_evap = limit_cascade_pressure_ratios(
        np.array([205.0, 60.0]), np.array([-30.0]), args
    )

    # The upper stage evaporates dt below the lower stage's condenser.
    assert T_cond[0] == 205.0
    assert _ratio("water", T_cond[0], T_cond[1] - 5.0) == pytest.approx(15.0, rel=1e-6)
    assert _ratio("ammonia", T_cond[1], T_evap[0]) == pytest.approx(15.0, rel=1e-6)


def test_a_cascade_of_two_evaporators_raises_the_lower_evaporator():
    args = _args("water", "water", dt=0.0)

    _, T_evap = limit_cascade_pressure_ratios(
        np.array([205.0]), np.array([150.0, 20.0]), args
    )

    assert T_evap[0] == 150.0
    assert _ratio("water", 150.0, T_evap[1]) == pytest.approx(15.0, rel=1e-6)


def _high_lift_args():
    args = _base_args(n_cond=1, n_evap=1)
    args.T_cold = np.array([205.0, 150.0, 100.0])
    args.T_hot = np.array([100.0, 60.0, 20.0])
    args.refrigerant_ls = ["water"]
    args.max_pressure_ratio = 15.0
    return args


def test_decoding_a_single_stage_design_applies_the_limit():
    args = _high_lift_args()
    _, bnds = hp_cascade._get_cascade_hp_opt_setup(None, args)

    # All zeros decode to the hottest condenser and the coldest evaporator.
    state = hp_cascade._parse_cascade_hp_state_variables(np.zeros(len(bnds)), args)

    assert _ratio("water", state.T_cond[0], state.T_evap[0]) <= 15.0 * (1 + 1e-6)


def test_decoding_a_parallel_design_applies_the_limit():
    args = _high_lift_args()
    x = np.zeros(hp_parallel._parallel_layout(args).size)

    state = hp_parallel._parse_parallel_hp_state_temperatures(x, args)

    assert _ratio("water", state.T_cond[0], state.T_evap[0]) <= 15.0 * (1 + 1e-6)


def test_decoding_an_mvr_design_limits_the_closed_stage():
    args = _high_lift_args()
    x = np.zeros(hp_mvr._vc_mvr_layout(args).size)

    state = hp_mvr._parse_vc_mvr_state_variables(x, args)

    T_cond_vc = state.T_cond[-1]
    assert _ratio("water", T_cond_vc, state.T_evap[-1]) <= 15.0 * (1 + 1e-6)


def _ahmad_utility_refrigeration(*, limit: float | None = None):
    path = Path(__file__).parents[3] / "examples/stream_data/p_Ahmad (example 3).json"
    problem = PinchProblem(json.loads(path.read_text()))
    problem.target.direct_heat_integration()
    options = {} if limit is None else {"HPR_MAX_PRESSURE_RATIO": limit}
    return problem.target.vapour_compression_refrigeration(
        load_fraction=0.25,
        condensers=2,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=3,
        # The penalty decides these outcomes at 28, but not at 24 or 32: the
        # failure tests check their reason, so a change here must keep it.
        maximum_evaluations=28,
        refrigerants=["Water"],
        is_utility_refrigeration=True,
        is_cascade_cycle=False,
        options=options,
    )


def _rejected_for_a_penalty(error: HPRTargetingError) -> bool:
    return any(
        failure.summary == INFEASIBLE_REFRIGERATION_REASON
        for failure in error.diagnostics.representative_failures
    )


def test_a_refrigerator_that_cannot_reach_its_duty_within_the_limit_is_a_failure():
    with pytest.raises(HPRTargetingError) as raised:
        _ahmad_utility_refrigeration(limit=1.5)

    assert _rejected_for_a_penalty(raised.value)


def test_a_refrigerator_still_carrying_a_penalty_is_a_failure():
    # This small search budget ends on penalised designs.
    with pytest.raises(HPRTargetingError) as raised:
        _ahmad_utility_refrigeration()

    assert _rejected_for_a_penalty(raised.value)


def test_the_same_refrigerator_is_a_target_without_the_limit():
    target = _ahmad_utility_refrigeration(limit=1e6)

    assert target.hpr_details.obj > 0.0
