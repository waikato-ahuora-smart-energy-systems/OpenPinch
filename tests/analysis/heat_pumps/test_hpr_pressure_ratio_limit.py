"""A compressor's pressure ratio limit is a cycle penalty, not a decode clamp."""

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
from OpenPinch.contracts.hpr import HPRTargetingError
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS

from .helpers import _base_args


def _ratio(fluid: str, t_cond: float, t_evap: float) -> float:
    p_hi = CoolProp.PropsSI("P", "T", t_cond + 273.15, "Q", 0, fluid)
    p_lo = CoolProp.PropsSI("P", "T", t_evap + 273.15, "Q", 1, fluid)
    return p_hi / p_lo


def test_the_default_limit_is_15():
    assert CONFIG_FIELD_SPECS["HPR_MAX_PRESSURE_RATIO"].default == 15.0


def _high_lift_args():
    args = _base_args(n_cond=1, n_evap=1)
    args.T_cold = np.array([205.0, 150.0, 100.0])
    args.T_hot = np.array([100.0, 60.0, 20.0])
    args.refrigerant_ls = ["water"]
    args.max_pressure_ratio = 15.0
    return args


def test_decoding_a_single_stage_design_leaves_the_limit_to_the_cycle_penalty():
    args = _high_lift_args()
    _, bnds = hp_cascade._get_cascade_hp_opt_setup(None, args)

    # All zeros decode to the hottest condenser and the coldest evaporator.
    state = hp_cascade._parse_cascade_hp_state_variables(np.zeros(len(bnds)), args)

    assert _ratio("water", state.T_cond[0], state.T_evap[0]) > 15.0


def test_decoding_a_parallel_design_leaves_the_limit_to_the_cycle_penalty():
    args = _high_lift_args()
    x = np.zeros(hp_parallel._parallel_layout(args).size)

    state = hp_parallel._parse_parallel_hp_state_temperatures(x, args)

    assert _ratio("water", state.T_cond[0], state.T_evap[0]) > 15.0


def test_decoding_an_mvr_design_leaves_the_limit_to_the_cycle_penalty():
    args = _high_lift_args()
    x = np.zeros(hp_mvr._vc_mvr_layout(args).size)

    state = hp_mvr._parse_vc_mvr_state_variables(x, args)

    T_cond_vc = state.T_cond[-1]
    assert _ratio("water", T_cond_vc, state.T_evap[-1]) > 15.0


def _ahmad_utility_refrigeration(*, limit: float | None = None):
    path = Path(__file__).parents[3] / "examples/stream_data/p_Ahmad (example 3).json"
    problem = PinchProblem(json.loads(path.read_text()))
    problem.target.direct_heat_integration()
    # Water evaporates far below its 0.5 bar suction floor here, so the floor
    # is off: these tests isolate the pressure ratio limit.
    options = {"HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.0}
    if limit is not None:
        options["HPR_MAX_PRESSURE_RATIO"] = limit
    return problem.target.vapour_compression_refrigeration(
        load_fraction=0.25,
        condensers=2,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=3,
        # The penalty decides the default limit's outcome at 28, but not at 24
        # or 32: its test checks the reason, so a change here must keep it.
        maximum_evaluations=28,
        refrigerants=["Water"],
        is_utility_refrigeration=True,
        is_cascade_cycle=False,
        options=options,
    )


def test_a_refrigerator_that_cannot_reach_its_duty_within_the_limit_is_a_failure():
    # Whether this ends on a penalised or a zero-duty design depends on the
    # platform's numerics. The test below shows the limit is the cause.
    with pytest.raises(HPRTargetingError):
        _ahmad_utility_refrigeration(limit=1.5)


def test_the_same_refrigerator_is_a_target_without_the_limit():
    target = _ahmad_utility_refrigeration(limit=1e6)

    assert target.hpr_details.obj > 0.0


# The Carnot warm start knows nothing about pressure ratios. It is seeded
# within the limit, as decoding used to clamp it, so a small search still
# starts from a design the penalty does not reject.
_HIGH_LIFT_SEED = SimpleNamespace(
    # Within the limit, water condensing at 190 C evaporates at about 95 C,
    # inside the 100 C hottest source of _high_lift_args.
    T_cond=np.array([190.0]),
    T_evap=np.array([20.0]),
    Q_cond=np.array([100.0]),
    Q_evap=np.array([60.0]),
    Q_amb_hot=0.0,
    Q_amb_cold=0.0,
)


def test_a_single_stage_warm_start_is_seeded_within_the_limit():
    args = _high_lift_args()
    x0, _ = hp_cascade._get_cascade_hp_opt_setup(_HIGH_LIFT_SEED, args)

    state = hp_cascade._parse_cascade_hp_state_variables(x0, args)

    assert _ratio("water", state.T_cond[0], state.T_evap[0]) <= 15.0 * (1 + 1e-6)


def test_a_parallel_warm_start_is_seeded_within_the_limit():
    args = _high_lift_args()
    x0, _ = hp_parallel._get_parallel_hp_opt_setup(_HIGH_LIFT_SEED, args)

    state = hp_parallel._parse_parallel_hp_state_temperatures(x0, args)

    assert _ratio("water", state.T_cond[0], state.T_evap[0]) <= 15.0 * (1 + 1e-6)


def test_an_mvr_warm_start_seeds_the_closed_stage_within_the_limit():
    args = _high_lift_args()
    x0, _ = hp_mvr._get_vc_mvr_opt_setup(_HIGH_LIFT_SEED, args)

    state = hp_mvr._parse_vc_mvr_state_variables(x0, args)

    assert _ratio("water", state.T_cond[-1], state.T_evap[-1]) <= 15.0 * (1 + 1e-6)
