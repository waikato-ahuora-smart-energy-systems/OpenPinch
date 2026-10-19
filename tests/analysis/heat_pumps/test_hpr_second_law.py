"""Entropy-generation oracle for Carnot heat pump targets (audit C2).

Entropy generation is sum(Q/T) delivered by the condensers minus sum(Q/T)
absorbed by the evaporators, heat-engine and recovery duty included. Work
carries no entropy, so a target with negative entropy generation moves heat
uphill for free.
"""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch import PinchProblem

SAMPLE_CASES = "OpenPinch/tutorials/sample_cases"
SMALL_BUDGET = dict(maximum_restarts=1, maximum_iterations=20, maximum_evaluations=50)


def _duty(values) -> np.ndarray | float:
    # Parallel results report no separate heat-engine duty.
    return 0.0 if values is None else np.atleast_1d(values)


def _entropy_generation(target) -> tuple[float, float]:
    """Return (entropy generation, entropy flow scale) in kW/K."""
    d = target.hpr_details
    T_cond = np.atleast_1d(d.T_cond) + 273.15
    T_evap = np.atleast_1d(d.T_evap) + 273.15
    S_out = (np.atleast_1d(d.Q_cond) + _duty(d.Q_cond_he)) / T_cond
    S_in = (np.atleast_1d(d.Q_evap) + _duty(d.Q_evap_he)) / T_evap
    return float(S_out.sum() - S_in.sum()), float(S_out.sum() + S_in.sum())


def _assert_second_law(target) -> None:
    s_gen, scale = _entropy_generation(target)
    # Relative to the entropy flows, so the check is independent of duty scale.
    assert s_gen >= -1e-6 * scale


def test_default_cascade_carnot_on_basic_pinch_generates_entropy():
    # With this budget the pooled allocation delivered 450 kW at 204 and
    # 172 C from 121 C for 0.0005 kW of work (S_gen -0.117 kW/K).
    target = PinchProblem(
        f"{SAMPLE_CASES}/basic_pinch.json", project_name="C2"
    ).target.carnot_heat_pump(
        load_fraction=0.5,
        maximum_restarts=1,
        maximum_iterations=100,
        maximum_evaluations=300,
    )

    assert target.hpr_success
    _assert_second_law(target)
    assert target.hpr_work > 0.0


@pytest.mark.performance
def test_default_cascade_carnot_on_pulp_mill_generates_entropy():
    # The audit's default call gave S_gen -43.7 kW/K (58,414 kW lifted for
    # 0.23 kW). Three restarts are the smallest budget that still found that
    # exploit (S_gen -36.7 kW/K), so the run stays within the lane's timeout.
    target = PinchProblem(
        f"{SAMPLE_CASES}/pulp_mill.json", project_name="C2"
    ).target.carnot_heat_pump(load_fraction=0.5, maximum_restarts=3)

    assert target.hpr_success
    _assert_second_law(target)


def test_single_stage_cascade_carnot_is_unchanged():
    # One condenser and one evaporator cannot pool across pairs, so the
    # notebook 08 result is the same before and after the pairwise fix.
    target = PinchProblem(
        f"{SAMPLE_CASES}/basic_pinch.json", project_name="C2"
    ).target.carnot_heat_pump(
        load_fraction=0.25, condensers=1, evaporators=1, **SMALL_BUDGET
    )

    assert target.hpr_work == pytest.approx(53.713002507, rel=1e-6)
    assert target.hpr_cop == pytest.approx(3.42294389, rel=1e-6)
    assert target.hot_utility_target == pytest.approx(566.143407188, rel=1e-6)
    _assert_second_law(target)


def test_parallel_carnot_generates_entropy():
    # Parallel stages pair one condenser with one evaporator already, so the
    # pairwise fix leaves their cycle maths as it was (test_parallel_carnot.py
    # pins it). The budgeted search lands on slightly different designs on
    # different platforms (35.94 kW here, 36.57 kW on CI's Linux), so this
    # end-to-end run checks the second law rather than pinning the work.
    target = PinchProblem(
        f"{SAMPLE_CASES}/basic_pinch.json", project_name="C2"
    ).target.carnot_heat_pump(
        load_fraction=0.25,
        condensers=2,
        evaporators=2,
        is_cascade_cycle=False,
        **SMALL_BUDGET,
    )

    assert target.hpr_success
    _assert_second_law(target)
