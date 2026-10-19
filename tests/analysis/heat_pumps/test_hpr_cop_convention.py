"""``hpr_cop`` follows the duty (audit M1).

Heat pumps report the heating COP (Q_cond/W) and refrigeration reports the
cooling COP (Q_evap/W), whichever cycle model produced the target.
"""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch import PinchProblem

SAMPLE_CASES = "OpenPinch/tutorials/sample_cases"
SMALL_BUDGET = dict(maximum_restarts=1, maximum_iterations=20, maximum_evaluations=50)


@pytest.mark.parametrize("is_cascade_cycle", [True, False])
def test_carnot_refrigeration_reports_the_cooling_cop(is_cascade_cycle):
    # Before the fix this reported Q_cond/W (2.667 on chocolate_factory at a
    # 0.25 load), while vapour-compression refrigeration reported Q_evap/W.
    target = PinchProblem(
        f"{SAMPLE_CASES}/chocolate_factory.json", project_name="M1"
    ).target.carnot_refrigeration(
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        is_cascade_cycle=is_cascade_cycle,
        **SMALL_BUDGET,
    )

    cooling = float(np.sum(target.hpr_details.Q_evap))
    assert target.hpr_cop == pytest.approx(cooling / target.hpr_work, rel=1e-9)


@pytest.mark.parametrize("is_cascade_cycle", [True, False])
def test_carnot_heat_pump_reports_the_heating_cop(is_cascade_cycle):
    target = PinchProblem(
        f"{SAMPLE_CASES}/basic_pinch.json", project_name="M1"
    ).target.carnot_heat_pump(
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        is_cascade_cycle=is_cascade_cycle,
        **SMALL_BUDGET,
    )

    heating = float(np.sum(target.hpr_details.Q_cond))
    assert target.hpr_cop == pytest.approx(heating / target.hpr_work, rel=1e-9)
