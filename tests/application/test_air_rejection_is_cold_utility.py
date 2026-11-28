"""Heat a heat pump target rejects to ambient air is cooling duty.

Every rejected heat ends at ambient and is charged at the ambient price
(decided 2026-11-27, review finding 20). With free ambient air enabled, the
air took the process cooling the evaporator left: on ``basic_pinch`` the
target reported 0 kW of cold utility with 513 kW of process cooling left,
and its heat recovery above its own limit. Ambient air as a source of heat
stays free.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem

_SHORT = {"maximum_restarts": 0, "maximum_evaluations": 1}


def test_the_cooling_left_after_the_evaporator_is_cold_utility():
    problem = PinchProblem("basic_pinch.json")
    base = problem.target.direct_heat_integration()

    target = problem.target.carnot_heat_pump(
        load_fraction=1.0,
        condensers=1,
        evaporators=1,
        options={"HPR_AMBIENT_AIR_ENABLED": True},
        **_SHORT,
    )

    left = base.cold_utility_target - float(sum(target.hpr_details.Q_evap))
    assert target.cold_utility_target == pytest.approx(left, abs=1e-3)
