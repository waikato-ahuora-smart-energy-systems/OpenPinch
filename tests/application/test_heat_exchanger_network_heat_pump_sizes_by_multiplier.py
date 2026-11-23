"""The sizes tried at a multiplier are the load kinks at that multiplier.

Scaling every contribution moves the pinch and reshapes the net load
profile, so its kinks move too. On the basic pinch sample the profile
steps at 0.4 of the load at multipliers up to 1.5 and has no step at 2.0.
A size taken from the unscaled problem would be designed at 2.0 as if the
kink were there.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.application.heat_pump_stream_cases import basic_source


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.skipif(not _highs_available(), reason="needs highspy"),
]


def test_only_the_kinks_of_the_scaled_profile_are_designed():
    design = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        dt_cont_multipliers=[2.0],
        depth=0,
        max_parallel=1,
    )

    sizes = {
        round(outcome.network.heat_pumps[0].load_fraction, 6)
        for outcome in design.top(1000)
        if outcome.network.heat_pumps
    }
    assert sizes == {1.0}
