"""Without a load, a heat pump design tries every size worth trying.

The size of a heat pump is chosen outside the design: each size is its
own target and its own networks. The sizes worth trying are the kinks of
the net load profile (``target.hpr_load_kinks``), where the lift a heat
pump needs jumps or steepens, and the full load. A ``load`` in
``heat_pump`` asks for that one size instead.
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

HEAT_PUMP = {
    "levels": {"condensers": 1, "evaporators": 1},
    "search": {"restarts": 0, "evaluations": 1},
}


def _sizes(heat_pump) -> set[float]:
    """The load fractions of the heat pumps in the ranked networks."""

    design = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump=heat_pump, heat_pump_cost=200.0, width=0, depth=0, max_parallel=1
    )
    return {
        round(outcome.network.heat_pumps[0].load_fraction, 6)
        for outcome in design.top(1000)
        if outcome.network.heat_pumps
    }


def test_every_load_kink_and_the_full_load_is_designed():
    # The basic pinch sample's net load profile steps at 0.4 of its load.
    assert _sizes(HEAT_PUMP) == {0.4, 1.0}
