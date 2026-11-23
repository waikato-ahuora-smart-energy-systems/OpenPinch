"""A ``load`` in ``heat_pump`` asks for that one size.

The sweep over the load kinks is what a design does when it is not told a
size. Told one, it designs that size at each multiplier and no other.
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


def test_a_load_in_the_request_is_the_only_size_designed():
    design = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "load": {"fraction": 0.7},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )

    sizes = {
        round(outcome.network.heat_pumps[0].load_fraction, 6)
        for outcome in design.top(1000)
        if outcome.network.heat_pumps
    }
    assert sizes == {0.7}
