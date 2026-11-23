"""A design's costing options price its heat pumps too.

``options`` of ``heat_exchanger_network`` set the prices and the
annualisation a design costs its networks with, for that call alone. The
heat pump's electricity and compressor are part of the same total, so
they take the same values: an electricity price given for the call, not
the one stored on the problem.
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


def test_an_electricity_price_given_for_the_call_prices_the_heat_pump():
    design = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        # The problem stores 100 $/MWh and 8300 h.
        options={"COSTING_HPR_ELE_PRICE": 50.0, "COSTING_ANNUAL_OP_TIME": 8000.0},
        width=0,
        depth=0,
        max_parallel=1,
    )

    machine = next(
        outcome.network.heat_pumps[0]
        for outcome in design.top(1000)
        if outcome.network.heat_pumps
    )
    assert machine.electricity_cost == pytest.approx(machine.work * 50.0 * 8.0)
