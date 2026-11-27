"""A retrofit adds a heat pump beside the one a network already holds.

The retrofit's heat pump step named its streams as every heat pump target
does (``HP_H1``, ``HP_C1``), the names the existing machine's streams
already hold in the case built from the network. The step's case was
refused ("already has a stream named 'HP_H1' ... rename the process
stream") and the step left out at every size, so such a network could
never gain a second machine.
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


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.solver,
    pytest.mark.skipif(
        not (_available("highs") and _available("ipopt-pyomo")),
        reason="needs highspy and ipopt",
    ),
]


def test_a_retrofit_adds_a_machine_beside_the_existing_one():
    view = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "load": {"fraction": 0.5},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )
    half = next(o.network for o in view.top(1000) if o.network.heat_pumps)
    problem = PinchProblem(source=basic_source(), project_name="basic")

    result = problem.design.retrofit(
        half.model_dump(mode="json"),
        max_modifications=1,
        heat_pump_cycle="carnot",
        heat_pump_cost=200.0,
        heat_pump_sizes=(1.0,),
        # Electricity cheap enough for a second machine to pay on the
        # heaters the first leaves (at the default 100 $/MWh none does),
        # and a search long enough to find one the network can serve.
        heat_pump_search={"maximum_evaluations": 300},
        options={"COSTING_HPR_ELE_PRICE": 20.0},
    )

    stepped = [
        design
        for design in result.frontier
        if any(m.kind == "heat_pump" for m in design.modifications)
    ]
    assert stepped
    existing, added = stepped[0].network.heat_pumps
    assert existing == half.heat_pumps[0]
    assert added.machine != existing.machine
    assert added.condenser_duty > 1.0
    own = {record["name"] for record in added.streams}
    assert own and own.isdisjoint(record["name"] for record in existing.streams)
