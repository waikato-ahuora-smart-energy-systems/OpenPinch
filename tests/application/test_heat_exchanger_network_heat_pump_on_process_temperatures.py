"""A converged heat pump target gives a heat pump network.

A search that converges puts the condenser at the temperature a cold
stream ends at, to its own tolerance: 1e-5 K under it on the basic pinch
case at a multiplier of 2.0. The case then had a second pinch at the
condenser, no pinch design closed, and the design came back without a
heat pump ("No heat pump network at dt_cont multiplier 2"). Most tests
take the search's first start, which is never that close.
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


def test_a_converged_target_gives_a_heat_pump_network_at_a_multiplier_of_two():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    view = problem.design.heat_exchanger_network(
        heat_pump={
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            # One run of the search, which converges (4 s).
            "search": {"restarts": 1},
            # Heat at the electricity price: at the steam's own price no
            # heat pump pays on this case.
            "options": {"COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE": 1.0},
        },
        heat_pump_cost=200.0,
        dt_cont_multipliers=[2.0],
        width=0,
        depth=0,
        max_parallel=1,
    )

    pumped = [o.network for o in view.top(1000) if o.network.heat_pumps]
    assert pumped
    assert pumped[0].heat_pumps[0].condenser_duty == pytest.approx(1150.0, abs=0.01)
