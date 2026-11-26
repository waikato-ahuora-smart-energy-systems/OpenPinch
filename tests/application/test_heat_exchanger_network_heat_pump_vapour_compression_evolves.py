"""A vapour compression heat pump network is evolved, not only designed.

A vapour compression condenser is a stream in segments. From #497 until
#511 every network evolution task of a case with a segmented stream
failed, so a design with a vapour compression heat pump returned its
heat pump networks as pinch design left them, with "Every network
evolution task failed". The suite had no test of an evolved heat pump
network on a segmented condenser, so the loss was found by a review.

On the basic pinch sample at a quarter load, with one evolution level:
the ranking holds an evolved network that carries its heat pump.
"""

from __future__ import annotations

import warnings

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.application.heat_pump_stream_cases import basic_source

EVOLUTION = "network_evolution_method"


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(
    not (_available("highs") and _available("ipopt-pyomo")),
    reason="needs highspy and ipopt",
)
def test_a_water_vapour_compression_heat_pump_network_is_evolved():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        design = problem.design.heat_exchanger_network(
            heat_pump={
                "cycle": "vapour_compression",
                "load": {"fraction": 0.25},
                "levels": {"condensers": 1, "evaporators": 1},
                "search": {"restarts": 0, "evaluations": 1},
                "cycle_options": {"refrigerants": ["water"]},
            },
            heat_pump_cost=200.0,
            width=0,
            depth=1,
            max_parallel=1,
        )

    methods = [
        str(getattr(outcome.method, "value", outcome.method))
        for outcome in design.top(1000)
        if outcome.network.heat_pumps
    ]
    assert EVOLUTION in methods
    assert not [
        warning
        for warning in caught
        if "Every network evolution task failed" in str(warning.message)
    ]
