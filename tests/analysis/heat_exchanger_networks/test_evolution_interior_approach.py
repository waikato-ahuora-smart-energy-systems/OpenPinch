"""Network evolution designs a case with a segmented stream.

Evolution's non-isothermal model has no integers and is solved with
Ipopt, which holds a variable's bounds to a relative tolerance: a branch
outlet at its stream's 140 degC target came back 4e-6 K past it. The
interior approach check looked that outlet up on the stream's profile,
which raises outside its range, so every evolution task of a case with a
segmented stream failed and the design returned its pinch design
networks with "Every network evolution task failed".
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
from tests.analysis.heat_exchanger_networks.kink_case import kink_problem

EVOLUTION = "network_evolution_method"


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _methods(view) -> list[str]:
    return [
        str(getattr(outcome.method, "value", outcome.method))
        for outcome in view.result.ranked_networks
    ]


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs highspy and ipopt",
)
def test_the_default_design_evolves_a_stream_that_kinks_inside_a_match():
    problem = kink_problem({"HENS_MAX_PARALLEL": 1})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view = problem.design.heat_exchanger_network()

    assert EVOLUTION in _methods(view)
    assert not [
        warning
        for warning in caught
        if "Every network evolution task failed" in str(warning.message)
    ]


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs highspy and ipopt",
)
def test_evolution_designs_a_segmented_hot_and_a_segmented_cold_stream():
    # The kink case's condensing stream V beside a cold stream W whose
    # heat capacity flowrate falls from 2 to 1 kW/K at 90 degC.
    source = kink_problem().to_problem_json()
    source["streams"].append(
        {
            "zone": "Plant",
            "name": "W",
            "dt_cont": {"value": 5.0, "unit": "delta_degC"},
            "segments": [
                {"t_supply": 30.0, "t_target": 90.0, "heat_flow": 120.0},
                {"t_supply": 90.0, "t_target": 130.0, "heat_flow": 40.0},
            ],
        }
    )
    source["options"] = {
        **source.get("options", {}),
        "HENS_DT_CONT_MULTIPLIERS": [1.0],
        "HENS_MAX_PARALLEL": 1,
    }
    problem = PinchProblem(source=source, project_name="segmented pair")
    seeds = [
        outcome.network
        for outcome in problem.design.pinch_design().result.ranked_networks
    ]

    view = problem.design.network_evolution(seeds)

    assert set(_methods(view)) == {EVOLUTION}
