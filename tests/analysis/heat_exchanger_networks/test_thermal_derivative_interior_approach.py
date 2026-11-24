"""The thermal derivative step keeps the approach inside a match on a
segmented stream.

On the kink case, seeded with its pinch design, the stage-wise model
matched the condensing stream whole against a cold stream that ran hotter
than its plateau past the desuperheat. Both match ends kept their
approach; the crossing at the kink was found only when the network's
areas were taken ("Segmented exchanger slice has a non-positive terminal
temperature difference"), and the task failed with no network.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.analysis.heat_exchanger_networks.kink_case import kink_problem


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('couenne'))",
    reason="needs highspy and couenne",
)
def test_thermal_derivative_designs_a_stream_that_kinks_inside_a_match():
    problem = kink_problem({"HENS_DT_CONT_MULTIPLIERS": [1.0]})
    seeds = [
        outcome.network
        for outcome in problem.design.pinch_design().result.ranked_networks
    ]

    view = problem.design.thermal_derivative(seeds)

    assert [outcome.network for outcome in view.result.ranked_networks]
