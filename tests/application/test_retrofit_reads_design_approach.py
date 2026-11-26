"""A retrofit takes the network a design has just returned.

With no ``min_approach_temperature`` the retrofit holds every exchanger to
the approach the network records it was designed to: its streams'
contributions at its dt_cont multiplier and, for an evolved network, at its
approach share.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)


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
@pytest.mark.parametrize(
    "design",
    [
        # Pinch design at 0.75: every match closes to 7.5 K, not 10 K.
        {"width": 0, "depth": 0, "dt_cont_multipliers": [0.75]},
        # Evolution at the default share: matches close to a quarter of 10 K.
        {"width": 0, "depth": 1},
    ],
    ids=["pinch design at 0.75", "evolved"],
)
def test_retrofit_rates_a_designed_network_as_it_was_designed(design):
    """The existing design of a retrofit is the designed network itself:
    it is accepted, and at its own areas it draws the utility the design
    draws. It was refused ("does not meet its targets at its installed
    areas") for a design below a multiplier of 1 and for an evolved one."""
    problem = PinchProblem("basic_pinch.json")
    network = problem.design.heat_exchanger_network(**design).selected_network

    # An evolved network sits on its area limits, so at its exact areas the
    # solver may find it just infeasible; 0.1 % of area settles that and
    # moves the utility cost by less than 0.1 % (0.04 % here).
    result = PinchProblem("basic_pinch.json").design.retrofit(
        network.model_dump(mode="json"), max_modifications=0, area_tolerance=1e-3
    )

    assert result.frontier[0].utility_cost == pytest.approx(
        network.utility_cost, rel=1e-3
    )
