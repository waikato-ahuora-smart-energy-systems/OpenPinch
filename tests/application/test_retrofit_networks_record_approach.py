"""A retrofit's networks record the approach the retrofit held.

A network ``design.retrofit`` returns is read again by a rating, an
overdesign or a second retrofit at the approach it records: its streams'
contributions at the multiplier the retrofit scaled them by, or the
``min_approach_temperature`` the retrofit was given.
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
def test_a_second_retrofit_holds_the_contributions_the_first_held():
    """A retrofit of a pinch design made at 0.75 holds every match to 7.5 K.
    Its selected network, given back as JSON, is rated at that 7.5 K and so
    at the utility cost the first retrofit reported. It was read as an
    evolved network at a quarter of 10 K, which rated it 0.9 % cheaper."""
    design = PinchProblem("basic_pinch.json").design.heat_exchanger_network(
        width=0, depth=0, dt_cont_multipliers=[0.75]
    )
    first = PinchProblem("basic_pinch.json").design.retrofit(
        design.selected_network, max_modifications=0
    )

    second = PinchProblem("basic_pinch.json").design.retrofit(
        first.selected.network.model_dump(mode="json"), max_modifications=0
    )

    assert second.frontier[0].utility_cost == pytest.approx(
        first.selected.utility_cost, rel=1e-6
    )


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(
    not (_available("highs") and _available("ipopt-pyomo")),
    reason="needs highspy and ipopt",
)
def test_a_second_retrofit_holds_the_approach_the_first_was_given():
    """A retrofit given ``min_approach_temperature`` holds every exchanger to
    it, and so does a second retrofit of its selected network that is given
    none. The 7.5 K was not recorded, and the second retrofit read the
    network as an evolved one at a quarter of 10 K."""
    design = PinchProblem("basic_pinch.json").design.heat_exchanger_network(
        width=0, depth=0, dt_cont_multipliers=[0.75]
    )
    first = PinchProblem("basic_pinch.json").design.retrofit(
        design.selected_network, max_modifications=0, min_approach_temperature=7.5
    )

    second = PinchProblem("basic_pinch.json").design.retrofit(
        first.selected.network.model_dump(mode="json"), max_modifications=0
    )

    assert second.frontier[0].utility_cost == pytest.approx(
        first.selected.utility_cost, rel=1e-6
    )
