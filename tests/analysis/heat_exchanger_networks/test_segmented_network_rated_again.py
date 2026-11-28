"""A network on a condensing stream is a feasible point of its own rating.

The kink case's hot stream V condenses over 0.01 K, as a water vapour
compression condenser does. The fixed-structure model rounded that
boundary over 0.05 K and took V for a 550 kW stream, not a 187 kW one, so
the network a design returned could not be rated, cost-optimised or
retrofitted on the problem it was designed for.
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


pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.solver,
    pytest.mark.skipif(
        not (_available("highs") and _available("ipopt-pyomo")),
        reason="needs highspy and ipopt",
    ),
]


@pytest.fixture(scope="module")
def designed():
    """The kink case's pinch design network."""
    view = kink_problem({"HENS_MAX_PARALLEL": 1}).design.heat_exchanger_network(
        width=0, depth=0
    )
    return view.selected_network


def test_a_network_on_a_condensing_stream_rates_at_its_designed_cost(designed):
    rated = kink_problem().design.optimise_duties(designed, objective="rating")

    assert rated.selected_network.total_annual_cost == pytest.approx(
        designed.total_annual_cost, rel=1e-6
    )


def test_a_network_on_a_condensing_stream_is_cost_optimised(designed):
    view = kink_problem().design.optimise_duties(designed, objective="cost")

    optimised = view.selected_network
    assert optimised.total_annual_cost <= designed.total_annual_cost * (1 + 1e-6)
    # Every exchanger on the condensing stream, however many the design
    # gave it: together they take its 187.47 kW, not the 550 kW it was to
    # the model.
    condensing = sum(
        exchanger.period_states[0].duty
        for exchanger in optimised.exchangers
        if exchanger.source_stream == "Plant.V"
    )
    assert condensing == pytest.approx(187.47, rel=1e-6)


def test_a_retrofit_takes_a_network_on_a_condensing_stream(designed):
    result = kink_problem().design.retrofit(designed, max_modifications=0)

    (existing,) = result.frontier
    assert existing.modifications == ()
    assert existing.utility_cost == pytest.approx(designed.utility_cost, rel=1e-6)


def test_a_design_on_a_condensing_stream_takes_an_overdesign_margin(designed):
    # The margin is applied by rating the design's network with more area.
    view = kink_problem({"HENS_MAX_PARALLEL": 1}).design.heat_exchanger_network(
        width=0, depth=0, overdesign_percentage=10.0
    )

    assert view.selected_network.total_annual_cost > designed.total_annual_cost
