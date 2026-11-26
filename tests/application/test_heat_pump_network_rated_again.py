"""A heat pump network is rated and retrofitted on the problem it was
designed for.

The network names the heat pump's condenser and evaporator streams, which
the problem does not hold: they were in the case the design derived. Its
machine carries them, so the problem builds that case again for a rating,
a duty optimisation or a retrofit of the network, from the object or from
its JSON.
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


@pytest.fixture(scope="module")
def pumped():
    """The cheapest pinch design network around a full-load Carnot heat
    pump on the basic pinch case."""
    view = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )
    return next(o.network for o in view.top(1000) if o.network.heat_pumps)


def test_the_networks_json_is_rated_at_the_cost_it_was_designed_at(pumped):
    """It was refused: "'Plant.HP_H1' is not a hot process stream"."""
    problem = PinchProblem(source=basic_source(), project_name="basic")

    rated = problem.design.optimise_duties(
        pumped.model_dump(mode="json"), objective="rating"
    ).selected_network

    assert rated.heat_pumps == pumped.heat_pumps
    assert rated.total_annual_cost == pytest.approx(pumped.total_annual_cost, rel=1e-6)


def test_a_rating_on_the_networks_own_case_keeps_the_machine(pumped):
    """A problem that already holds the machine's streams (the case, built
    again) rates the network as the plain problem does. It returned the
    network without its machine or the machine's cost."""
    from OpenPinch.application.heat_pump_design import heat_pump_network_case

    case = heat_pump_network_case(
        PinchProblem(source=basic_source(), project_name="basic"), pumped
    )

    rated = case.design.optimise_duties(pumped, objective="rating").selected_network

    assert rated.heat_pumps == pumped.heat_pumps
    assert rated.total_annual_cost == pytest.approx(pumped.total_annual_cost, rel=1e-6)


def test_a_retrofit_takes_the_networks_json_as_its_existing_network(pumped):
    """The existing network draws the utility it was designed to draw, and
    the retrofit's network keeps the machine. The design's utility cost
    includes the machine's electricity, as a retrofit's heat pump step
    reports it; the network's own is its utilities'."""
    problem = PinchProblem(source=basic_source(), project_name="basic")

    result = problem.design.retrofit(
        pumped.model_dump(mode="json"), max_modifications=0
    )

    existing = result.frontier[0]
    assert existing.network.utility_cost == pytest.approx(pumped.utility_cost, rel=1e-6)
    assert existing.utility_cost == pytest.approx(
        pumped.utility_cost + pumped.heat_pumps[0].electricity_cost, rel=1e-6
    )
    assert existing.network.heat_pumps == pumped.heat_pumps


def test_a_retrofit_whose_heat_pump_step_serves_nothing_adds_no_machine():
    """A retrofit with ``heat_pump_cycle`` on a network that holds a machine,
    at half load here. The step's Carnot target of the heaters left serves
    no duty, which is no target, so the step is left out and every design
    keeps the one machine. That target used to come back as a success of
    0 kW, and the retrofit recorded it as a heat pump modification and a
    second machine with no streams."""
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

    with pytest.warns(UserWarning, match="step at size 1 was left out of the"):
        result = problem.design.retrofit(
            half.model_dump(mode="json"),
            max_modifications=1,
            heat_pump_cycle="carnot",
            heat_pump_cost=200.0,
            heat_pump_sizes=(1.0,),
        )

    for design in result.frontier:
        assert not any(m.kind == "heat_pump" for m in design.modifications)
        assert design.network.heat_pumps == half.heat_pumps
