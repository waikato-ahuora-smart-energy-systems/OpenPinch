"""The ranking decides whether the heat pump pays.

A heat pump network's total annual cost is its utilities, its exchangers
and the machine's electricity and capital. At the basic pinch sample's
prices (steam at 10 $/MWh, electricity at 100 $/MWh) the machine costs far
more than the steam it saves, and a design without it is selected. With
steam at 200 $/MWh the heat pump design is.
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

HEAT_PUMP = {
    "cycle": "carnot",
    "levels": {"condensers": 1, "evaporators": 1},
    "search": {"restarts": 0, "evaluations": 1},
}


def _design(steam_price: float):
    source = basic_source()
    for utility in source["utilities"]:
        if utility["type"] == "Hot":
            utility["price"] = {"value": steam_price, "unit": "$/MWh"}
    return PinchProblem(
        source=source, project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump=HEAT_PUMP, heat_pump_cost=200.0, width=0, depth=0, max_parallel=1
    )


@pytest.mark.parametrize(("steam_price", "pumps"), [(10.0, 0), (200.0, 1)])
def test_the_selected_design_has_a_heat_pump_only_when_it_pays(steam_price, pumps):
    design = _design(steam_price)

    selected = design.selected_network

    assert len(selected.heat_pumps) == pumps
    for outcome in design.top(1000):
        network = outcome.network
        assert network.total_annual_cost == pytest.approx(
            network.utility_cost
            + network.capital_cost
            + sum(machine.annual_cost for machine in network.heat_pumps)
        )
        assert selected.total_annual_cost <= network.total_annual_cost + 1e-6
