"""The basic pinch sample designed with a Carnot heat pump, searched.

The numbers of the review that asked for this design: at every dt_cont
multiplier a network is returned at its heat pump target; at 1.0 the
machine draws 263.4 kW, 218,636 $/y at 100 $/MWh over 8300 h; and at the
sample's 10 $/MWh steam the heat pump does not pay, so a design without
it is selected.
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


pytestmark = pytest.mark.synthesis


# The solver lane: the heat pump search takes 2 s a multiplier.
@pytest.mark.solver
@pytest.mark.skipif(not _highs_available(), reason="needs highspy")
def test_the_basic_pinch_sample_with_a_searched_carnot_heat_pump():
    design = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            # The review's numbers are for the full load; without a load
            # the sizes at the load kinks are designed as well.
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 1},
        },
        heat_pump_cost=200.0,
        width=1,
        depth=0,
        max_parallel=1,
    )

    best = {}
    for outcome in design.top(1000):
        network = outcome.network
        if network.heat_pumps:
            best.setdefault(network.summary_metrics["dt_cont_multiplier"], network)
    assert sorted(best) == [0.75, 1.0, 1.5]
    for network in best.values():
        assert network.summary_metrics["hot_utility_load"] == pytest.approx(
            0.0, abs=0.5
        )
    machine = best[1.0].heat_pumps[0]
    assert machine.work == pytest.approx(263.4, abs=0.1)
    assert machine.electricity_cost == pytest.approx(218_636.0, rel=1e-3)
    assert design.selected_network.heat_pumps == ()
