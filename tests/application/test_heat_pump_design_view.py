"""Designs with and without the heat pump are ranked in one list.

A heat pump pays or it does not. The answer comes from ranking the
networks designed around its streams, each carrying the machine's
electricity and capital, beside the networks designed without it.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.heat_pump_design import (
    heat_pump_design_case,
    heat_pump_design_view,
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

DESIGN = {"width": 0, "depth": 0, "max_parallel": 1}


@pytest.fixture(scope="module")
def designs():
    """The basic pinch case designed without a heat pump and with one."""

    problem = PinchProblem(source=basic_source(), project_name="basic")
    plain = problem.design.heat_exchanger_network(**DESIGN)
    plain_costs = sorted(o.network.total_annual_cost for o in plain.top(1000))
    run = heat_pump_design_case(
        problem,
        1.0,
        cycle="carnot",
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 1, "evaluations": 100},
        cost_per_kw=200.0,
    )
    pumped = run.case.design.heat_exchanger_network(dt_cont_multipliers=[1.0], **DESIGN)
    pumped_costs = sorted(o.network.total_annual_cost for o in pumped.top(1000))
    merged = heat_pump_design_view(problem, plain, [(run, pumped)])
    return run, plain_costs, pumped_costs, merged


def test_every_network_of_both_designs_is_ranked_with_its_full_cost(designs):
    run, plain_costs, pumped_costs, merged = designs

    ranked = merged.top(1000)

    without = [o.network for o in ranked if not o.network.heat_pumps]
    with_pump = [o.network for o in ranked if o.network.heat_pumps]
    assert sorted(n.total_annual_cost for n in without) == pytest.approx(plain_costs)
    assert len(with_pump) == len(pumped_costs)
    assert all(n.heat_pumps == (run.record,) for n in with_pump)
    assert sorted(n.total_annual_cost for n in with_pump) == pytest.approx(
        [cost + run.record.annual_cost for cost in pumped_costs]
    )
    costs = [o.network.total_annual_cost for o in ranked]
    assert costs == sorted(costs)
    assert merged.selected_network.total_annual_cost == pytest.approx(costs[0])
