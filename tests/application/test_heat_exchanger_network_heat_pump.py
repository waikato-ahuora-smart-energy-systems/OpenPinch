"""``heat_exchanger_network(heat_pump=...)`` designs with a heat pump too.

At each dt_cont multiplier the heat pump is targeted, its streams join the
case and a network is designed around them. Those networks, each carrying
the machine's electricity and capital, are ranked with the networks
designed without a heat pump, so the result says whether the heat pump
pays.
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

# The target's first start alone: see ``full_load_target``.
HEAT_PUMP = {
    "cycle": "carnot",
    "levels": {"condensers": 1, "evaporators": 1},
    "search": {"restarts": 0, "evaluations": 1},
}


@pytest.fixture(scope="module")
def ranked():
    """Every network of the basic pinch case at three multipliers, with and
    without a Carnot heat pump at full load."""

    problem = PinchProblem(source=basic_source(), project_name="basic")
    design = problem.design.heat_exchanger_network(
        heat_pump=HEAT_PUMP, width=1, depth=0, max_parallel=1
    )
    return design, design.top(1000)


def test_a_heat_pump_network_is_designed_at_every_multiplier(ranked):
    _design, outcomes = ranked

    with_pump = [o.network for o in outcomes if o.network.heat_pumps]

    assert {n.summary_metrics["dt_cont_multiplier"] for n in with_pump} == {
        0.75,
        1.0,
        1.5,
    }
    # The heat pump takes the whole heating load: no hot utility is left.
    for network in with_pump:
        assert network.summary_metrics["hot_utility_load"] == pytest.approx(
            0.0, abs=0.5
        )
