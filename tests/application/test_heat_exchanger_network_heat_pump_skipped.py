"""A multiplier with no heat pump is left out with a warning.

A heat pump target can fail at one multiplier and not another: at wider
approaches there may be no design that beats the baseline. The other
multipliers' heat pump networks, and every network without a heat pump,
are still ranked.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application import heat_pump_design
from OpenPinch.contracts.hpr import HPRTargetingError
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


def test_a_multiplier_whose_target_fails_is_skipped_with_a_warning(monkeypatch):
    solve = heat_pump_design.heat_pump_design_case

    def fails_at_three_quarters(problem, multiplier, **arguments):
        if multiplier == 0.75:
            raise HPRTargetingError("found no heat pump design", diagnostics={})
        return solve(problem, multiplier, **arguments)

    monkeypatch.setattr(
        heat_pump_design, "heat_pump_design_case", fails_at_three_quarters
    )
    problem = PinchProblem(source=basic_source(), project_name="basic")

    with pytest.warns(UserWarning, match="multiplier 0.75.*found no heat pump design"):
        design = problem.design.heat_exchanger_network(
            heat_pump={
                "levels": {"condensers": 1, "evaporators": 1},
                "search": {"restarts": 0, "evaluations": 1},
            },
            width=1,
            depth=0,
            max_parallel=1,
        )

    with_pump = [o.network for o in design.top(1000) if o.network.heat_pumps]
    assert {n.summary_metrics["dt_cont_multiplier"] for n in with_pump} == {1.0, 1.5}
