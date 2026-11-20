"""A thermal derivative task stops at a time limit unless the user sets one.

The thermal derivative step solves a MINLP per parent network and threshold,
which can run for hours in Couenne. Each task takes
``backend.TDM_TIME_LIMIT`` seconds by default; a limit the user gives in the
task's solver options is kept, and other stages are unchanged.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend


def _built_options(framework, solver, solver_options=None, integers=True):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        raise ValueError("built")

    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(),
        framework=framework,
        solver=solver,
        stages=2,
        integers=integers,
        solver_options=solver_options,
    )
    with pytest.raises(ValueError, match="built"):
        problem.load_model(model_factories={"stagewise": factory})
    return seen["solver_options"]


@pytest.mark.parametrize(
    ("solver", "option"), [("couenne", "time_limit"), ("scip", "limits/time")]
)
def test_a_thermal_derivative_task_has_a_default_time_limit(solver, option):
    assert _built_options("TDM", solver) == {option: backend.TDM_TIME_LIMIT}
    assert backend.TDM_TIME_LIMIT == pytest.approx(300.0)


def test_a_users_time_limit_is_kept():
    assert _built_options("TDM", "couenne", {"time_limit": 30.0}) == {
        "time_limit": 30.0
    }


def test_evolution_tasks_take_no_tdm_limit():
    assert _built_options("ESM", "ipopt-pyomo", integers=False) is None
