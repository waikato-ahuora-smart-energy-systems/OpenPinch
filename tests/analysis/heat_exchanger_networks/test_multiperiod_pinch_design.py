"""Multiperiod PDM with streams that are inactive in some periods (audit H7).

The pinch decomposition clips a stream that is absent from one side of the
pinch in a period to supply = target = 0 K there. Approach constraints,
theta and area must not apply to it in that period, or the shared topology
cannot be solved (``crude_preheat_train_multiperiod`` failed at every dTmin).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
)
from OpenPinch.application.problem import PinchProblem

CRUDE_MULTIPERIOD = "crude_preheat_train_multiperiod.json"


def _require_couenne() -> None:
    try:
        require_solver_binary("couenne", purpose="multiperiod PDM regression")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError) as exc:
        pytest.skip(str(exc))


@pytest.mark.solver
@pytest.mark.parametrize("d_tmin", [5.0, 10.0, 20.0])
def test_crude_multiperiod_pinch_design_solves(d_tmin):
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
        verify_network_feasibility,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
        execute_pinch_design_method_stage,
    )

    _require_couenne()
    problem = PinchProblem(CRUDE_MULTIPERIOD)
    problem.target.all_heat_integration()
    settings = replace(
        workflow_settings_from_problem(problem),
        approach_temperatures=(d_tmin,),
        max_parallel=1,
    )

    _tasks, outcomes = execute_pinch_design_method_stage(
        problem=problem,
        settings=settings,
        executor=LocalSynthesisExecutor(print_output=False),
    )

    solved = [outcome for outcome in outcomes if outcome.status == "success"]
    assert solved, [outcome.error for outcome in outcomes]
    for outcome in solved:
        assert verify_network_feasibility(outcome.network) == ()
