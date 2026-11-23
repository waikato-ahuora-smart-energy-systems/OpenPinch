"""Pinch design that cannot honour forbidden matches hands over to evolution.

A pinch side takes no heater below the pinch and no cooler above it, so a
side that loses a recovery match it needs has no design at its utility
target: on a four-stream case, forbidding one match the best design uses
failed both pinch design methods and the run. The stage-wise model has no
such limit. With forbidden matches set and no pinch design network, the run
starts network evolution directly, as it does when the pinch design solver
is missing.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.errors import WorkflowContractError
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method import (
    execute_open_hens_method,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from tests.analysis.heat_exchanger_networks.test_full_sequence import (
    _small_problem,
    _small_problem_with_options,
)

PINCH_DESIGN = {"pinch_design_method", "heuristic_pinch_design_method"}


class _NoPinchDesign(FakeSynthesisExecutor):
    """Every pinch design task fails, as an infeasible pinch side does."""

    def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
        if tasks and tasks[0].method in PINCH_DESIGN:
            return tuple(
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="failed",
                    solver_status="failed",
                    error="Pinch Decomposition failed: Above 1 Below 0 dTmin 10.0",
                )
                for task in tasks
            )
        return super().execute(
            tasks,
            problem=problem,
            parent_outcomes=parent_outcomes,
            max_parallel=max_parallel,
        )


def test_no_pinch_design_with_forbidden_matches_starts_evolution_directly():
    arrays = problem_to_solver_arrays(_small_problem(), 10.0)
    pair = [
        arrays.stream_identities["hot_process_streams"][0],
        arrays.stream_identities["cold_process_streams"][0],
    ]
    problem = _small_problem_with_options({"HENS_FORBIDDEN_MATCHES": [pair]})

    with pytest.warns(UserWarning, match="HENS_FORBIDDEN_MATCHES"):
        result = execute_open_hens_method(
            problem, workflow_settings_from_problem(problem), executor=_NoPinchDesign()
        )

    evolved = [
        outcome
        for outcome in result.outcomes
        if outcome.task.method == "network_evolution_method"
        and outcome.status == "success"
    ]
    assert evolved
    assert all(outcome.task.parent_task_id is None for outcome in evolved)
    assert result.accepted_result.network is not None

    # Without the option a run with no pinch design network fails as before.
    unrestricted = _small_problem()
    with pytest.raises(WorkflowContractError, match="no successful task outcomes"):
        execute_open_hens_method(
            unrestricted,
            workflow_settings_from_problem(unrestricted),
            executor=_NoPinchDesign(),
        )
