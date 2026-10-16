"""Progress reports from heat exchanger network synthesis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution import (
    executor as executor_module,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain import progress


def _task(index: int) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        run_id=f"task-{index}",
        method="pinch_design_method",
        approach_temperature=10.0 + index,
    )


@pytest.fixture
def stub_solves(monkeypatch):
    """Tasks that build without a model and solve to a cost of 100 - 10 * i."""
    monkeypatch.setattr(
        LocalSynthesisExecutor, "_build_problem", lambda self, task, **kwargs: task
    )
    monkeypatch.setattr(
        LocalSynthesisExecutor,
        "_model_factories_for_task",
        lambda self, task, problem: None,
    )
    monkeypatch.setattr(
        executor_module, "_evolution_options", lambda problem, task, **kwargs: {}
    )

    def solve(worker_args):
        task = worker_args[0]
        cost = 100.0 - 10.0 * task.approach_temperature + 100.0
        return (
            task,
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=task,
                status="success",
                objective_value=cost,
            ),
            None,
        )

    monkeypatch.setattr(executor_module, "_solve_built_task", solve)


def test_serial_synthesis_reports_each_task_solved_with_the_best_cost(stub_solves):
    events = []
    tasks = [_task(index) for index in range(3)]

    with progress.reporting(events.append, min_interval=0):
        LocalSynthesisExecutor().execute(
            tasks, problem=None, parent_outcomes={}, max_parallel=1
        )

    assert [
        (event.analysis, event.stage, event.done, event.total, event.unit)
        for event in events
    ] == [
        ("hen_synthesis", "pinch_design_method", 0, 3, "tasks"),
        ("hen_synthesis", "pinch_design_method", 1, 3, "tasks"),
        ("hen_synthesis", "pinch_design_method", 2, 3, "tasks"),
        ("hen_synthesis", "pinch_design_method", 3, 3, "tasks"),
    ]
    assert [event.best_objective for event in events] == [None, 100.0, 90.0, 80.0]
    assert events[-1].objective == "total annual cost"


def test_parallel_synthesis_reports_each_task_and_keeps_task_order(stub_solves):
    events = []
    tasks = [_task(index) for index in range(3)]
    executor = LocalSynthesisExecutor(
        worker_pool_factory=lambda workers: ThreadPoolExecutor(max_workers=workers)
    )

    with progress.reporting(events.append, min_interval=0):
        outcomes = executor.execute(
            tasks, problem=None, parent_outcomes={}, max_parallel=2
        )

    assert [event.done for event in events] == [0, 1, 2, 3]
    assert [outcome.task.task_id for outcome in outcomes] == [
        task.task_id for task in tasks
    ]
