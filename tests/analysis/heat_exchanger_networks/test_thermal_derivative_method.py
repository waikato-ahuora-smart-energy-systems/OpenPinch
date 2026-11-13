"""Standalone thermal-derivative method tests."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    SynthesisWorkflowSettings,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting import (
    pinch_design_method,
    thermal_derivative_method,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.thermal_derivative_method import (
    build_seeded_thermal_derivative_method_tasks,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def test_thermal_derivative_method_builds_seeded_tasks() -> None:
    tasks = build_seeded_thermal_derivative_method_tasks(
        _settings(),
        (_seed_network(method="pinch_design_method"),),
    )

    assert len(tasks) == 1
    assert tasks[0].method == "thermal_derivative_method"
    assert tasks[0].seed_network_index == 0
    assert tasks[0].topology_restrictions


def test_pinch_design_tasks_run_each_multiplier_and_the_stage_pair_sweep() -> None:
    settings = _settings(
        method_sequence=("pinch_design_method",),
        dt_cont_multipliers=(1.0, 1.5, 1.0),
        pdm_stage_pair_limit=2,
    )

    tasks = pinch_design_method.build_pinch_design_method_tasks(settings)

    assert [task.dt_cont_multiplier for task in tasks] == [1.0, 1.5, 1.0, 1.0]
    assert [task.settings.get("stage_selection") for task in tasks] == [
        None,
        None,
        [1, 2],
        [2, 1],
    ]
    assert (
        len(
            pinch_design_method.build_pinch_design_method_tasks(
                _settings(dt_cont_multipliers=(1.0, 1.5), pdm_stage_pair_limit=0)
            )
        )
        == 2
    )


def test_thermal_derivative_skips_failed_parents_and_empty_default_stage():
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        dt_cont_multiplier=1.0,
    )

    tasks, outcomes = thermal_derivative_method.execute_thermal_derivative_method_stage(
        problem=object(),
        settings=_settings(),
        pdm_outcomes=[],
        parent_outcomes={},
    )

    assert tasks == ()
    assert outcomes == ()

    failed_outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="failed",
        error="solver failed",
    )
    assert (
        thermal_derivative_method.build_thermal_derivative_method_tasks(
            _settings(),
            [failed_outcome],
        )
        == ()
    )


def test_seeded_quality_thermal_derivative_tasks_deduplicate_topologies() -> None:
    settings = _settings(derivative_thresholds=(0.25, 0.25, 0.5))
    seed = _seed_network(method="pinch_design_method")

    tasks = build_seeded_thermal_derivative_method_tasks(settings, (seed, seed))

    assert len(tasks) == 2
    assert [task.derivative_threshold for task in tasks] == [0.25, 0.5]
    assert {task.seed_network_index for task in tasks} == {0}
    assert all(task.stage_count == 1 for task in tasks)


def test_thermal_derivative_parent_task_builder_accepts_plain_successful_pdm():
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        dt_cont_multiplier=1.0,
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=_seed_network(method="pinch_design_method"),
    )

    tasks = thermal_derivative_method.build_thermal_derivative_method_tasks(
        _settings(),
        [outcome],
    )

    assert len(tasks) == 1
    assert tasks[0].parent_task_id == task.task_id
    assert tasks[0].topology_restrictions


def test_pinch_design_workflow_wrappers_use_executor_contract(monkeypatch):
    settings = _settings(
        method_sequence=("pinch_design_method",), heuristic_pdm_networks=0
    )
    default_executor = _RecordingExecutor()
    monkeypatch.setattr(
        pinch_design_method,
        "LocalSynthesisExecutor",
        lambda: default_executor,
    )

    tasks, outcomes = pinch_design_method.execute_pinch_design_method_stage(
        problem=object(),
        settings=settings,
    )

    assert tasks
    assert len(outcomes) == len(tasks)
    assert default_executor.calls[0]["parent_outcomes"] == {}

    workflow_executor = _RecordingExecutor()
    result = pinch_design_method._execute_pinch_design_method_workflow(
        problem=object(),
        settings=settings,
        executor=workflow_executor,
    )

    assert result.accepted_result.method == "pinch_design_method"
    assert workflow_executor.calls[0]["max_parallel"] == settings.max_parallel


def test_thermal_derivative_workflow_wrappers_use_executor_contract(monkeypatch):
    settings = _settings()
    default_executor = _RecordingExecutor()
    monkeypatch.setattr(
        thermal_derivative_method,
        "LocalSynthesisExecutor",
        lambda: default_executor,
    )

    tasks, outcomes = (
        thermal_derivative_method.execute_seeded_thermal_derivative_method_stage(
            problem=object(),
            settings=settings,
            seed_networks=(_seed_network(method="pinch_design_method"),),
        )
    )

    assert tasks
    assert len(outcomes) == len(tasks)
    assert default_executor.calls[0]["parent_outcomes"] == {}

    workflow_executor = _RecordingExecutor()
    result = thermal_derivative_method._execute_thermal_derivative_method_workflow(
        problem=object(),
        settings=settings,
        seed_networks=(_seed_network(method="pinch_design_method"),),
        executor=workflow_executor,
    )

    assert result.accepted_result.method == "thermal_derivative_method"
    assert workflow_executor.calls[0]["max_parallel"] == settings.max_parallel


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
        self.calls.append(
            {
                "tasks": tasks,
                "problem": problem,
                "parent_outcomes": parent_outcomes,
                "max_parallel": max_parallel,
            }
        )
        return tuple(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=task,
                status="success",
                network=_seed_network(method=_method_value(task.method)),
                objective_value=float(index + 1),
                solver_status="ok",
            )
            for index, task in enumerate(tasks)
        )


def _method_value(method) -> str:
    return method.value if hasattr(method, "value") else str(method)


def _settings(**updates) -> SynthesisWorkflowSettings:
    params = dict(
        run_id="method-test",
        dt_cont_multipliers=(1.0,),
        derivative_thresholds=(0.5,),
        stage_selection=(1,),
        method_sequence=("thermal_derivative_method",),
        output_formats=(),
        solve_tolerance=1e-3,
        best_solutions_to_save=1,
        max_parallel=1,
        pdm_solver="couenne",
        tdm_solver="couenne",
        evm_solver="ipopt-pyomo",
        pdm_solver_options={},
        tdm_solver_options={},
        evm_solver_options={},
    )
    params.update(updates)
    return SynthesisWorkflowSettings(**params)


def _seed_network(*, method: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="seed-recovery",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Process,
                stage=1,
                period_states=(
                    HeatExchangerPeriodState(
                        period_id="0",
                        period_idx=0,
                        duty=100.0,
                        approach_temperatures=(10.0,),
                    ),
                ),
                area=10.0,
            ),
        ),
        run_id="seed-run",
        method=method,
        stage_count=1,
        summary_metrics={"dt_cont_multiplier": 1.0},
    )
