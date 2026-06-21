"""Network evolution method orchestration for HEN synthesis."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Sequence

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from ....lib.enums import HENDesignMethod
from ....lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ..common.execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..common.execution.settings import SynthesisWorkflowSettings
from ..common.execution.task_builders import (
    _required_stage_count,
    _successful_method,
    approach_temperature_from_network,
    derivative_threshold_from_network,
    required_topology_restrictions_from_outcome,
    stage_count_from_network,
    topology_restrictions_from_network,
)
from ..common.results.assembly import SynthesisWorkflowResult, build_synthesis_result


def _execute_network_evolution_method_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Execute only the seeded evolution method and collect validated outputs."""
    method_settings = replace(
        settings,
        method_sequence=("network_evolution_method",),
        design_method=HENDesignMethod.NetworkEvolution,
    )
    start = perf_counter()

    tasks, outcomes = execute_seeded_network_evolution_method_stage(
        problem=problem,
        settings=method_settings,
        seed_networks=seed_networks,
        executor=executor,
    )
    return SynthesisWorkflowResult(
        tasks=tasks,
        outcomes=outcomes,
        accepted_result=build_synthesis_result(method_settings, tasks, outcomes),
        total_run_time=perf_counter() - start,
    )


def execute_network_evolution_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    tdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one evolution stage from TDM parent outcomes."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_network_evolution_method_tasks(settings, tdm_outcomes)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes=parent_outcomes,
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def execute_seeded_network_evolution_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one standalone seeded evolution stage."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_seeded_network_evolution_method_tasks(settings, seed_networks)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def execute_network_evolution_method_from_pinch_design_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    executor: SynthesisExecutor | None = None,
):
    """Build and execute evolution directly from successful PDM outcomes."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_network_evolution_method_tasks_from_pinch_design_method(pdm_outcomes)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes=parent_outcomes,
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def execute_direct_network_evolution_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute evolution without PDM/TDM parent tasks."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_direct_network_evolution_method_tasks(settings)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def build_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
    tdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate one ESM refinement task for each successful TDM topology."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in tdm_outcomes:
        if not _successful_method(outcome, "thermal_derivative_method"):
            continue
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "network_evolution_method",
        )
        stage_count = _required_stage_count(outcome, "network_evolution_method")
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="network_evolution_method",
                approach_temperature=outcome.task.approach_temperature,
                derivative_threshold=outcome.task.derivative_threshold,
                stage_count=stage_count,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                state_id=settings.state_id,
                parent_task_id=outcome.task.task_id,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def build_seeded_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate standalone evolution tasks from existing seed-network topologies."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for seed_index, network in enumerate(seed_networks):
        restrictions = topology_restrictions_from_network(
            network,
            downstream_method="network_evolution_method",
        )
        stage_count = stage_count_from_network(
            network,
            downstream_method="network_evolution_method",
        )
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="network_evolution_method",
                approach_temperature=approach_temperature_from_network(
                    network,
                    settings,
                ),
                derivative_threshold=derivative_threshold_from_network(network),
                stage_count=stage_count,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                state_id=settings.state_id,
                seed_network_index=seed_index,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def build_network_evolution_method_tasks_from_pinch_design_method(
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate ESM tasks directly from PDM when TDM is unavailable."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in pdm_outcomes:
        if not _successful_method(outcome, "pinch_design_method"):
            continue
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "network_evolution_method",
        )
        stage_count = _required_stage_count(outcome, "network_evolution_method")
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=outcome.task.run_id,
                method="network_evolution_method",
                approach_temperature=outcome.task.approach_temperature,
                derivative_threshold=None,
                stage_count=stage_count,
                problem_id=outcome.task.problem_id,
                workspace_variant=outcome.task.workspace_variant,
                state_id=outcome.task.state_id,
                parent_task_id=outcome.task.task_id,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def build_direct_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate ESM tasks without Couenne-backed PDM/TDM parent tasks."""
    return tuple(
        HeatExchangerNetworkSynthesisTask(
            run_id=settings.run_id,
            method="network_evolution_method",
            approach_temperature=approach_temperature,
            derivative_threshold=None,
            stage_count=stage_count,
            problem_id=settings.problem_id,
            workspace_variant=settings.workspace_variant,
            state_id=settings.state_id,
        )
        for approach_temperature in settings.approach_temperatures
        for stage_count in settings.stage_selection
    )


__all__ = [
    "_execute_network_evolution_method_workflow",
    "build_direct_network_evolution_method_tasks",
    "build_network_evolution_method_tasks",
    "build_network_evolution_method_tasks_from_pinch_design_method",
    "build_seeded_network_evolution_method_tasks",
    "execute_direct_network_evolution_method_stage",
    "execute_network_evolution_method_from_pinch_design_stage",
    "execute_network_evolution_method_stage",
    "execute_seeded_network_evolution_method_stage",
]
