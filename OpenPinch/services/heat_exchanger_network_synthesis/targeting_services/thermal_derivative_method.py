"""Thermal derivative method orchestration for HEN synthesis."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Sequence

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from ....lib.enums import HENDesignMethod
from ....lib.schemas.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ..common.execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..common.execution.pathways import pathway_metadata, pathways_from_metadata
from ..common.execution.settings import SynthesisWorkflowSettings
from ..common.execution.task_builders import (
    _required_stage_count,
    _successful_method,
    approach_temperature_from_network,
    required_topology_restrictions_from_outcome,
    stage_count_from_network,
    topology_restrictions_from_network,
)
from ..common.results.assembly import SynthesisWorkflowResult, build_synthesis_result
from .topology import (
    canonical_stage_count,
    canonical_topology_restrictions,
    topology_restriction_signature,
)


def _execute_thermal_derivative_method_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Execute only the seeded TDM method and collect validated method outputs."""
    method_settings = replace(
        settings,
        method_sequence=("thermal_derivative_method",),
        design_method=HENDesignMethod.ThermalDerivative,
    )
    start = perf_counter()

    tasks, outcomes = execute_seeded_thermal_derivative_method_stage(
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


def execute_thermal_derivative_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one TDM stage from PDM parent outcomes."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_thermal_derivative_method_tasks(settings, pdm_outcomes)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes=parent_outcomes,
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def execute_seeded_thermal_derivative_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one standalone seeded TDM stage."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_seeded_thermal_derivative_method_tasks(settings, seed_networks)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def build_thermal_derivative_method_tasks(
    settings: SynthesisWorkflowSettings,
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Fan successful PDM topologies out over derivative thresholds."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in pdm_outcomes:
        if not _successful_method(outcome, "pinch_design_method"):
            continue
        pathways = pathways_from_metadata(outcome.task.metadata)
        tdm_pathways = tuple(pathway for pathway in pathways if pathway.uses_tdm)
        if pathways and not tdm_pathways:
            continue
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "thermal_derivative_method",
        )
        stage_count = _required_stage_count(outcome, "thermal_derivative_method")
        for derivative_threshold in settings.derivative_thresholds:
            metadata = pathway_metadata(tdm_pathways)
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method="thermal_derivative_method",
                    approach_temperature=outcome.task.approach_temperature,
                    derivative_threshold=derivative_threshold,
                    stage_count=stage_count,
                    problem_id=settings.problem_id,
                    workspace_variant=settings.workspace_variant,
                    period_id=settings.period_id,
                    parent_task_id=outcome.task.task_id,
                    topology_restrictions=restrictions,
                    metadata=metadata,
                )
            )
    return tuple(tasks)


def build_seeded_thermal_derivative_method_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate standalone TDM tasks from existing seed-network topologies."""
    if settings.synthesis_quality_tier > 1:
        return _build_seeded_quality_thermal_derivative_method_tasks(
            settings,
            seed_networks,
        )

    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for seed_index, network in enumerate(seed_networks):
        restrictions = topology_restrictions_from_network(
            network,
            downstream_method="thermal_derivative_method",
        )
        stage_count = stage_count_from_network(
            network,
            downstream_method="thermal_derivative_method",
        )
        approach_temperature = approach_temperature_from_network(network, settings)
        for derivative_threshold in settings.derivative_thresholds:
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method="thermal_derivative_method",
                    approach_temperature=approach_temperature,
                    derivative_threshold=derivative_threshold,
                    stage_count=stage_count,
                    problem_id=settings.problem_id,
                    workspace_variant=settings.workspace_variant,
                    period_id=settings.period_id,
                    seed_network_index=seed_index,
                    topology_restrictions=restrictions,
                )
            )
    return tuple(tasks)


def _build_seeded_quality_thermal_derivative_method_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    seen: set[tuple[float, float, tuple[tuple[str, str, int], ...]]] = set()
    for seed_index, network in enumerate(seed_networks):
        restrictions = canonical_topology_restrictions(
            topology_restrictions_from_network(
                network,
                downstream_method="thermal_derivative_method",
            )
        )
        approach_temperature = approach_temperature_from_network(network, settings)
        signature = topology_restriction_signature(restrictions)
        for derivative_threshold in settings.quality_derivative_thresholds:
            key = (approach_temperature, derivative_threshold, signature)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method="thermal_derivative_method",
                    approach_temperature=approach_temperature,
                    derivative_threshold=derivative_threshold,
                    stage_count=canonical_stage_count(restrictions),
                    problem_id=settings.problem_id,
                    workspace_variant=settings.workspace_variant,
                    period_id=settings.period_id,
                    seed_network_index=seed_index,
                    topology_restrictions=restrictions,
                )
            )
    return tuple(tasks)


__all__ = [
    "_execute_thermal_derivative_method_workflow",
    "build_seeded_thermal_derivative_method_tasks",
    "build_thermal_derivative_method_tasks",
    "execute_seeded_thermal_derivative_method_stage",
    "execute_thermal_derivative_method_stage",
]
