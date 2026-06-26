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
    HeatExchangerNetworkTopologyRestriction,
)
from ..common.execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..common.execution.pathways import (
    TierPathway,
    pathway_metadata,
    pathways_from_metadata,
)
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
from .topology import (
    canonical_stage_count,
    canonical_topology_restrictions,
    topology_restriction_signature,
)


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

    tasks = build_network_evolution_method_tasks_from_pinch_design_method(
        pdm_outcomes,
        settings=settings,
    )
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
        pathways = pathways_from_metadata(outcome.task.metadata)
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "network_evolution_method",
        )
        stage_count = _required_stage_count(outcome, "network_evolution_method")
        if pathways:
            tasks.extend(
                _evolution_tasks_for_pathways(
                    settings=settings,
                    parent_outcome=outcome,
                    pathways=pathways,
                    approach_temperature=outcome.task.approach_temperature,
                    derivative_threshold=outcome.task.derivative_threshold,
                    stage_count=stage_count,
                    topology_restrictions=restrictions,
                )
            )
            continue
        tasks.append(
            _standard_evolution_task(
                settings=settings,
                parent_outcome=outcome,
                approach_temperature=outcome.task.approach_temperature,
                derivative_threshold=outcome.task.derivative_threshold,
                stage_count=stage_count,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def build_seeded_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate standalone evolution tasks from existing seed-network topologies."""
    if settings.synthesis_quality_tier > 1:
        return _build_seeded_quality_network_evolution_method_tasks(
            settings,
            seed_networks,
        )

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
                period_id=settings.period_id,
                seed_network_index=seed_index,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def build_network_evolution_method_tasks_from_pinch_design_method(
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    settings: SynthesisWorkflowSettings | None = None,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate ESM tasks directly from PDM when TDM is unavailable."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in pdm_outcomes:
        if not _successful_method(outcome, "pinch_design_method"):
            continue
        pathways = tuple(
            pathway
            for pathway in pathways_from_metadata(outcome.task.metadata)
            if not pathway.uses_tdm
        )
        if (
            settings is not None
            and settings.synthesis_quality_tier > 1
            and not pathways
        ):
            continue
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "network_evolution_method",
        )
        stage_count = _required_stage_count(outcome, "network_evolution_method")
        if pathways and settings is not None:
            tasks.extend(
                _evolution_tasks_for_pathways(
                    settings=settings,
                    parent_outcome=outcome,
                    pathways=pathways,
                    approach_temperature=outcome.task.approach_temperature,
                    derivative_threshold=None,
                    stage_count=stage_count,
                    topology_restrictions=restrictions,
                )
            )
            continue
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=outcome.task.run_id,
                method="network_evolution_method",
                approach_temperature=outcome.task.approach_temperature,
                derivative_threshold=None,
                stage_count=stage_count,
                problem_id=outcome.task.problem_id,
                workspace_variant=outcome.task.workspace_variant,
                period_id=outcome.task.period_id,
                parent_task_id=outcome.task.task_id,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def _standard_evolution_task(
    *,
    settings: SynthesisWorkflowSettings,
    parent_outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    approach_temperature: float,
    derivative_threshold: float | None,
    stage_count: int,
    topology_restrictions: tuple[HeatExchangerNetworkTopologyRestriction, ...],
) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        run_id=settings.run_id,
        method="network_evolution_method",
        approach_temperature=approach_temperature,
        derivative_threshold=derivative_threshold,
        stage_count=stage_count,
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        period_id=settings.period_id,
        parent_task_id=parent_outcome.task.task_id,
        topology_restrictions=topology_restrictions,
    )


def _evolution_tasks_for_pathways(
    *,
    settings: SynthesisWorkflowSettings,
    parent_outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    pathways: tuple[TierPathway, ...],
    approach_temperature: float,
    derivative_threshold: float | None,
    stage_count: int,
    topology_restrictions: tuple[HeatExchangerNetworkTopologyRestriction, ...],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    grouped: dict[tuple[int, int, int | None], list[TierPathway]] = {}
    for pathway in pathways:
        key = (
            pathway.evm_n_ad_branches,
            pathway.evm_n_rm_branches,
            pathway.evm_no_improvement_patience,
        )
        grouped.setdefault(key, []).append(pathway)

    tasks = []
    for (n_ad, n_rm, patience), grouped_pathways in grouped.items():
        task_settings = _evolution_task_settings(
            n_ad_branches=n_ad,
            n_rm_branches=n_rm,
            no_improvement_patience=patience,
        )
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="network_evolution_method",
                approach_temperature=approach_temperature,
                derivative_threshold=derivative_threshold,
                stage_count=stage_count,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                parent_task_id=parent_outcome.task.task_id,
                settings=task_settings,
                topology_restrictions=topology_restrictions,
                metadata=pathway_metadata(grouped_pathways),
            )
        )
    return tuple(tasks)


def _evolution_task_settings(
    *,
    n_ad_branches: int,
    n_rm_branches: int,
    no_improvement_patience: int | None,
) -> dict[str, int]:
    settings: dict[str, int] = {}
    if n_ad_branches != 1:
        settings["evolution_n_ad_branches"] = int(n_ad_branches)
    if n_rm_branches != 1:
        settings["evolution_n_rm_branches"] = int(n_rm_branches)
    if no_improvement_patience is not None:
        settings["evolution_no_improvement_patience"] = int(no_improvement_patience)
    return settings


def _build_seeded_quality_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    seen: set[tuple[float, tuple[tuple[str, str, int], ...]]] = set()
    for seed_index, network in enumerate(seed_networks):
        restrictions = canonical_topology_restrictions(
            topology_restrictions_from_network(
                network,
                downstream_method="network_evolution_method",
            )
        )
        approach_temperature = approach_temperature_from_network(network, settings)
        signature = topology_restriction_signature(restrictions)
        key = (approach_temperature, signature)
        if key in seen:
            continue
        seen.add(key)
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="network_evolution_method",
                approach_temperature=approach_temperature,
                derivative_threshold=derivative_threshold_from_network(network),
                stage_count=canonical_stage_count(restrictions),
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                seed_network_index=seed_index,
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
            period_id=settings.period_id,
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
