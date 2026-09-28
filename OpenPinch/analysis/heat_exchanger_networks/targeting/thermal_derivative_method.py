"""Thermal derivative method orchestration for HEN synthesis."""

from __future__ import annotations

from typing import Sequence

from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.pathways import pathway_metadata, pathways_from_metadata
from ..execution.settings import SynthesisWorkflowSettings
from ..execution.task_builders import (
    _required_stage_count,
    _successful_method,
    approach_temperature_from_network,
    required_topology_restrictions_from_outcome,
    stage_count_from_network,
    topology_restrictions_from_network,
)
from ..results.assembly import SynthesisWorkflowResult
from ._stage import build_seeded_quality_tasks, run_single_method_workflow, run_stage


def _execute_thermal_derivative_method_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Execute only the seeded TDM method and collect validated method outputs."""
    return run_single_method_workflow(
        settings,
        HeatExchangerNetworkDesignMethod.ThermalDerivative,
        lambda method_settings: execute_seeded_thermal_derivative_method_stage(
            problem=problem,
            settings=method_settings,
            seed_networks=seed_networks,
            executor=executor,
        ),
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
    return run_stage(
        build_thermal_derivative_method_tasks(settings, pdm_outcomes),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
        parent_outcomes=parent_outcomes,
    )


def execute_seeded_thermal_derivative_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one standalone seeded TDM stage."""
    return run_stage(
        build_seeded_thermal_derivative_method_tasks(settings, seed_networks),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )


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
    return build_seeded_quality_tasks(
        settings,
        seed_networks,
        method="thermal_derivative_method",
        derivative_thresholds=lambda _network: settings.quality_derivative_thresholds,
        distinct_thresholds=True,
    )


__all__ = [
    "_execute_thermal_derivative_method_workflow",
    "build_seeded_thermal_derivative_method_tasks",
    "build_thermal_derivative_method_tasks",
    "execute_seeded_thermal_derivative_method_stage",
    "execute_thermal_derivative_method_stage",
]
