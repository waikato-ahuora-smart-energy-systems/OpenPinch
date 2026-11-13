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
from ..execution.settings import SynthesisWorkflowSettings
from ..execution.task_builders import (
    _required_stage_count,
    _successful_method,
    is_pdm_variant_outcome,
    is_utility_only_outcome,
    required_topology_restrictions_from_outcome,
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
        if not _successful_method(
            outcome, HeatExchangerNetworkDesignMethod.PinchDesign
        ):
            continue
        if is_utility_only_outcome(outcome):
            continue  # utility-only: a final candidate, nothing to refine
        if is_pdm_variant_outcome(outcome):
            continue  # extra PDM networks seed evolution directly
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            HeatExchangerNetworkDesignMethod.ThermalDerivative,
        )
        stage_count = _required_stage_count(
            outcome, HeatExchangerNetworkDesignMethod.ThermalDerivative
        )
        for derivative_threshold in settings.derivative_thresholds:
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method=HeatExchangerNetworkDesignMethod.ThermalDerivative,
                    dt_cont_multiplier=outcome.task.dt_cont_multiplier,
                    derivative_threshold=derivative_threshold,
                    stage_count=stage_count,
                    problem_id=settings.problem_id,
                    workspace_variant=settings.workspace_variant,
                    period_id=settings.period_id,
                    parent_task_id=outcome.task.task_id,
                    topology_restrictions=restrictions,
                )
            )
    return tuple(tasks)


def build_seeded_thermal_derivative_method_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Standalone TDM tasks from seed topologies, one per distinct threshold."""
    return build_seeded_quality_tasks(
        settings,
        seed_networks,
        method=HeatExchangerNetworkDesignMethod.ThermalDerivative,
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
