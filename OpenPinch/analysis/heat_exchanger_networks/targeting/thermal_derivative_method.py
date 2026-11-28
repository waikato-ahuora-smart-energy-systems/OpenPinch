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
    """Build and execute one standalone seeded TDM stage.

    The step minimises hot utility, not total annual cost, so each seed is
    ranked alongside its networks: under the usual ranking (networks that
    hold their recorded utility levels first, then the cheapest) what the
    step returns is never dearer than what it started from (review finding
    29).
    """
    tasks, outcomes = run_stage(
        build_seeded_thermal_derivative_method_tasks(settings, seed_networks),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )
    return tasks, (*outcomes, *_seed_outcomes(tasks, seed_networks))


def _seed_outcomes(
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    seed_networks: Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
    """Each priced seed as an outcome of the task run for it.

    A seed whose task was dropped as a duplicate of an earlier seed's takes
    the first task run. The seed takes its task's run and task ids, as a
    result's networks do; its method stays its own.
    """
    if not tasks:
        return ()
    run_for = {task.seed_network_index: task for task in reversed(tasks)}
    outcomes = []
    for index, network in enumerate(seed_networks):
        cost = network.total_annual_cost
        if cost is None:
            continue
        task = run_for.get(index, tasks[0])
        outcomes.append(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=task,
                status="success",
                network=network.model_copy(
                    update={"run_id": task.run_id, "task_id": task.task_id}
                ),
                objective_value=float(cost),
                solver_status="seed",
            )
        )
    return tuple(outcomes)


def build_thermal_derivative_method_tasks(
    settings: SynthesisWorkflowSettings,
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Fan successful PDM topologies out over derivative thresholds.

    Each distinct threshold gives one task per parent. With
    ``tdm_parent_limit`` set, only that many parents are refined, the
    cheapest by objective value first (one without a value last).
    """
    parents = [
        outcome
        for outcome in pdm_outcomes
        if _successful_method(outcome, HeatExchangerNetworkDesignMethod.PinchDesign)
        # utility-only: a final candidate, nothing to refine; extra PDM
        # networks seed evolution directly
        and not is_utility_only_outcome(outcome)
        and not is_pdm_variant_outcome(outcome)
    ]
    if settings.tdm_parent_limit is not None:
        parents = sorted(parents, key=_objective_or_inf)[
            : max(1, int(settings.tdm_parent_limit))
        ]
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in parents:
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            HeatExchangerNetworkDesignMethod.ThermalDerivative,
        )
        stage_count = _required_stage_count(
            outcome, HeatExchangerNetworkDesignMethod.ThermalDerivative
        )
        for derivative_threshold in settings.quality_derivative_thresholds:
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


def _objective_or_inf(outcome: HeatExchangerNetworkSynthesisTaskOutcome) -> float:
    value = outcome.objective_value
    return float("inf") if value is None else float(value)


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
