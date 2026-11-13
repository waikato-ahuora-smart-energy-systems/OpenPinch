"""Network evolution method orchestration for HEN synthesis."""

from __future__ import annotations

from typing import Sequence

from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....contracts.synthesis.topology import HeatExchangerNetworkTopologyRestriction
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.settings import SynthesisWorkflowSettings
from ..execution.task_builders import (
    _required_stage_count,
    _successful_method,
    derivative_threshold_from_network,
    is_utility_only_outcome,
    required_topology_restrictions_from_outcome,
)
from ..results.assembly import SynthesisWorkflowResult
from ._stage import build_seeded_quality_tasks, run_single_method_workflow, run_stage


def _execute_network_evolution_method_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Execute only the seeded evolution method and collect validated outputs."""
    return run_single_method_workflow(
        settings,
        HeatExchangerNetworkDesignMethod.NetworkEvolution,
        lambda method_settings: execute_seeded_network_evolution_method_stage(
            problem=problem,
            settings=method_settings,
            seed_networks=seed_networks,
            executor=executor,
        ),
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
    return run_stage(
        build_network_evolution_method_tasks(settings, tdm_outcomes),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
        parent_outcomes=parent_outcomes,
    )


def execute_seeded_network_evolution_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one standalone seeded evolution stage."""
    return run_stage(
        build_seeded_network_evolution_method_tasks(settings, seed_networks),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )


def execute_network_evolution_method_from_pinch_design_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    executor: SynthesisExecutor | None = None,
):
    """Build and execute evolution directly from successful PDM outcomes."""
    return run_stage(
        build_network_evolution_method_tasks_from_pinch_design_method(
            pdm_outcomes,
            settings=settings,
        ),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
        parent_outcomes=parent_outcomes,
    )


def execute_direct_network_evolution_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute evolution without PDM/TDM parent tasks."""
    return run_stage(
        build_direct_network_evolution_method_tasks(settings),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )


def build_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
    tdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate one ESM refinement task for each successful TDM topology."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in tdm_outcomes:
        if not _successful_method(
            outcome, HeatExchangerNetworkDesignMethod.ThermalDerivative
        ):
            continue
        if is_utility_only_outcome(outcome):
            continue  # utility-only: a final candidate, nothing to refine
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            HeatExchangerNetworkDesignMethod.NetworkEvolution,
        )
        stage_count = _required_stage_count(
            outcome, HeatExchangerNetworkDesignMethod.NetworkEvolution
        )
        tasks.append(
            _standard_evolution_task(
                settings=settings,
                parent_outcome=outcome,
                dt_cont_multiplier=outcome.task.dt_cont_multiplier,
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
    """Standalone evolution tasks, one per distinct seed topology."""
    return build_seeded_quality_tasks(
        settings,
        seed_networks,
        method=HeatExchangerNetworkDesignMethod.NetworkEvolution,
        derivative_thresholds=lambda network: (
            derivative_threshold_from_network(network),
        ),
        distinct_thresholds=False,
    )


def build_network_evolution_method_tasks_from_pinch_design_method(
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    settings: SynthesisWorkflowSettings | None = None,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate ESM tasks directly from PDM or heuristic pinch design outcomes."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in pdm_outcomes:
        heuristic = _successful_method(
            outcome, HeatExchangerNetworkDesignMethod.HeuristicPinchDesign
        )
        if not heuristic and not _successful_method(
            outcome, HeatExchangerNetworkDesignMethod.PinchDesign
        ):
            continue
        if is_utility_only_outcome(outcome):
            continue  # utility-only: a final candidate, nothing to refine
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            HeatExchangerNetworkDesignMethod.NetworkEvolution,
        )
        stage_count = _required_stage_count(
            outcome, HeatExchangerNetworkDesignMethod.NetworkEvolution
        )
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=outcome.task.run_id,
                method=HeatExchangerNetworkDesignMethod.NetworkEvolution,
                dt_cont_multiplier=outcome.task.dt_cont_multiplier,
                derivative_threshold=None,
                stage_count=stage_count,
                problem_id=outcome.task.problem_id,
                workspace_variant=outcome.task.workspace_variant,
                period_id=outcome.task.period_id,
                parent_task_id=outcome.task.task_id,
                settings={} if settings is None else _run_breadth_settings(settings),
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def _standard_evolution_task(
    *,
    settings: SynthesisWorkflowSettings,
    parent_outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    dt_cont_multiplier: float,
    derivative_threshold: float | None,
    stage_count: int,
    topology_restrictions: tuple[HeatExchangerNetworkTopologyRestriction, ...],
) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        run_id=settings.run_id,
        method=HeatExchangerNetworkDesignMethod.NetworkEvolution,
        dt_cont_multiplier=dt_cont_multiplier,
        derivative_threshold=derivative_threshold,
        stage_count=stage_count,
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        period_id=settings.period_id,
        parent_task_id=parent_outcome.task.task_id,
        settings=_run_breadth_settings(settings),
        topology_restrictions=topology_restrictions,
    )


def _run_breadth_settings(settings: SynthesisWorkflowSettings) -> dict[str, int]:
    """Branch breadth of this run; wider searches stop after two idle rounds."""
    n_ad = settings.effective_evm_n_ad_branches
    n_rm = settings.effective_evm_n_rm_branches
    breadth = {"evolution_n_ad_branches": n_ad, "evolution_n_rm_branches": n_rm}
    if max(n_ad, n_rm) > 1:
        breadth["evolution_no_improvement_patience"] = 2
    return breadth


def build_direct_network_evolution_method_tasks(
    settings: SynthesisWorkflowSettings,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate ESM tasks without Couenne-backed PDM/TDM parent tasks."""
    return tuple(
        HeatExchangerNetworkSynthesisTask(
            run_id=settings.run_id,
            method=HeatExchangerNetworkDesignMethod.NetworkEvolution,
            dt_cont_multiplier=dt_cont_multiplier,
            derivative_threshold=None,
            stage_count=stage_count,
            problem_id=settings.problem_id,
            workspace_variant=settings.workspace_variant,
            period_id=settings.period_id,
            settings=_run_breadth_settings(settings),
        )
        for dt_cont_multiplier in settings.dt_cont_multipliers
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
