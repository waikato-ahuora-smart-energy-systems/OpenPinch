"""Shared stage runner and seeded-task builder for HEN method orchestration."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable, Sequence

from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..execution.executor import SynthesisExecutor
from ..execution.settings import SynthesisWorkflowSettings
from ..execution.task_builders import (
    approach_temperature_from_network,
    topology_restrictions_from_network,
)
from ..results.assembly import SynthesisWorkflowResult, build_synthesis_result
from .topology import (
    canonical_stage_count,
    canonical_topology_restrictions,
    topology_restriction_signature,
)

StageResult = tuple[
    tuple[HeatExchangerNetworkSynthesisTask, ...],
    tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...],
]


def run_stage(
    tasks: tuple[HeatExchangerNetworkSynthesisTask, ...],
    *,
    problem,
    settings: SynthesisWorkflowSettings,
    executor: SynthesisExecutor | None,
    default_executor: Callable[[], SynthesisExecutor],
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome] | None = None,
) -> StageResult:
    """Execute one built task stage, creating the default executor if needed.

    ``default_executor`` is passed by each method module (its own
    ``LocalSynthesisExecutor`` name) so the default stays patchable per module.
    """

    if executor is None:
        executor = default_executor()
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={} if parent_outcomes is None else parent_outcomes,
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def run_single_method_workflow(
    settings: SynthesisWorkflowSettings,
    method: HeatExchangerNetworkDesignMethod,
    run_method_stage: Callable[[SynthesisWorkflowSettings], StageResult],
) -> SynthesisWorkflowResult:
    """Run one method stage with method-only settings and assemble the result."""

    method_settings = replace(
        settings,
        method_sequence=(method,),
        design_method=method,
    )
    start = perf_counter()
    tasks, outcomes = run_method_stage(method_settings)
    return SynthesisWorkflowResult(
        tasks=tasks,
        outcomes=outcomes,
        accepted_result=build_synthesis_result(method_settings, tasks, outcomes),
        total_run_time=perf_counter() - start,
    )


def build_seeded_quality_tasks(
    settings: SynthesisWorkflowSettings,
    seed_networks: Sequence[HeatExchangerNetwork],
    *,
    method: HeatExchangerNetworkDesignMethod,
    derivative_thresholds: Callable[[HeatExchangerNetwork], Sequence[float | None]],
    distinct_thresholds: bool,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Build canonical, de-duplicated seeded tasks for quality tiers above 1.

    Tasks are de-duplicated on approach temperature and canonical topology,
    plus the derivative threshold when ``distinct_thresholds`` is true.
    """

    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    seen: set[tuple] = set()
    for seed_index, network in enumerate(seed_networks):
        restrictions = canonical_topology_restrictions(
            topology_restrictions_from_network(
                network,
                downstream_method=method,
            )
        )
        approach_temperature = approach_temperature_from_network(network, settings)
        signature = topology_restriction_signature(restrictions)
        for derivative_threshold in derivative_thresholds(network):
            key = (
                (approach_temperature, derivative_threshold, signature)
                if distinct_thresholds
                else (approach_temperature, signature)
            )
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method=method,
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
