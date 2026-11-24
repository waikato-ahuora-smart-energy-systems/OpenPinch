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
    dt_cont_multiplier_from_network,
    topology_restrictions_from_network,
)
from ..reporting.ranking import held_utility_levels
from ..results.assembly import (
    SynthesisWorkflowResult,
    build_synthesis_result,
    demote_infeasible_outcomes,
)
from ..results.record import admit
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
    if not getattr(executor, "admits_outcomes", False):
        # An executor that does not offer its outcomes as it solves them
        # has them offered here, once the stage is done.
        for outcome in outcomes:
            admit(outcome)
    # The executor can add outcomes for tasks it derived, such as the extra
    # networks of a pinch design task.
    task_ids = {task.task_id for task in tasks}
    return tasks + tuple(
        outcome.task for outcome in outcomes if outcome.task.task_id not in task_ids
    ), outcomes


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
    outcomes = demote_infeasible_outcomes(outcomes)
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

    Tasks are de-duplicated on dt_cont multiplier and canonical topology,
    plus the derivative threshold when ``distinct_thresholds`` is true.
    """

    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    seen: set[tuple] = set()
    for seed_index, network in enumerate(seed_networks):
        restrictions = topology_restrictions_from_network(
            network,
            downstream_method=method,
        )
        # A site seed holds each process's levels; its tasks hold them too.
        site_processes = network.source_metadata.get("site_processes")
        if site_processes is None:
            restrictions = canonical_topology_restrictions(restrictions)
            stage_count = canonical_stage_count(restrictions)
        else:
            restrictions, stage_count = _site_stages(network, restrictions)
        dt_cont_multiplier = dt_cont_multiplier_from_network(network, settings)
        signature = topology_restriction_signature(restrictions)
        if site_processes is not None:
            # Staged heaters and coolers sit between stages only, so an
            # empty stage at each end leaves room for a cooler before the
            # first recovery stage (steam raised from a hot stream's hottest
            # part) and a heater after the last.
            restrictions = tuple(
                restriction.model_copy(update={"stage": restriction.stage + 1})
                for restriction in restrictions
            )
            stage_count += 2
        task_settings = (
            {} if site_processes is None else {"site_processes": site_processes}
        )
        # A seed that records utility level targets passes them on, with the
        # part of them it holds: its tasks have no parent task to take
        # either from (``HENS_EVOLUTION_LEVEL_HOLDS``).
        level_targets = network.source_metadata.get("utility_level_targets")
        if level_targets and site_processes is None:
            task_settings = task_settings | {
                "utility_level_targets": level_targets,
                "held_utility_levels": held_utility_levels(network),
            }
        for derivative_threshold in derivative_thresholds(network):
            key = (
                (dt_cont_multiplier, derivative_threshold, signature)
                if distinct_thresholds
                else (dt_cont_multiplier, signature)
            ) + (repr(site_processes),)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method=method,
                    dt_cont_multiplier=dt_cont_multiplier,
                    derivative_threshold=derivative_threshold,
                    stage_count=stage_count,
                    problem_id=settings.problem_id,
                    workspace_variant=settings.workspace_variant,
                    period_id=settings.period_id,
                    seed_network_index=seed_index,
                    topology_restrictions=restrictions,
                    settings=task_settings,
                )
            )
    return tuple(tasks)


def _site_stages(network: HeatExchangerNetwork, restrictions) -> tuple[tuple, int]:
    """A site seed's canonical recovery restrictions and stage count.

    A stream runs at most one staged heater or cooler at a stage boundary,
    so a stream holding two levels between its recovery matches needs two
    boundaries there. The seed's staged units each keep a canonical stage
    of their own (in order with the matches), and are then dropped: the
    task places its staged units itself.
    """
    from ....contracts.synthesis.topology import (
        HeatExchangerNetworkTopologyRestriction,
    )
    from ....domain.enums import HeatExchangerKind

    staged = tuple(
        HeatExchangerNetworkTopologyRestriction(
            source_stream=exchanger.source_stream,
            sink_stream=exchanger.sink_stream,
            stage=exchanger.stage,
            duty=max(state.duty for state in exchanger.period_states),
        )
        for exchanger in network.exchangers
        if exchanger.kind is not HeatExchangerKind.RECOVERY
        and exchanger.stage is not None
        and any(state.active for state in exchanger.period_states)
    )
    canonical = canonical_topology_restrictions(restrictions + staged)
    matches = {(r.source_stream, r.sink_stream) for r in restrictions}
    return (
        tuple(r for r in canonical if (r.source_stream, r.sink_stream) in matches),
        canonical_stage_count(canonical),
    )
