"""Pinch design method orchestration for HEN synthesis."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.pathways import (
    TierPathway,
    pathway_metadata,
    tier_pathways,
)
from ..execution.settings import SynthesisWorkflowSettings
from ..results.assembly import SynthesisWorkflowResult, build_synthesis_result

_PDM_STAGE_PAIR_SWEEP = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)


def _execute_pinch_design_method_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Execute only the PDM method and collect validated method outputs."""
    method_settings = replace(
        settings,
        method_sequence=(HeatExchangerNetworkDesignMethod.PinchDesign,),
        design_method=HeatExchangerNetworkDesignMethod.PinchDesign,
    )
    start = perf_counter()
    tasks, outcomes = execute_pinch_design_method_stage(
        problem,
        method_settings,
        executor=executor,
    )
    return SynthesisWorkflowResult(
        tasks=tasks,
        outcomes=outcomes,
        accepted_result=build_synthesis_result(method_settings, tasks, outcomes),
        total_run_time=perf_counter() - start,
    )


def execute_pinch_design_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one pinch-design stage."""
    if executor is None:
        executor = LocalSynthesisExecutor()

    tasks = build_pinch_design_method_tasks(settings, problem=problem)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def build_pinch_design_method_tasks(
    settings: SynthesisWorkflowSettings,
    *,
    problem=None,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate root PDM tasks by sweeping configured approach temperatures."""
    del problem
    return _build_pathway_pinch_design_method_tasks(settings)


def _build_pathway_pinch_design_method_tasks(
    settings: SynthesisWorkflowSettings,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    pathways = tier_pathways(settings)
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    grouped: dict[tuple[str, float], list[TierPathway]] = {}

    for pathway in pathways:
        if pathway.exact_open_hens:
            for approach_temperature in settings.approach_temperatures:
                tasks.append(
                    HeatExchangerNetworkSynthesisTask(
                        run_id=settings.run_id,
                        method="pinch_design_method",
                        approach_temperature=approach_temperature,
                        problem_id=settings.problem_id,
                        workspace_variant=settings.workspace_variant,
                        period_id=settings.period_id,
                        metadata=pathway_metadata((pathway,)),
                    )
                )
            continue
        if pathway.multiplier is None:
            continue
        approach_temperature = _pathway_approach_temperature(settings, pathway)
        key = (pathway.pdm_mode, approach_temperature)
        grouped.setdefault(key, []).append(pathway)

    for (pdm_mode, approach_temperature), grouped_pathways in grouped.items():
        settings_data = {"pdm_mode": pdm_mode}
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="pinch_design_method",
                approach_temperature=approach_temperature,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                settings=settings_data,
                metadata=pathway_metadata(grouped_pathways),
            )
        )

    if settings.quality_pdm_stage_pair_count <= 0:
        return tuple(tasks)
    return _with_quality_pdm_tasks(settings, tuple(tasks))


def _with_quality_pdm_tasks(
    settings: SynthesisWorkflowSettings,
    tasks: tuple[HeatExchangerNetworkSynthesisTask, ...],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    seen = {
        (
            float(task.approach_temperature),
            tuple(int(value) for value in task.settings.get("stage_selection", ())),
        )
        for task in tasks
    }
    quality_tasks: list[HeatExchangerNetworkSynthesisTask] = []
    candidate_temperatures = settings.quality_pdm_approach_temperatures
    for approach_temperature in candidate_temperatures:
        key = (float(approach_temperature), ())
        if key in seen:
            continue
        seen.add(key)
        quality_tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="pinch_design_method",
                approach_temperature=approach_temperature,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                metadata={"quality_candidate": "dt_cont_multiplier"},
            )
        )

    stage_pair_count = settings.quality_pdm_stage_pair_count
    if stage_pair_count <= 0:
        return (*tasks, *quality_tasks)

    stage_pair_temperature = (
        candidate_temperatures[0]
        if candidate_temperatures
        else float(settings.approach_temperatures[0])
    )
    for above_stages, below_stages in _PDM_STAGE_PAIR_SWEEP[:stage_pair_count]:
        key = (float(stage_pair_temperature), (above_stages, below_stages))
        if key in seen:
            continue
        seen.add(key)
        quality_tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="pinch_design_method",
                approach_temperature=stage_pair_temperature,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                settings={"stage_selection": [above_stages, below_stages]},
                metadata={"quality_candidate": "stage_pair"},
            )
        )
    return (*tasks, *quality_tasks)


def _pathway_approach_temperature(
    settings: SynthesisWorkflowSettings,
    pathway: TierPathway,
) -> float:
    """Return the concrete dTmin represented by one PDM multiplier pathway."""

    if pathway.multiplier is None:
        raise ValueError("PDM multiplier pathway requires a multiplier.")
    return float(settings.approach_temperatures[0]) * float(pathway.multiplier)


__all__ = [
    "_execute_pinch_design_method_workflow",
    "build_pinch_design_method_tasks",
    "execute_pinch_design_method_stage",
]
