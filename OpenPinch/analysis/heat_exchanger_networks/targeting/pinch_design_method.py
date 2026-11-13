"""Pinch design method orchestration for HEN synthesis."""

from __future__ import annotations

from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.pathways import (
    TierPathway,
    pathway_metadata,
    tier_pathways,
)
from ..execution.settings import SynthesisWorkflowSettings
from ..results.assembly import SynthesisWorkflowResult
from ._stage import run_single_method_workflow, run_stage
from .heuristic_pinch_design_method import (
    execute_heuristic_pinch_design_method_stage,
)

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
    """Execute only the pinch design stages and collect validated outputs."""

    def run_stages(method_settings: SynthesisWorkflowSettings):
        milp_tasks, milp_outcomes, heuristic_tasks, heuristic_outcomes = (
            execute_pinch_design_stages(problem, method_settings, executor=executor)
        )
        return milp_tasks + heuristic_tasks, milp_outcomes + heuristic_outcomes

    return run_single_method_workflow(
        settings, HeatExchangerNetworkDesignMethod.PinchDesign, run_stages
    )


def execute_pinch_design_stages(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
):
    """Run the MILP stage when it is on, then the heuristic stage.

    Returns the MILP tasks and outcomes, then the heuristic ones; a stage that
    does not run gives empty tuples.
    """
    milp_tasks, milp_outcomes = (), ()
    if settings.runs_milp_pinch_design:
        milp_tasks, milp_outcomes = execute_pinch_design_method_stage(
            problem, settings, executor=executor
        )
    heuristic_tasks, heuristic_outcomes = (), ()
    if settings.heuristic_pdm_networks > 0:
        heuristic_tasks, heuristic_outcomes = (
            execute_heuristic_pinch_design_method_stage(
                problem, settings, executor=executor
            )
        )
    return milp_tasks, milp_outcomes, heuristic_tasks, heuristic_outcomes


def execute_pinch_design_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one pinch-design stage."""
    return run_stage(
        build_pinch_design_method_tasks(settings, problem=problem),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )


def build_pinch_design_method_tasks(
    settings: SynthesisWorkflowSettings,
    *,
    problem=None,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate root PDM tasks by sweeping the configured dt_cont multipliers."""
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
            for multiplier in settings.dt_cont_multipliers:
                tasks.append(
                    HeatExchangerNetworkSynthesisTask(
                        run_id=settings.run_id,
                        method=HeatExchangerNetworkDesignMethod.PinchDesign,
                        dt_cont_multiplier=multiplier,
                        problem_id=settings.problem_id,
                        workspace_variant=settings.workspace_variant,
                        period_id=settings.period_id,
                        metadata=pathway_metadata((pathway,)),
                    )
                )
            continue
        if pathway.multiplier is None:
            continue
        key = (pathway.pdm_mode, float(pathway.multiplier))
        grouped.setdefault(key, []).append(pathway)

    for (pdm_mode, multiplier), grouped_pathways in grouped.items():
        settings_data = {"pdm_mode": pdm_mode}
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method=HeatExchangerNetworkDesignMethod.PinchDesign,
                dt_cont_multiplier=multiplier,
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
            float(task.dt_cont_multiplier),
            tuple(int(value) for value in task.settings.get("stage_selection", ())),
        )
        for task in tasks
    }
    quality_tasks: list[HeatExchangerNetworkSynthesisTask] = []
    candidate_multipliers = settings.quality_pdm_dt_cont_multipliers
    for multiplier in candidate_multipliers:
        key = (float(multiplier), ())
        if key in seen:
            continue
        seen.add(key)
        quality_tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method=HeatExchangerNetworkDesignMethod.PinchDesign,
                dt_cont_multiplier=multiplier,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                metadata={"quality_candidate": "dt_cont_multiplier"},
            )
        )

    stage_pair_count = settings.quality_pdm_stage_pair_count
    if stage_pair_count <= 0:
        return (*tasks, *quality_tasks)

    stage_pair_multiplier = (
        candidate_multipliers[0]
        if candidate_multipliers
        else float(settings.dt_cont_multipliers[0])
    )
    for above_stages, below_stages in _PDM_STAGE_PAIR_SWEEP[:stage_pair_count]:
        key = (float(stage_pair_multiplier), (above_stages, below_stages))
        if key in seen:
            continue
        seen.add(key)
        quality_tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method=HeatExchangerNetworkDesignMethod.PinchDesign,
                dt_cont_multiplier=stage_pair_multiplier,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                period_id=settings.period_id,
                settings={"stage_selection": [above_stages, below_stages]},
                metadata={"quality_candidate": "stage_pair"},
            )
        )
    return (*tasks, *quality_tasks)


__all__ = [
    "_execute_pinch_design_method_workflow",
    "build_pinch_design_method_tasks",
    "execute_pinch_design_method_stage",
    "execute_pinch_design_stages",
]
