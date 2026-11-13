"""Pinch design method orchestration for HEN synthesis."""

from __future__ import annotations

from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
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
    """One MILP pinch design task per dt_cont multiplier.

    ``HENS_PDM_STAGE_PAIR_LIMIT`` adds candidates with set stage counts
    above and below the pinch, at the first multiplier.
    """
    del problem
    tasks = [
        _pinch_design_task(settings, multiplier)
        for multiplier in dict.fromkeys(settings.dt_cont_multipliers)
    ]
    multiplier = float(settings.dt_cont_multipliers[0])
    for above_stages, below_stages in _PDM_STAGE_PAIR_SWEEP[
        : settings.quality_pdm_stage_pair_count
    ]:
        tasks.append(
            _pinch_design_task(
                settings,
                multiplier,
                settings={"stage_selection": [above_stages, below_stages]},
                metadata={"quality_candidate": "stage_pair"},
            )
        )
    return tuple(tasks)


def _pinch_design_task(
    workflow: SynthesisWorkflowSettings,
    multiplier: float,
    **fields,
) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        run_id=workflow.run_id,
        method=HeatExchangerNetworkDesignMethod.PinchDesign,
        dt_cont_multiplier=multiplier,
        problem_id=workflow.problem_id,
        workspace_variant=workflow.workspace_variant,
        period_id=workflow.period_id,
        **fields,
    )


__all__ = [
    "_execute_pinch_design_method_workflow",
    "build_pinch_design_method_tasks",
    "execute_pinch_design_method_stage",
    "execute_pinch_design_stages",
]
