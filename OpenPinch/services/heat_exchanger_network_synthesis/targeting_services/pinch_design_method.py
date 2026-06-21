"""Pinch design method orchestration for HEN synthesis."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from ....lib.enums import HENDesignMethod
from ....lib.schemas.synthesis import HeatExchangerNetworkSynthesisTask
from ..common.execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..common.execution.settings import SynthesisWorkflowSettings
from ..common.results.assembly import SynthesisWorkflowResult, build_synthesis_result
from ..common.solver.pinch_design_snapshot import (
    PinchDecompositionSnapshot,
    PinchLocation,
    PinchTargetSnapshot,
    StageSelection,
    build_pinch_design_method_snapshot,
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
        method_sequence=("pinch_design_method",),
        design_method=HENDesignMethod.PinchDesign,
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

    tasks = build_pinch_design_method_tasks(settings)
    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    return tasks, outcomes


def build_pinch_design_method_tasks(
    settings: SynthesisWorkflowSettings,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate root PDM tasks by sweeping configured approach temperatures."""
    return tuple(
        HeatExchangerNetworkSynthesisTask(
            run_id=settings.run_id,
            method="pinch_design_method",
            approach_temperature=approach_temperature,
            problem_id=settings.problem_id,
            workspace_variant=settings.workspace_variant,
            state_id=settings.state_id,
        )
        for approach_temperature in settings.approach_temperatures
    )


__all__ = [
    "PinchDecompositionSnapshot",
    "PinchLocation",
    "PinchTargetSnapshot",
    "StageSelection",
    "_execute_pinch_design_method_workflow",
    "build_pinch_design_method_snapshot",
    "build_pinch_design_method_tasks",
    "execute_pinch_design_method_stage",
]
