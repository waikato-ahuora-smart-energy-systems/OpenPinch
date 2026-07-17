"""OpenHENS workflow orchestration for HEN synthesis."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from ....domain.enums import HENDesignMethod
from ..errors import WorkflowContractError
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.fallbacks import (
    _can_skip_derivative_stage_for_missing_couenne,
    _can_skip_preliminary_stages_for_missing_couenne,
    _warn_couenne_fallback,
)
from ..execution.settings import SynthesisWorkflowSettings
from ..execution.task_builders import _outcome_map
from ..results.assembly import SynthesisWorkflowResult, build_synthesis_result
from .network_evolution_method import (
    execute_direct_network_evolution_method_stage,
    execute_network_evolution_method_from_pinch_design_stage,
    execute_network_evolution_method_stage,
)
from .pinch_design_method import execute_pinch_design_method_stage
from .thermal_derivative_method import execute_thermal_derivative_method_stage

_METHOD_SEQUENCE = (
    "pinch_design_method",
    "thermal_derivative_method",
    "network_evolution_method",
)


def execute_open_hens_method(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Generate, execute, and collect the PDM -> TDM -> EVM task graph."""
    settings = replace(settings, design_method=HENDesignMethod.OpenHENS)
    if settings.method_sequence != _METHOD_SEQUENCE:
        raise WorkflowContractError(
            "HENS_METHOD_SEQUENCE must preserve pinch_design_method -> "
            "thermal_derivative_method -> network_evolution_method for "
            "this migration slice."
        )

    if executor is None:
        executor = LocalSynthesisExecutor()
    start = perf_counter()

    pdm_tasks, pdm_outcomes = execute_pinch_design_method_stage(
        problem=problem,
        settings=settings,
        executor=executor,
    )

    pdm_outcome_map = _outcome_map(pdm_outcomes)
    if _can_skip_preliminary_stages_for_missing_couenne(
        settings,
        pdm_tasks,
        pdm_outcomes,
    ):
        _warn_couenne_fallback(
            "Couenne is unavailable for the heat exchanger network "
            "pinch-design-method stage",
        )
        tdm_tasks = ()
        tdm_outcomes = ()
        esm_tasks, esm_outcomes = execute_direct_network_evolution_method_stage(
            problem=problem,
            settings=settings,
            executor=executor,
        )
    elif settings.skips_thermal_derivative_method:
        tdm_tasks = ()
        tdm_outcomes = ()
        esm_tasks, esm_outcomes = (
            execute_network_evolution_method_from_pinch_design_stage(
                problem=problem,
                settings=settings,
                pdm_outcomes=pdm_outcomes,
                parent_outcomes=pdm_outcome_map,
                executor=executor,
            )
        )
    else:
        direct_esm_tasks = ()
        direct_esm_outcomes = ()
        if settings.synthesis_quality_tier > 1:
            direct_esm_tasks, direct_esm_outcomes = (
                execute_network_evolution_method_from_pinch_design_stage(
                    problem=problem,
                    settings=settings,
                    pdm_outcomes=pdm_outcomes,
                    parent_outcomes=pdm_outcome_map,
                    executor=executor,
                )
            )

        tdm_tasks, tdm_outcomes = execute_thermal_derivative_method_stage(
            problem=problem,
            settings=settings,
            pdm_outcomes=pdm_outcomes,
            parent_outcomes=pdm_outcome_map,
            executor=executor,
        )

        upstream_outcomes = pdm_outcome_map | _outcome_map(tdm_outcomes)
        esm_tasks, esm_outcomes = execute_network_evolution_method_stage(
            problem=problem,
            settings=settings,
            tdm_outcomes=tdm_outcomes,
            parent_outcomes=upstream_outcomes,
            executor=executor,
        )
        esm_tasks = tuple(direct_esm_tasks + esm_tasks)
        esm_outcomes = tuple(direct_esm_outcomes + esm_outcomes)
    if not esm_tasks and _can_skip_derivative_stage_for_missing_couenne(
        settings,
        tdm_tasks,
        tdm_outcomes,
    ):
        _warn_couenne_fallback(
            "Couenne is unavailable for the heat exchanger network "
            "thermal-derivative-method topology stage",
        )
        esm_tasks, esm_outcomes = (
            execute_network_evolution_method_from_pinch_design_stage(
                problem=problem,
                settings=settings,
                pdm_outcomes=pdm_outcomes,
                parent_outcomes=pdm_outcome_map,
                executor=executor,
            )
        )

    tasks = tuple(pdm_tasks + tdm_tasks + esm_tasks)
    outcomes = tuple(pdm_outcomes + tdm_outcomes + esm_outcomes)
    return SynthesisWorkflowResult(
        tasks=tasks,
        outcomes=outcomes,
        accepted_result=build_synthesis_result(settings, tasks, outcomes),
        total_run_time=perf_counter() - start,
    )


__all__ = ["execute_open_hens_method"]
