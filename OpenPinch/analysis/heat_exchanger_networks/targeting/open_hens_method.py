"""OpenHENS workflow orchestration for HEN synthesis."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.fallbacks import (
    _can_skip_derivative_stage_for_missing_couenne,
    _can_skip_preliminary_stages_for_missing_couenne,
    _warn_couenne_fallback,
)
from ..execution.settings import SynthesisWorkflowSettings
from ..execution.task_builders import _outcome_map, is_pdm_variant_outcome
from ..results.assembly import (
    SynthesisWorkflowResult,
    build_synthesis_result,
    demote_infeasible_outcomes,
)
from .network_evolution_method import (
    execute_direct_network_evolution_method_stage,
    execute_network_evolution_method_from_pinch_design_stage,
    execute_network_evolution_method_stage,
)
from .pinch_design_method import execute_pinch_design_stages
from .thermal_derivative_method import execute_thermal_derivative_method_stage


def execute_open_hens_method(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Generate, execute, and collect the synthesis task graph.

    The search depth sets the stages: pinch design (heuristic, and MILP when
    on), then from depth 1 network evolution of the heuristic networks, and
    from depth 2 the thermal derivative method on the main MILP networks
    before their evolution. Extra minimum-unit MILP networks, and from depth
    3 every MILP network, seed evolution directly.
    """
    settings = replace(
        settings, design_method=HeatExchangerNetworkDesignMethod.OpenHENS
    )
    if executor is None:
        executor = LocalSynthesisExecutor()
    start = perf_counter()

    pdm_tasks, pdm_outcomes, heuristic_tasks, heuristic_outcomes = (
        execute_pinch_design_stages(problem, settings, executor=executor)
    )
    pdm_outcome_map = _outcome_map(pdm_outcomes) | _outcome_map(heuristic_outcomes)
    tdm_tasks, tdm_outcomes, esm_tasks, esm_outcomes = (), (), (), ()

    def evolve_from_pinch_design(seeds):
        return execute_network_evolution_method_from_pinch_design_stage(
            problem=problem,
            settings=settings,
            pdm_outcomes=seeds,
            parent_outcomes=pdm_outcome_map,
            executor=executor,
        )

    if not settings.runs_evolution:
        pass
    elif _can_skip_preliminary_stages_for_missing_couenne(
        settings,
        pdm_tasks,
        pdm_outcomes,
    ):
        _warn_couenne_fallback(
            "Couenne is unavailable for the heat exchanger network "
            "pinch-design-method stage",
        )
        esm_tasks, esm_outcomes = execute_direct_network_evolution_method_stage(
            problem=problem,
            settings=settings,
            executor=executor,
        )
        if heuristic_outcomes:
            seeded_tasks, seeded_outcomes = evolve_from_pinch_design(heuristic_outcomes)
            esm_tasks = tuple(esm_tasks + seeded_tasks)
            esm_outcomes = tuple(esm_outcomes + seeded_outcomes)
    else:
        refines_milp = settings.runs_thermal_derivative_method
        direct_pdm_outcomes = (
            pdm_outcomes
            if settings.milp_networks_seed_evolution or not refines_milp
            else tuple(o for o in pdm_outcomes if is_pdm_variant_outcome(o))
        ) + heuristic_outcomes
        if direct_pdm_outcomes:
            esm_tasks, esm_outcomes = evolve_from_pinch_design(direct_pdm_outcomes)
        if refines_milp and pdm_outcomes:
            tdm_tasks, tdm_outcomes = execute_thermal_derivative_method_stage(
                problem=problem,
                settings=settings,
                pdm_outcomes=pdm_outcomes,
                parent_outcomes=pdm_outcome_map,
                executor=executor,
            )
            if _can_skip_derivative_stage_for_missing_couenne(
                settings,
                tdm_tasks,
                tdm_outcomes,
            ):
                _warn_couenne_fallback(
                    "Couenne is unavailable for the heat exchanger network "
                    "thermal-derivative-method topology stage",
                )
                seeded = {task.parent_task_id for task in esm_tasks}
                after_tasks, after_outcomes = evolve_from_pinch_design(
                    tuple(o for o in pdm_outcomes if o.task.task_id not in seeded)
                )
            else:
                after_tasks, after_outcomes = execute_network_evolution_method_stage(
                    problem=problem,
                    settings=settings,
                    tdm_outcomes=tdm_outcomes,
                    parent_outcomes=pdm_outcome_map | _outcome_map(tdm_outcomes),
                    executor=executor,
                )
            esm_tasks = tuple(esm_tasks + after_tasks)
            esm_outcomes = tuple(esm_outcomes + after_outcomes)

    tasks = tuple(pdm_tasks + heuristic_tasks + tdm_tasks + esm_tasks)
    outcomes = demote_infeasible_outcomes(
        pdm_outcomes + heuristic_outcomes + tdm_outcomes + esm_outcomes
    )
    return SynthesisWorkflowResult(
        tasks=tasks,
        outcomes=outcomes,
        accepted_result=build_synthesis_result(settings, tasks, outcomes),
        total_run_time=perf_counter() - start,
    )


__all__ = ["execute_open_hens_method"]
