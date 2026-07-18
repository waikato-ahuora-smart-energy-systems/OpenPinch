"""Result assembly helpers for HEN synthesis methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ....contracts.synthesis.common import (
    HeatExchangerNetworkSynthesisManifest,
)
from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..errors import WorkflowContractError
from ..execution.pathways import pathways_from_metadata
from ..execution.settings import SynthesisWorkflowSettings
from ..reporting.verification import verify_network_feasibility

_METHOD_SEQUENCE: tuple[HeatExchangerNetworkDesignMethod, ...] = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.ThermalDerivative,
    HeatExchangerNetworkDesignMethod.NetworkEvolution,
)


@dataclass(frozen=True)
class SynthesisWorkflowResult:
    """Executed task graph plus accepted design data."""

    tasks: tuple[HeatExchangerNetworkSynthesisTask, ...]
    outcomes: tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]
    accepted_result: HeatExchangerNetworkSynthesisResult
    total_run_time: float


def build_synthesis_result(
    settings: SynthesisWorkflowSettings,
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> HeatExchangerNetworkSynthesisResult:
    """Convert accepted task outcomes into the canonical design data."""
    _assert_successful_outcomes_feasible(outcomes)
    accepted = _accepted_outcome(outcomes)
    if accepted.network is None:
        raise WorkflowContractError(
            "Accepted heat exchanger network outcome must include a network."
        )

    network = accepted.network.model_copy(
        update={
            "run_id": settings.run_id,
            "task_id": accepted.task.task_id,
            "period_id": settings.period_id,
            "method": accepted.task.method,
            "stage_count": accepted.network.stage_count or accepted.task.stage_count,
        }
    )
    objective_values = {
        key: value
        for key, value in {
            "total_annual_cost": network.total_annual_cost,
            "utility_cost": network.utility_cost,
            "capital_cost": network.capital_cost,
        }.items()
        if value is not None
    }
    selected_pathway = _selected_pathway_metadata(accepted)
    manifest = HeatExchangerNetworkSynthesisManifest(
        run_id=settings.run_id,
        approach_temperatures=settings.approach_temperatures,
        derivative_thresholds=settings.quality_derivative_thresholds,
        stage_selection=settings.stage_selection,
        method_sequence=settings.method_sequence,
        export_formats=settings.output_formats,
        solve_tolerance=settings.solve_tolerance,
        best_solutions_to_save=settings.best_solutions_to_save,
        synthesis_quality_tier=settings.synthesis_quality_tier,
        pdm_stage_pair_limit=settings.pdm_stage_pair_limit,
        tdm_parent_limit=settings.tdm_parent_limit,
        stage_packing=settings.stage_packing,
        evm_n_ad_branches=settings.effective_evm_n_ad_branches,
        evm_n_rm_branches=settings.effective_evm_n_rm_branches,
        task_ids=tuple(task.task_id for task in tasks if task.task_id is not None),
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        period_id=settings.period_id,
        design_method=settings.design_method,
        selected_pathway_id=selected_pathway.get("pathway_id"),
        selected_pathway_kind=selected_pathway.get("pathway_kind"),
        selected_pdm_mode=selected_pathway.get("pdm_mode"),
        selected_tier_origin=selected_pathway.get("tier_origin"),
        selected_protected_pathway=bool(selected_pathway.get("protected", False)),
        task_count_by_method=_task_count_by_method(tasks),
    )
    return HeatExchangerNetworkSynthesisResult(
        network=network,
        run_id=settings.run_id,
        task_id=accepted.task.task_id,
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        period_id=settings.period_id,
        solver_name=settings.solver_for(accepted.task.method),
        solver_status=accepted.solver_status,
        design_method=settings.design_method,
        method=accepted.task.method,
        stage_count=network.stage_count,
        objective_values=objective_values,
        ranked_networks=tuple(outcomes),
        manifest=manifest,
    )


def _selected_pathway_metadata(
    accepted: HeatExchangerNetworkSynthesisTaskOutcome,
) -> dict[str, object]:
    pathways = pathways_from_metadata(accepted.task.metadata)
    if not pathways:
        return {}
    pathway = pathways[0]
    return pathway.metadata()


def _task_count_by_method(
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[str(task.method)] = counts.get(str(task.method), 0) + 1
    return counts


def _accepted_outcome(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    for method in reversed(_METHOD_SEQUENCE):
        candidates = [
            outcome
            for outcome in outcomes
            if outcome.status == "success" and outcome.task.method == method
        ]
        if candidates:
            return min(
                candidates,
                key=lambda outcome: (
                    float("inf")
                    if outcome.objective_value is None
                    else outcome.objective_value
                ),
            )
    detail = _failed_outcome_summary(outcomes)
    message = "heat exchanger network synthesis produced no successful task outcomes."
    if detail:
        message = f"{message} Failed task details: {detail}"
    raise WorkflowContractError(message)


def _assert_successful_outcomes_feasible(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> None:
    infeasible = [
        (outcome, failures)
        for outcome in outcomes
        if outcome.status == "success"
        and (failures := verify_network_feasibility(outcome.network))
    ]
    if not infeasible:
        return

    raise WorkflowContractError(
        "solver-success heat exchanger network task failed post-solve "
        "feasibility checks, which indicates a solver model or extraction "
        "contract issue. Details: " + _infeasible_outcome_summary(infeasible)
    )


def _failed_outcome_summary(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> str:
    method_rank = {method: rank for rank, method in enumerate(_METHOD_SEQUENCE)}
    failed = sorted(
        (outcome for outcome in outcomes if outcome.status != "success"),
        key=lambda outcome: method_rank.get(outcome.task.method, -1),
        reverse=True,
    )
    if not failed:
        return ""
    summaries = []
    for outcome in failed[:3]:
        reason = outcome.error or outcome.solver_status or "no failure reason"
        summaries.append(
            f"{outcome.task.method}({outcome.task.approach_temperature:g} K): {reason}"
        )
    if len(failed) > len(summaries):
        summaries.append(f"{len(failed) - len(summaries)} more failed task(s)")
    return "; ".join(summaries)


def _infeasible_outcome_summary(
    outcomes: Sequence[
        tuple[HeatExchangerNetworkSynthesisTaskOutcome, tuple[str, ...]]
    ],
) -> str:
    if not outcomes:
        return ""
    summaries = []
    for outcome, failures in outcomes[:3]:
        task_id = outcome.task.task_id or "unknown-task"
        summaries.append(f"{outcome.task.method}({task_id}) infeasible: {failures[0]}")
    if len(outcomes) > len(summaries):
        summaries.append(f"{len(outcomes) - len(summaries)} more infeasible task(s)")
    return "; ".join(summaries)


__all__ = ["SynthesisWorkflowResult", "build_synthesis_result"]
