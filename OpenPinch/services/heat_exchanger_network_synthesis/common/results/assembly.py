"""Result assembly helpers for HEN synthesis methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .....lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisManifest,
    HeatExchangerNetworkSynthesisResult,
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    SynthesisMethod,
)
from ..errors import WorkflowContractError
from ..execution.settings import SynthesisWorkflowSettings

_METHOD_SEQUENCE: tuple[SynthesisMethod, ...] = (
    "pinch_design_method",
    "thermal_derivative_method",
    "network_evolution_method",
)


@dataclass(frozen=True)
class SynthesisWorkflowResult:
    """Executed task graph plus accepted design payload."""

    tasks: tuple[HeatExchangerNetworkSynthesisTask, ...]
    outcomes: tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]
    accepted_result: HeatExchangerNetworkSynthesisResult
    total_run_time: float


def build_synthesis_result(
    settings: SynthesisWorkflowSettings,
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> HeatExchangerNetworkSynthesisResult:
    """Convert accepted task outcomes into the canonical design payload."""
    accepted = _accepted_outcome(outcomes)
    if accepted.network is None:
        raise WorkflowContractError(
            "Accepted heat exchanger network outcome must include a network."
        )

    network = accepted.network.model_copy(
        update={
            "run_id": settings.run_id,
            "task_id": accepted.task.task_id,
            "state_id": settings.state_id,
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
    manifest = HeatExchangerNetworkSynthesisManifest(
        run_id=settings.run_id,
        approach_temperatures=settings.approach_temperatures,
        derivative_thresholds=settings.derivative_thresholds,
        stage_selection=settings.stage_selection,
        method_sequence=settings.method_sequence,
        export_formats=settings.output_formats,
        solve_tolerance=settings.solve_tolerance,
        best_solutions_to_save=settings.best_solutions_to_save,
        task_ids=tuple(task.task_id for task in tasks if task.task_id is not None),
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        state_id=settings.state_id,
        design_method=settings.design_method,
    )
    return HeatExchangerNetworkSynthesisResult(
        network=network,
        run_id=settings.run_id,
        task_id=accepted.task.task_id,
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        state_id=settings.state_id,
        solver_name=settings.solver_for(accepted.task.method),
        solver_status=accepted.solver_status,
        design_method=settings.design_method,
        method=accepted.task.method,
        stage_count=network.stage_count,
        objective_values=objective_values,
        ranked_networks=tuple(outcomes),
        manifest=manifest,
    )


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


__all__ = ["SynthesisWorkflowResult", "build_synthesis_result"]
