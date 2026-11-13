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
from ..execution.fallbacks import _couenne_was_missing
from ..execution.settings import SynthesisWorkflowSettings
from ..reporting.verification import verify_network_feasibility

_METHOD_SEQUENCE: tuple[HeatExchangerNetworkDesignMethod, ...] = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.HeuristicPinchDesign,
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
    outcomes = demote_infeasible_outcomes(outcomes)
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
    manifest = HeatExchangerNetworkSynthesisManifest(
        run_id=settings.run_id,
        # The dt_cont multipliers the tasks ran.
        dt_cont_multipliers=tuple(
            sorted({float(task.dt_cont_multiplier) for task in tasks})
        )
        or settings.dt_cont_multipliers,
        derivative_thresholds=settings.quality_derivative_thresholds,
        stage_selection=settings.stage_selection,
        method_sequence=_executed_method_sequence(outcomes),
        export_formats=settings.output_formats,
        solve_tolerance=settings.solve_tolerance,
        best_solutions_to_save=settings.best_solutions_to_save,
        search_width=settings.search_width,
        search_depth=settings.search_depth,
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


def _executed_method_sequence(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkDesignMethod, ...]:
    """Return the methods that ran, in order, without stages Couenne could not run."""
    return tuple(
        dict.fromkeys(
            outcome.task.method
            for outcome in outcomes
            if not _couenne_was_missing(outcome)
        )
    )


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
    """Return the cheapest successful outcome, whichever method produced it.

    The result records that outcome's method. A later method usually improves
    on its parent, but a utility-only PDM network, or one whose descendants
    failed, can still be the cheapest.
    """
    candidates = [outcome for outcome in outcomes if outcome.status == "success"]
    if candidates:
        from ..reporting.ranking import outcome_ranking_key

        return min(candidates, key=outcome_ranking_key)
    detail = _failed_outcome_summary(outcomes)
    message = "heat exchanger network synthesis produced no successful task outcomes."
    if detail:
        message = f"{message} Failed task details: {detail}"
    if any("unconverged_termination" in outcome.diagnostics for outcome in outcomes):
        message = (
            f"{message} A solve stopped at a solver limit before converging: "
            "raise the limit (time_limit for HiGHS, max_iter for Ipopt) in "
            "HENS_SOLVER_OPTIONS_PDM, HENS_SOLVER_OPTIONS_TDM or "
            "HENS_SOLVER_OPTIONS_EVM."
        )
    raise WorkflowContractError(message)


def demote_infeasible_outcomes(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
    """Mark solver successes that fail post-solve feasibility checks as failed.

    One infeasible outcome no longer aborts the synthesis: it keeps its network
    and reasons for diagnostics, and the result is selected from the feasible
    outcomes. Selection raises only when none is left. A solve that stopped
    before converging is demoted too: it stayed a success only so downstream
    tasks could start from it, and its objective is not a converged optimum.
    """
    demoted = []
    for outcome in outcomes:
        termination = outcome.diagnostics.get("unconverged_termination")
        if outcome.status == "success" and termination is not None:
            demoted.append(
                outcome.model_copy(
                    update={
                        "status": "failed",
                        "error": f"solver stopped before converging ({termination})",
                    }
                )
            )
            continue
        failures = (
            verify_network_feasibility(outcome.network)
            if outcome.status == "success"
            else ()
        )
        if failures:
            reasons = "; ".join(failures[:3])
            outcome = outcome.model_copy(
                update={
                    "status": "failed",
                    "error": f"post-solve feasibility check failed: {reasons}",
                }
            )
        demoted.append(outcome)
    return tuple(demoted)


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
            f"{outcome.task.method}(dt_cont x{outcome.task.dt_cont_multiplier:g}): "
            f"{reason}"
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


__all__ = [
    "SynthesisWorkflowResult",
    "build_synthesis_result",
    "demote_infeasible_outcomes",
]
