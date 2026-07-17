"""Fallback policy for OpenHENS sequence execution."""

from __future__ import annotations

import warnings
from typing import Sequence

from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from .settings import SynthesisWorkflowSettings
from .task_builders import _successful_method


def _can_skip_derivative_stage_for_missing_couenne(
    settings: SynthesisWorkflowSettings,
    tdm_tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    tdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> bool:
    if settings.tdm_solver != "couenne" or not tdm_tasks or not tdm_outcomes:
        return False
    if any(
        _successful_method(outcome, "thermal_derivative_method")
        for outcome in tdm_outcomes
    ):
        return False
    return all(_missing_couenne_failure(outcome) for outcome in tdm_outcomes)


def _can_skip_preliminary_stages_for_missing_couenne(
    settings: SynthesisWorkflowSettings,
    pdm_tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> bool:
    if settings.pdm_solver != "couenne" or not pdm_tasks or not pdm_outcomes:
        return False
    if any(
        _successful_method(outcome, "pinch_design_method") for outcome in pdm_outcomes
    ):
        return False
    return all(_missing_couenne_failure(outcome) for outcome in pdm_outcomes)


def _missing_couenne_failure(outcome: HeatExchangerNetworkSynthesisTaskOutcome) -> bool:
    if outcome.status == "success" or outcome.task.method not in {
        "pinch_design_method",
        "thermal_derivative_method",
    }:
        return False
    error = outcome.error or ""
    return "couenne" in error and "not found on PATH" in error


def _warn_couenne_fallback(stage: str) -> None:
    warnings.warn(
        f"{stage}; skipping Couenne-backed derivative/topology setup and "
        "running network_evolution_method directly.",
        RuntimeWarning,
        stacklevel=3,
    )


__all__ = [
    "_can_skip_derivative_stage_for_missing_couenne",
    "_can_skip_preliminary_stages_for_missing_couenne",
    "_warn_couenne_fallback",
]
