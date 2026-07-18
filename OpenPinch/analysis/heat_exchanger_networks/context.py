"""Shared service-boundary helpers for HEN synthesis entry calls."""

from __future__ import annotations

from typing import Any

from ...application.problem import PinchProblem
from ...contracts.output import TargetOutput
from .execution.settings import (
    SynthesisWorkflowSettings,
    workflow_settings_from_problem,
)
from .reporting.ranking import rank_unique_network_outcomes
from .reporting.verification import verify_synthesis_result
from .results.assembly import SynthesisWorkflowResult


def prepare_service_context(
    problem: PinchProblem,
    *,
    options: dict[str, Any] | None,
    workspace_variant: str | None,
) -> tuple[TargetOutput, SynthesisWorkflowSettings]:
    """Validate the problem, resolve targets, and build HEN workflow settings."""

    if not isinstance(problem, PinchProblem):
        raise TypeError(
            "heat_exchanger_network_synthesis_service requires a live PinchProblem."
        )

    runtime_options = normalise_runtime_options(options)
    period_id = optional_text(runtime_options.get("period_id"))
    target_output = ensure_target_results(problem, runtime_options, period_id)
    settings = workflow_settings_from_problem(
        problem,
        period_id=period_id,
        workspace_variant=workspace_variant,
    )
    return target_output, settings


def finalise_design_result(
    problem: PinchProblem,
    target_output: TargetOutput,
    workflow_result: SynthesisWorkflowResult,
):
    """Cache, rank, verify, and return the accepted HEN design result."""

    design = workflow_result.accepted_result.model_copy(
        update={
            "ranked_networks": rank_unique_network_outcomes(
                workflow_result.accepted_result
            )
        }
    )
    from .results.selection import select_network

    select_network(design)
    verification_failures = verify_synthesis_result(design)
    if verification_failures:
        raise RuntimeError(
            "heat exchanger network synthesis verification failed: "
            + "; ".join(verification_failures)
        )

    problem._results = TargetOutput.model_validate(
        target_output.model_copy(update={"design": design})
    )
    return design


def ensure_target_results(
    problem: PinchProblem,
    runtime_options: dict[str, Any],
    period_id: str | None,
) -> TargetOutput:
    """Return cached target results or compute them for the active problem."""

    cached = problem.results
    if (
        cached is not None
        and cached.targets
        and (period_id is None or cached.period_id == period_id)
    ):
        return cached
    problem.target.direct_heat_integration(options=runtime_options)
    return TargetOutput.model_validate(problem.results)


def normalise_runtime_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Validate runtime options accepted by design accessors."""

    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError(
            "heat exchanger network synthesis runtime options must be supplied "
            "as a dict. Persistent heat exchanger network controls belong in "
            "TargetInput.options before the PinchProblem is loaded."
        )

    hens_overrides = sorted(str(key) for key in options if str(key).startswith("HENS_"))
    if hens_overrides:
        raise ValueError(
            "heat exchanger network synthesis configuration must be loaded through "
            "TargetInput.options / prepared Configuration, not passed as "
            "separate design options: " + ", ".join(hens_overrides)
        )
    return dict(options)


def optional_text(value: Any) -> str | None:
    """Return a string value when provided."""

    if value is None:
        return None
    return str(value)


__all__ = [
    "ensure_target_results",
    "finalise_design_result",
    "normalise_runtime_options",
    "optional_text",
    "prepare_service_context",
]
