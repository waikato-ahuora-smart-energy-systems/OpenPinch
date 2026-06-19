"""Internal service boundary for problem-rooted heat exchanger network synthesis."""

from __future__ import annotations

from typing import Any

from ...classes.pinch_problem import PinchProblem
from ...lib.schemas.io import TargetOutput
from ...lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from .verification import verify_synthesis_result
from .workflow import (
    SynthesisExecutor,
    _execute_synthesis_workflow,
    workflow_settings_from_problem,
)


def heat_exchanger_network_synthesis_service(
    problem: PinchProblem,
    *,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run heat exchanger network synthesis and update the problem cache.

    This function is intentionally not re-exported from ``OpenPinch`` or
    ``OpenPinch.services``. The stable public semantics are the problem-owned
    design accessor and workspace workflow dispatch; this service is the single
    internal implementation boundary behind those entry points.
    """
    if not isinstance(problem, PinchProblem):
        raise TypeError(
            "heat_exchanger_network_synthesis_service requires a live PinchProblem."
        )

    runtime_options = _normalise_runtime_options(options)
    state_id = _optional_text(runtime_options.get("state_id"))
    target_output = _ensure_target_results(problem, runtime_options, state_id)
    settings = workflow_settings_from_problem(
        problem,
        state_id=state_id,
        workspace_variant=workspace_variant,
    )
    workflow_result = _execute_synthesis_workflow(
        problem,
        settings,
        executor=executor,
    )
    design = workflow_result.accepted_result
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


def _ensure_target_results(
    problem: PinchProblem,
    runtime_options: dict[str, Any],
    state_id: str | None,
) -> TargetOutput:
    cached = problem.results
    if (
        cached is not None
        and cached.targets
        and (state_id is None or cached.state_id == state_id)
    ):
        return cached
    return TargetOutput.model_validate(problem.target(options=runtime_options))


def _normalise_runtime_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError(
            "heat exchanger network synthesis runtime options must be supplied "
            "as a dict. Persistent heat exchanger network controls belong in "
            "TargetInput.options before the "
            "PinchProblem is loaded."
        )

    hens_overrides = sorted(str(key) for key in options if str(key).startswith("HENS_"))
    if hens_overrides:
        raise ValueError(
            "heat exchanger network synthesis configuration must be loaded through "
            "TargetInput.options / prepared Configuration, not passed as "
            "separate design options: " + ", ".join(hens_overrides)
        )
    return dict(options)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
