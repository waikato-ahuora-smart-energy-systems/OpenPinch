"""Internal service boundary for problem-rooted heat exchanger network synthesis."""

from __future__ import annotations

from typing import Any, Sequence

from ...classes.heat_exchanger_network import HeatExchangerNetwork
from ...classes.pinch_problem import PinchProblem
from ...lib.schemas.io import TargetOutput
from ...lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from .methods.full_sequence import (
    SynthesisExecutor,
    SynthesisWorkflowResult,
    _execute_network_evolution_method_workflow,
    _execute_pinch_design_method_workflow,
    _execute_synthesis_workflow,
    _execute_thermal_derivative_method_workflow,
    workflow_settings_from_problem,
)
from .methods.seeds import resolve_seed_networks
from .reporting.ranking import rank_unique_network_outcomes
from .reporting.verification import verify_synthesis_result


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
    return _finalise_design_result(problem, target_output, workflow_result)


def heat_exchanger_network_pinch_design_method_service(
    problem: PinchProblem,
    *,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run only PDM and update the problem cache."""
    target_output, settings = _prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )
    workflow_result = _execute_pinch_design_method_workflow(
        problem,
        settings,
        executor=executor,
    )
    return _finalise_design_result(problem, target_output, workflow_result)


def heat_exchanger_network_thermal_derivative_method_service(
    problem: PinchProblem,
    *,
    initial_networks: HeatExchangerNetwork
    | Sequence[HeatExchangerNetwork]
    | None = None,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run only seeded TDM and update the problem cache."""
    seed_networks = resolve_seed_networks(
        problem,
        initial_networks,
        method_name="thermal_derivative_method",
        cached_source_method="pinch_design_method",
    )
    target_output, settings = _prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )
    workflow_result = _execute_thermal_derivative_method_workflow(
        problem,
        settings,
        seed_networks,
        executor=executor,
    )
    return _finalise_design_result(problem, target_output, workflow_result)


def heat_exchanger_network_evolution_method_service(
    problem: PinchProblem,
    *,
    initial_networks: HeatExchangerNetwork
    | Sequence[HeatExchangerNetwork]
    | None = None,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run only seeded network evolution and update the problem cache."""
    seed_networks = resolve_seed_networks(
        problem,
        initial_networks,
        method_name="network_evolution_method",
        cached_source_method="thermal_derivative_method",
    )
    target_output, settings = _prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )
    workflow_result = _execute_network_evolution_method_workflow(
        problem,
        settings,
        seed_networks,
        executor=executor,
    )
    return _finalise_design_result(problem, target_output, workflow_result)


def _prepare_service_context(
    problem: PinchProblem,
    *,
    options: dict[str, Any] | None,
    workspace_variant: str | None,
):
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
    return target_output, settings


def _finalise_design_result(
    problem: PinchProblem,
    target_output: TargetOutput,
    workflow_result: SynthesisWorkflowResult,
) -> HeatExchangerNetworkSynthesisResult:
    design = workflow_result.accepted_result.model_copy(
        update={
            "ranked_networks": rank_unique_network_outcomes(
                workflow_result.accepted_result
            )
        }
    )
    design.select_network()
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
