"""Service entry and dispatch for heat exchanger network synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ...application.problem import PinchProblem
from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...domain.enums import HeatExchangerNetworkDesignMethod
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .context import finalise_design_result, prepare_service_context
from .execution.executor import SynthesisExecutor
from .results.seeds import resolve_seed_networks
from .targeting.network_evolution_method import (
    _execute_network_evolution_method_workflow,
)
from .targeting.open_hens_method import execute_open_hens_method
from .targeting.pinch_design_method import (
    _execute_pinch_design_method_workflow,
)
from .targeting.thermal_derivative_method import (
    _execute_thermal_derivative_method_workflow,
)

SeedNetworks = HeatExchangerNetwork | Sequence[HeatExchangerNetwork] | None


def heat_exchanger_network_synthesis_service(
    problem: PinchProblem,
    *,
    method: HeatExchangerNetworkDesignMethod | str | None = None,
    initial_networks: SeedNetworks = None,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Dispatch one HEN design method and update the problem cache."""

    if method is None:
        if initial_networks is not None:
            raise ValueError("open_hens_method does not accept initial_networks.")
        return _run_open_hens_quality_tier_service(
            problem,
            quality_tier=0,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )

    design_method = _coerce_design_method(method)
    if design_method is HeatExchangerNetworkDesignMethod.OpenHENS:
        if initial_networks is not None:
            raise ValueError("open_hens_method does not accept initial_networks.")
        return heat_exchanger_network_open_hens_method_service(
            problem,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    if design_method is HeatExchangerNetworkDesignMethod.PinchDesign:
        if initial_networks is not None:
            raise ValueError("pinch_design_method does not accept initial_networks.")
        return heat_exchanger_network_pinch_design_method_service(
            problem,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    if design_method is HeatExchangerNetworkDesignMethod.ThermalDerivative:
        return heat_exchanger_network_thermal_derivative_method_service(
            problem,
            initial_networks=initial_networks,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    if design_method is HeatExchangerNetworkDesignMethod.NetworkEvolution:
        return heat_exchanger_network_evolution_method_service(
            problem,
            initial_networks=initial_networks,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    raise AssertionError(f"Unhandled HEN design method: {design_method!r}")


def heat_exchanger_network_open_hens_method_service(
    problem: PinchProblem,
    *,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run the original tier-1 OpenHENS PDM -> TDM -> EVM sequence."""

    return _run_open_hens_quality_tier_service(
        problem,
        quality_tier=1,
        options=options,
        workspace_variant=workspace_variant,
        executor=executor,
    )


def _heat_exchanger_network_enhanced_synthesis_method_service(
    problem: PinchProblem,
    *,
    quality_tier: int = 2,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run OpenHENS synthesis with an explicit call-local quality tier."""

    return _run_open_hens_quality_tier_service(
        problem,
        quality_tier=quality_tier,
        options=options,
        workspace_variant=workspace_variant,
        executor=executor,
    )


def _run_open_hens_quality_tier_service(
    problem: PinchProblem,
    *,
    quality_tier: int,
    options: dict[str, Any] | None,
    workspace_variant: str | None,
    executor: SynthesisExecutor | None,
) -> HeatExchangerNetworkSynthesisResult:
    resolved_quality_tier = _validate_quality_tier(quality_tier)
    target_output, settings = prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )
    settings = replace(settings, synthesis_quality_tier=resolved_quality_tier)
    workflow_result = execute_open_hens_method(
        problem,
        settings,
        executor=executor,
    )
    return finalise_design_result(problem, target_output, workflow_result)


def heat_exchanger_network_pinch_design_method_service(
    problem: PinchProblem,
    *,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Run only the pinch design method and update the problem cache."""

    target_output, settings = prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )
    workflow_result = _execute_pinch_design_method_workflow(
        problem,
        settings,
        executor=executor,
    )
    return finalise_design_result(problem, target_output, workflow_result)


def heat_exchanger_network_thermal_derivative_method_service(
    problem: PinchProblem,
    *,
    initial_networks: SeedNetworks = None,
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
    target_output, settings = prepare_service_context(
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
    return finalise_design_result(problem, target_output, workflow_result)


def heat_exchanger_network_evolution_method_service(
    problem: PinchProblem,
    *,
    initial_networks: SeedNetworks = None,
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
    target_output, settings = prepare_service_context(
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
    return finalise_design_result(problem, target_output, workflow_result)


def _coerce_design_method(
    method: HeatExchangerNetworkDesignMethod | str,
) -> HeatExchangerNetworkDesignMethod:
    try:
        return HeatExchangerNetworkDesignMethod(method)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in HeatExchangerNetworkDesignMethod)
        raise ValueError(
            f"Unknown heat exchanger network design method {method!r}. "
            f"Allowed methods are: {allowed}."
        ) from exc


def _validate_quality_tier(quality_tier: int) -> int:
    if isinstance(quality_tier, bool) or not isinstance(quality_tier, int):
        raise TypeError("quality_tier must be an integer between 0 and 5.")
    if quality_tier < 0 or quality_tier > 5:
        raise ValueError("quality_tier must be between 0 and 5.")
    return int(quality_tier)


__all__ = [
    "heat_exchanger_network_evolution_method_service",
    "heat_exchanger_network_open_hens_method_service",
    "heat_exchanger_network_pinch_design_method_service",
    "heat_exchanger_network_synthesis_service",
    "heat_exchanger_network_thermal_derivative_method_service",
]
