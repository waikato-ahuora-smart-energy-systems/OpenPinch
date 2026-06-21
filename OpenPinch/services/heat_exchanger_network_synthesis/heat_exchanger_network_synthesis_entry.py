"""Service entry and dispatch for heat exchanger network synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...classes.heat_exchanger_network import HeatExchangerNetwork
from ...classes.pinch_problem import PinchProblem
from ...lib.enums import HeatExchangerNetworkDesignMethod, HENDesignMethod
from ...lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from .common.execution.executor import SynthesisExecutor
from .common.results.seeds import resolve_seed_networks
from .common.service_context import finalise_design_result, prepare_service_context
from .targeting_services.network_evolution_method import (
    _execute_network_evolution_method_workflow,
)
from .targeting_services.open_hens_method import execute_open_hens_method
from .targeting_services.pinch_design_method import (
    _execute_pinch_design_method_workflow,
)
from .targeting_services.thermal_derivative_method import (
    _execute_thermal_derivative_method_workflow,
)

SeedNetworks = HeatExchangerNetwork | Sequence[HeatExchangerNetwork] | None


def heat_exchanger_network_synthesis_service(
    problem: PinchProblem,
    *,
    method: HeatExchangerNetworkDesignMethod | str = HENDesignMethod.OpenHENS,
    initial_networks: SeedNetworks = None,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Dispatch one HEN design method and update the problem cache."""

    design_method = _coerce_design_method(method)
    if design_method is HENDesignMethod.OpenHENS:
        if initial_networks is not None:
            raise ValueError("open_hens_method does not accept initial_networks.")
        return heat_exchanger_network_open_hens_method_service(
            problem,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    if design_method is HENDesignMethod.PinchDesign:
        if initial_networks is not None:
            raise ValueError("pinch_design_method does not accept initial_networks.")
        return heat_exchanger_network_pinch_design_method_service(
            problem,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    if design_method is HENDesignMethod.ThermalDerivative:
        return heat_exchanger_network_thermal_derivative_method_service(
            problem,
            initial_networks=initial_networks,
            options=options,
            workspace_variant=workspace_variant,
            executor=executor,
        )
    if design_method is HENDesignMethod.NetworkEvolution:
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
    """Run the published OpenHENS PDM -> TDM -> EVOL sequence."""

    target_output, settings = prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )
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


__all__ = [
    "heat_exchanger_network_evolution_method_service",
    "heat_exchanger_network_open_hens_method_service",
    "heat_exchanger_network_pinch_design_method_service",
    "heat_exchanger_network_synthesis_service",
    "heat_exchanger_network_thermal_derivative_method_service",
]
