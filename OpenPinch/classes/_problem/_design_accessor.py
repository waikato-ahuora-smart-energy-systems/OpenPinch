"""Design-workflow accessor for problem-owned heat exchanger network operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from ..heat_exchanger_network import HeatExchangerNetwork
    from ...lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
    from ..pinch_problem import PinchProblem


class _DesignAccessor:
    """Callable design helper for advanced workflows rooted in one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def heat_exchanger_network_synthesis(
        self,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run heat exchanger network synthesis and cache the design result."""
        from ...services.heat_exchanger_network_synthesis.service import (
            heat_exchanger_network_synthesis_service,
        )

        return heat_exchanger_network_synthesis_service(
            self._problem,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def pinch_decomposition(
        self,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run only the PDM method and cache the design result."""
        from ...services.heat_exchanger_network_synthesis.service import (
            heat_exchanger_network_pinch_decomposition_service,
        )

        return heat_exchanger_network_pinch_decomposition_service(
            self._problem,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def thermal_derivative(
        self,
        initial_networks: HeatExchangerNetwork
        | Sequence[HeatExchangerNetwork]
        | None = None,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run only seeded TDM and cache the design result."""
        from ...services.heat_exchanger_network_synthesis.service import (
            heat_exchanger_network_thermal_derivative_service,
        )

        return heat_exchanger_network_thermal_derivative_service(
            self._problem,
            initial_networks=initial_networks,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def network_evolution(
        self,
        initial_networks: HeatExchangerNetwork
        | Sequence[HeatExchangerNetwork]
        | None = None,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run only seeded network evolution and cache the design result."""
        from ...services.heat_exchanger_network_synthesis.service import (
            heat_exchanger_network_evolution_service,
        )

        return heat_exchanger_network_evolution_service(
            self._problem,
            initial_networks=initial_networks,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def _runtime_options(
        self,
        options: Optional[dict[str, Any]],
        state_id: Optional[str],
    ) -> dict[str, Any]:
        from ...services.heat_exchanger_network_synthesis.service import (
            _normalise_runtime_options,
        )

        runtime_options = _normalise_runtime_options(options)
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return runtime_options


class _DesignAccessorDescriptor:
    """Non-data descriptor exposing design workflows on problem instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _DesignAccessor(obj)
