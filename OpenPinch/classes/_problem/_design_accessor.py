"""Design-workflow accessor for problem-owned heat exchanger network operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from ...lib.enums import HeatExchangerNetworkDesignMethod
    from ...lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
    from ..heat_exchanger_network import HeatExchangerNetwork
    from ..pinch_problem import PinchProblem


class _DesignNetworkAccessor:
    """Convenience accessors for the selected heat exchanger network design."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    @property
    def total_heat_recovery(self) -> float:
        """Return the selected network's total process heat recovery duty."""
        from ..heat_exchanger import HeatExchangerKind

        return self._selected_network().total_duty(kind=HeatExchangerKind.RECOVERY)

    @property
    def total_hot_utility(self) -> float:
        """Return the selected network's total hot utility duty."""
        from ..heat_exchanger import HeatExchangerKind

        return self._selected_network().total_duty(kind=HeatExchangerKind.HOT_UTILITY)

    @property
    def total_cold_utility(self) -> float:
        """Return the selected network's total cold utility duty."""
        from ..heat_exchanger import HeatExchangerKind

        return self._selected_network().total_duty(kind=HeatExchangerKind.COLD_UTILITY)

    def utility(self, name: str) -> float:
        """Return selected-network hot/cold utility exchanger duty for ``name``."""
        from ..heat_exchanger import HeatExchangerKind

        if not isinstance(name, str) or not name.strip():
            raise ValueError("utility name must be a non-empty string")
        utility_name = name.strip()
        total = 0.0
        for exchanger in self._selected_network().exchangers:
            if (
                exchanger.kind is HeatExchangerKind.HOT_UTILITY
                and exchanger.source_stream == utility_name
            ) or (
                exchanger.kind is HeatExchangerKind.COLD_UTILITY
                and exchanger.sink_stream == utility_name
            ):
                total += float(exchanger.duty)
        return total

    def _selected_network(self) -> "HeatExchangerNetwork":
        results = self._problem.results
        design = None if results is None else results.design
        if design is None:
            raise RuntimeError(
                "Run problem.design.heat_exchanger_network_synthesis(...) before "
                "accessing problem.design.network helpers."
            )
        return design.network


class _DesignAccessor:
    """Callable design helper for advanced workflows rooted in one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem
        self.network = _DesignNetworkAccessor(problem)

    def heat_exchanger_network_synthesis(
        self,
        *,
        method: "HeatExchangerNetworkDesignMethod | str | None" = None,
        initial_networks: (
            "HeatExchangerNetwork | Sequence[HeatExchangerNetwork] | None"
        ) = None,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run a selected HEN design method and cache the design result."""
        from ...lib.enums import HENDesignMethod
        from ...services.heat_exchanger_network_synthesis import (
            heat_exchanger_network_synthesis_entry as hens_entry,
        )

        return hens_entry.heat_exchanger_network_synthesis_service(
            self._problem,
            method=HENDesignMethod.OpenHENS if method is None else method,
            initial_networks=initial_networks,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def open_hens_method(
        self,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run the published OpenHENS PDM -> TDM -> EVOL method."""
        from ...services.heat_exchanger_network_synthesis import (
            heat_exchanger_network_synthesis_entry as hens_entry,
        )

        return hens_entry.heat_exchanger_network_open_hens_method_service(
            self._problem,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def pinch_design_method(
        self,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run only the PDM method and cache the design result."""
        from ...services.heat_exchanger_network_synthesis import (
            heat_exchanger_network_synthesis_entry as hens_entry,
        )

        return hens_entry.heat_exchanger_network_pinch_design_method_service(
            self._problem,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def thermal_derivative_method(
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
        from ...services.heat_exchanger_network_synthesis import (
            heat_exchanger_network_synthesis_entry as hens_entry,
        )

        return hens_entry.heat_exchanger_network_thermal_derivative_method_service(
            self._problem,
            initial_networks=initial_networks,
            options=self._runtime_options(options, state_id),
            workspace_variant=workspace_variant,
        )

    def network_evolution_method(
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
        from ...services.heat_exchanger_network_synthesis import (
            heat_exchanger_network_synthesis_entry as hens_entry,
        )

        return hens_entry.heat_exchanger_network_evolution_method_service(
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
        from ...services.heat_exchanger_network_synthesis.common import (
            service_context,
        )

        runtime_options = service_context.normalise_runtime_options(options)
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return runtime_options


class _DesignAccessorDescriptor:
    """Non-data descriptor exposing design workflows on problem instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _DesignAccessor(obj)
