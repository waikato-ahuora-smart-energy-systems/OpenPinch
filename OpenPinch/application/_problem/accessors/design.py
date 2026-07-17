"""Descriptive heat-exchanger-network design workflows and result views."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from ..arguments import (
    split_runtime_and_configuration_options,
    temporary_zone_configuration,
)

if TYPE_CHECKING:
    from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
    from ....domain.heat_exchanger_network import HeatExchangerNetwork
    from ...problem import PinchProblem


def _set_if_supplied(values: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        values[key] = value


class HeatExchangerNetworkDesignView:
    """Explicit application behavior around one serializable HEN result."""

    def __init__(self, result: "HeatExchangerNetworkSynthesisResult") -> None:
        self._result = result

    @property
    def result(self) -> "HeatExchangerNetworkSynthesisResult":
        """Return the complete serializable synthesis result."""
        return self._result

    @property
    def selected_network(self) -> "HeatExchangerNetwork":
        """Return the network selected by the synthesis service."""
        return self.result.network

    def top(self, n: int):
        """Return the first ``n`` ranked successful outcomes."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        return self.result.ranked_networks[:n]

    def network(self, *, rank: int = 1) -> "HeatExchangerNetwork":
        """Return one ranked network using a one-based rank."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive one-based integer.")
        from ....analysis.heat_exchanger_networks.results.selection import (
            network_for_rank,
        )

        return network_for_rank(self.result, rank)

    def grid(
        self,
        *,
        rank: int = 1,
        period_id: str | None = None,
        temperature_scaled: bool = False,
    ):
        """Lazily render one ranked network as a grid diagram."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive one-based integer.")
        from ....presentation.network_grid.results import build_result_grid_diagram

        return build_result_grid_diagram(
            self.result,
            solution_rank=rank,
            period_id=period_id,
            temperature_scaled=temperature_scaled,
        )

    @property
    def total_heat_recovery(self) -> float:
        from ....domain.enums import HeatExchangerKind

        return self.selected_network.total_duty(kind=HeatExchangerKind.RECOVERY)

    @property
    def total_hot_utility(self) -> float:
        from ....domain.enums import HeatExchangerKind

        return self.selected_network.total_duty(kind=HeatExchangerKind.HOT_UTILITY)

    @property
    def total_cold_utility(self) -> float:
        from ....domain.enums import HeatExchangerKind

        return self.selected_network.total_duty(kind=HeatExchangerKind.COLD_UTILITY)

    def utility(self, name: str, *, period_id: str | None = None) -> float:
        """Return selected-network utility duty for one named utility."""
        from ....domain.enums import HeatExchangerKind

        if not isinstance(name, str) or not name.strip():
            raise ValueError("utility name must be a non-empty string")
        utility_name = name.strip()
        network = self.selected_network
        resolved_period_id = network.resolve_period_id(period_id)
        total = 0.0
        for exchanger in network.exchangers:
            matches = (
                exchanger.kind is HeatExchangerKind.HOT_UTILITY
                and exchanger.source_stream == utility_name
            ) or (
                exchanger.kind is HeatExchangerKind.COLD_UTILITY
                and exchanger.sink_stream == utility_name
            )
            if matches:
                state = exchanger.state(resolved_period_id)
                if state.active:
                    total += float(state.duty)
        return total


class _DesignAccessor:
    """Explicit HEN design workflows rooted in one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def _arguments(
        self,
        *,
        options: Mapping[str, Any] | None,
        period_id: str | None,
        approach_temperatures=None,
        dt_cont_multipliers=None,
        derivative_thresholds=None,
        stages=None,
        pack_stages: bool | None = None,
        solver=None,
        max_parallel=None,
        best_solutions=None,
        solve_tolerance=None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("design options must be supplied as a mapping.")
        runtime, configuration = split_runtime_and_configuration_options(options)
        named: dict[str, Any] = {}
        for key, value in (
            ("HENS_APPROACH_TEMPERATURES", approach_temperatures),
            ("HENS_DT_CONT_MULTIPLIERS", dt_cont_multipliers),
            ("HENS_DERIVATIVE_THRESHOLDS", derivative_thresholds),
            ("HENS_STAGE_SELECTION", stages),
            ("HENS_MAX_PARALLEL", max_parallel),
            ("HENS_BEST_SOLUTIONS_TO_SAVE", best_solutions),
            ("HENS_SOLVE_TOLERANCE", solve_tolerance),
        ):
            _set_if_supplied(named, key, value)
        if pack_stages is not None:
            if not isinstance(pack_stages, bool):
                raise TypeError("pack_stages must be a bool when supplied.")
            named["HENS_STAGE_PACKING"] = "all" if pack_stages else "none"
        if solver is not None:
            for key in ("HENS_SOLVER_PDM", "HENS_SOLVER_TDM", "HENS_SOLVER_EVM"):
                named[key] = solver
        configuration.update(named)
        if period_id is not None:
            runtime["period_id"] = period_id
        return runtime, configuration

    def _run(self, service, *, runtime, configuration, **service_kwargs):
        root = self._problem._build_execution_master_zone()
        with temporary_zone_configuration(root, configuration):
            result = service(
                self._problem,
                options=runtime,
                **service_kwargs,
            )
        return HeatExchangerNetworkDesignView(result)

    def heat_exchanger_network(
        self,
        *,
        initial_networks: (
            "HeatExchangerNetwork | Sequence[HeatExchangerNetwork] | None"
        ) = None,
        options: Mapping[str, Any] | None = None,
        period_id: str | None = None,
        approach_temperatures=None,
        dt_cont_multipliers=None,
        derivative_thresholds=None,
        stages=None,
        pack_stages: bool | None = None,
        solver=None,
        max_parallel=None,
        best_solutions=None,
        solve_tolerance=None,
    ) -> HeatExchangerNetworkDesignView:
        """Run the fixed default HEN synthesis workflow."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_synthesis_service,
        )

        runtime, configuration = self._arguments(
            options=options,
            period_id=period_id,
            approach_temperatures=approach_temperatures,
            dt_cont_multipliers=dt_cont_multipliers,
            derivative_thresholds=derivative_thresholds,
            stages=stages,
            pack_stages=pack_stages,
            solver=solver,
            max_parallel=max_parallel,
            best_solutions=best_solutions,
            solve_tolerance=solve_tolerance,
        )
        return self._run(
            heat_exchanger_network_synthesis_service,
            runtime=runtime,
            configuration=configuration,
            initial_networks=initial_networks,
            method=None,
            workspace_variant=None,
        )

    def enhanced_heat_exchanger_network(
        self,
        *,
        quality_tier: int = 2,
        options: Mapping[str, Any] | None = None,
        period_id: str | None = None,
        **kwargs,
    ) -> HeatExchangerNetworkDesignView:
        """Run enhanced OpenHENS synthesis at an explicit quality tier."""
        from ....analysis.heat_exchanger_networks.service import (
            _heat_exchanger_network_enhanced_synthesis_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            _heat_exchanger_network_enhanced_synthesis_method_service,
            runtime=runtime,
            configuration=configuration,
            quality_tier=quality_tier,
            workspace_variant=None,
        )

    def open_hens(self, *, options=None, period_id=None, **kwargs):
        """Run the tier-one OpenHENS PDM -> TDM -> EVM workflow."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_open_hens_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_open_hens_method_service,
            runtime=runtime,
            configuration=configuration,
            workspace_variant=None,
        )

    def pinch_design(self, *, options=None, period_id=None, **kwargs):
        """Run the pinch design method only."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_pinch_design_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_pinch_design_method_service,
            runtime=runtime,
            configuration=configuration,
            workspace_variant=None,
        )

    def thermal_derivative(
        self,
        initial_networks=None,
        *,
        options=None,
        period_id=None,
        **kwargs,
    ):
        """Run seeded thermal derivative synthesis only."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_thermal_derivative_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_thermal_derivative_method_service,
            runtime=runtime,
            configuration=configuration,
            initial_networks=initial_networks,
            workspace_variant=None,
        )

    def network_evolution(
        self,
        initial_networks=None,
        *,
        options=None,
        period_id=None,
        **kwargs,
    ):
        """Run seeded network evolution synthesis only."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_evolution_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_evolution_method_service,
            runtime=runtime,
            configuration=configuration,
            initial_networks=initial_networks,
            workspace_variant=None,
        )

    def multiperiod_heat_exchanger_network(self, **kwargs):
        """Synthesize one shared HEN after explicit all-period targeting."""
        if not self._problem.period_results:
            raise RuntimeError(
                "Run problem.target.all_periods.<method>() before multiperiod HEN "
                "synthesis."
            )
        if "period_id" in kwargs:
            raise TypeError("multiperiod synthesis does not accept period_id.")
        return self.heat_exchanger_network(**kwargs)


class _DesignAccessorDescriptor:
    """Non-data descriptor exposing design workflows on problem instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _DesignAccessor(obj)


__all__ = ["HeatExchangerNetworkDesignView"]
