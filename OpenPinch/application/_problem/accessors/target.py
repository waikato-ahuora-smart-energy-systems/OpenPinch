"""Descriptive targeting workflows owned by :class:`PinchProblem`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping

from ....analysis.heat_pumps.performance_maps.generation import (
    generate_hpr_performance_map,
)
from ....analysis.heat_pumps.performance_maps.target_basis import (
    build_hpr_target_map_basis,
)
from ....analysis.heat_pumps.performance_maps.targeting import (
    normalize_hpr_simulation_backend,
)
from ....contracts.hpr import HPRSearchBudget
from ....contracts.hpr_performance_map import (
    HprPerformanceMap,
    HprPerformanceMapRequest,
)
from ....contracts.output import TargetOutput
from ....domain.enums import (
    HeatPumpAndRefrigerationCycle,
    TargetType,
    TurbineModel,
    ZoneType,
)
from ....domain.targets import BaseTargetModel
from ...targeting import (
    area_cost_targeting_service,
    direct_heat_integration_service,
    direct_heat_pump_service,
    direct_orc_service,
    direct_refrigeration_service,
    energy_transfer_analysis_service,
    exergy_targeting_service,
    indirect_heat_integration_service,
    indirect_heat_pump_service,
    indirect_refrigeration_service,
    power_cogeneration_service,
)
from ..arguments import (
    split_runtime_and_configuration_options,
    temporary_zone_configuration,
)
from ..targeting.catalog import install_catalog_forwarders, require_available
from ..targeting.provenance import (
    resolve_hpr_residual_case,
    resolve_target_selection,
    validate_base_target,
)
from ..targeting.state import (
    prepared_period_state,
    snapshot_problem,
    target_transaction,
)

if TYPE_CHECKING:
    from ....contracts.heat_recovery_dt_min import HeatRecoveryDtMinResult
    from ....domain.zone import Zone
    from ...problem import PinchProblem


def _load_options(
    *,
    load_fraction: float | None,
    load_duty: float | None,
    period_loads: Mapping[str, float] | None,
) -> dict[str, Any]:
    supplied = {
        "load_fraction": load_fraction,
        "load_duty": load_duty,
        "period_loads": period_loads,
    }
    selected = [name for name, value in supplied.items() if value is not None]
    if len(selected) > 1:
        raise ValueError(
            "Supply only one of load_fraction, load_duty, or period_loads; "
            f"received {', '.join(selected)}."
        )
    if load_fraction is not None:
        return {"HPR_LOAD_MODE": "fraction", "HPR_LOAD_FRACTION": load_fraction}
    if load_duty is not None:
        return {"HPR_LOAD_MODE": "duty", "HPR_LOAD_DUTY": load_duty}
    if period_loads is not None:
        return {
            "HPR_LOAD_MODE": "period_values",
            "HPR_LOAD_PERIOD_VALUES": dict(period_loads),
        }
    return {}


def _set_if_not_none(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


class _AllPeriodsTargetAccessor:
    """Mirror supported target methods over canonical operating periods."""

    def __init__(self, target: "_TargetAccessor") -> None:
        self._target = target

    def _run(self, method_name: str, *, workers: int, kwargs: dict[str, Any]):
        if kwargs.get("base_target") is not None:
            raise ValueError(
                "all_periods cannot broadcast base_target; use per-period calls."
            )
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        problem = self._target._problem
        period_ids = list(problem.period_ids)
        if "period_id" in kwargs:
            raise ValueError("all_periods selects every canonical period itself.")

        def solve(period_id: str):
            isolated = snapshot_problem(problem, problem._period_states.get(period_id))
            method = getattr(isolated.target, method_name)
            method(period_id=period_id, **kwargs)
            return deepcopy(isolated._results), prepared_period_state(isolated)

        if workers == 1:
            solved = [solve(period_id) for period_id in period_ids]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                solved = list(executor.map(solve, period_ids))
        outputs = {sid: item[0] for sid, item in zip(period_ids, solved, strict=True)}
        states = {sid: item[1] for sid, item in zip(period_ids, solved, strict=True)}
        detached = deepcopy(outputs)
        problem._period_results = outputs
        problem._period_states = states
        return detached

    def heat_recovery_dt_min(
        self,
        *,
        heat_recovery,
        zone=None,
        workers=1,
    ) -> dict[str, "HeatRecoveryDtMinResult"]:
        from ...heat_recovery_dt_min import (
            calculate_all_period_heat_recovery_dt_min,
        )

        return calculate_all_period_heat_recovery_dt_min(
            self._target._problem,
            heat_recovery=heat_recovery,
            zone=zone,
            workers=workers,
        )

    def utility_placement(self, **kwargs):
        """Optimize one shared placement over every canonical period."""
        if kwargs.get("base_target") is not None:
            raise ValueError(
                "all_periods cannot broadcast base_target; use per-period calls."
            )
        kwargs.pop("period_ids", None)
        return self._target.utility_placement(
            period_ids=tuple(self._target._problem.period_ids),
            **kwargs,
        )


def _all_periods_forwarder(method_name: str):
    def forward(self, *, workers: int = 1, **kwargs):
        return self._run(method_name, workers=workers, kwargs=kwargs)

    forward.__doc__ = (
        f"Run ``problem.target.{method_name}`` once per canonical period.\n\n"
        "Returns detached per-period outputs keyed by ``period_id``; "
        "``workers`` sets how many periods are solved concurrently."
    )
    return forward


install_catalog_forwarders(
    _AllPeriodsTargetAccessor,
    surface="target",
    factory=_all_periods_forwarder,
    exclude=(
        "brayton_heat_pump",
        "brayton_refrigeration",
        "hpr_performance_map",
    ),
)


class _TargetAccessor:
    """Explicit, discoverable targeting workflows for one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    @property
    def all_periods(self) -> _AllPeriodsTargetAccessor:
        return _AllPeriodsTargetAccessor(self)

    def hpr_performance_map(
        self,
        *,
        target: BaseTargetModel,
        request: HprPerformanceMapRequest,
    ) -> HprPerformanceMap:
        """Generate one explicit map from a compatible scalar HPR target."""
        basis = build_hpr_target_map_basis(target)
        if not isinstance(request, HprPerformanceMapRequest):
            raise TypeError("request must be an HprPerformanceMapRequest")
        result = generate_hpr_performance_map(basis, request)
        if target.provenance is not None:
            from ..targeting.provenance import canonical_json

            provenance = target.provenance.model_copy(
                update={
                    "method_id": "target.hpr_performance_map",
                    "effective_settings_json": canonical_json(
                        {
                            "target": dict(target.provenance.effective_settings),
                            "request": request.model_dump(mode="json"),
                        }
                    ),
                    "prerequisite_ids": (target.provenance.identity,),
                }
            )
            result = result.model_copy(
                update={
                    "provenance": {
                        **result.provenance,
                        "analysis": provenance.model_dump(mode="json"),
                    }
                }
            )
        return result

    def _runtime(
        self,
        *,
        options: Mapping[str, Any] | None,
        period_id: str | None,
        configuration: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime, option_config = split_runtime_and_configuration_options(options)
        option_config.update(dict(configuration or {}))
        if period_id is not None:
            runtime["period_id"] = period_id
        return runtime, option_config

    @target_transaction
    def _execute(
        self,
        *,
        surface: str,
        target_id: str,
        zone: str | Zone | None,
        options: Mapping[str, Any] | None,
        configuration: Mapping[str, Any] | None,
        include_subzones: bool,
        period_id: str | None,
        direct_service=None,
        indirect_service=None,
    ) -> BaseTargetModel:
        from ...residual_utility import residual_basis, residual_utility_service

        basis = residual_basis(self._problem)
        if basis is not None:
            if surface != "direct_heat_integration" or include_subzones:
                raise ValueError(
                    "This analysis requires original physical streams; "
                    "the case contains a frozen HPR residual."
                )
            if options:
                raise ValueError(
                    "Residual allocation uses the frozen temperature basis; "
                    "edit utility definitions instead."
                )
            direct_service = residual_utility_service(basis.data)
        runtime, config_overrides = self._runtime(
            options=options,
            period_id=period_id,
            configuration=configuration,
        )
        root = self._problem._build_execution_master_zone()
        with temporary_zone_configuration(root, config_overrides):
            result = self._problem._execute_targeting(
                target_id=target_id,
                application_zone=zone,
                options=runtime,
                include_subzones=include_subzones,
                direct_service_func=direct_service,
                indirect_service_func=indirect_service,
            )
        self._problem._record_target_run(
            surface,
            options={**config_overrides, **runtime},
            zone_name=zone.name if hasattr(zone, "name") else zone,
            include_subzones=include_subzones,
        )
        return result

    def direct_heat_integration(
        self,
        *,
        zone: str | Zone | None = None,
        include_subzones: bool = False,
        period_id: str | None = None,
        options: Mapping[str, Any] | None = None,
        base_target: BaseTargetModel | None = None,
    ) -> BaseTargetModel:
        if base_target is not None:
            residual = resolve_hpr_residual_case(
                self._problem,
                base_target,
                zone=zone,
                period_id=period_id,
                include_subzones=include_subzones,
                options=options,
            )
            return residual.target.direct_heat_integration()
        return self._execute(
            surface="direct_heat_integration",
            target_id=TargetType.DI.value,
            zone=zone,
            options=options,
            configuration=None,
            include_subzones=include_subzones,
            period_id=period_id,
            direct_service=direct_heat_integration_service,
        )

    def heat_recovery_dt_min(
        self,
        *,
        heat_recovery,
        zone=None,
        period_id=None,
    ) -> "HeatRecoveryDtMinResult":
        """Return the global dt_min corresponding to requested process recovery."""
        from ...heat_recovery_dt_min import calculate_heat_recovery_dt_min

        return calculate_heat_recovery_dt_min(
            self._problem,
            heat_recovery=heat_recovery,
            zone=zone,
            period_id=period_id,
        )

    def indirect_heat_integration(self, **kwargs) -> BaseTargetModel:
        """Run focused utility-mediated heat integration."""
        return self._indirect("indirect_heat_integration", **kwargs)

    def total_site_heat_integration(self, **kwargs) -> BaseTargetModel:
        """Run indirect heat integration for a Site Zone."""
        root = self._problem._build_execution_master_zone()
        selected = self._problem._resolve_target_zone(
            kwargs.get("zone"),
            master_zone=root,
        )
        if selected.type != ZoneType.S.value:
            raise ValueError(
                "total_site_heat_integration requires a Zone of type 'Site'; "
                "use indirect_heat_integration for other aggregate Zone types."
            )
        return self._indirect("total_site_heat_integration", **kwargs)

    def _indirect(
        self,
        surface: str,
        *,
        zone: str | Zone | None = None,
        include_subzones: bool = False,
        period_id: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> BaseTargetModel:
        return self._execute(
            surface=surface,
            target_id=TargetType.II.value,
            zone=zone,
            options=options,
            configuration=None,
            include_subzones=include_subzones,
            period_id=period_id,
            indirect_service=indirect_heat_integration_service,
        )

    def all_heat_integration(
        self,
        *,
        zone: str | Zone | None = None,
        include_subzones: bool | None = None,
        period_id: str | None = None,
        options: Mapping[str, Any] | None = None,
        base_target: BaseTargetModel | None = None,
    ) -> TargetOutput:
        """Run integration, or allocate utilities on an explicit frozen HPR basis."""
        from ...residual_utility import residual_basis

        if base_target is not None:
            residual = resolve_hpr_residual_case(
                self._problem,
                base_target,
                zone=zone,
                period_id=period_id,
                include_subzones=include_subzones,
                options=options,
            )
            return residual.target.all_heat_integration()
        is_residual = residual_basis(self._problem) is not None
        if include_subzones is None:
            include_subzones = not is_residual
        if is_residual:
            self.direct_heat_integration(
                zone=zone,
                include_subzones=include_subzones,
                period_id=period_id,
                options=options,
            )
            return deepcopy(self._problem._results)
        return self._all_heat_integration(
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
        )

    @target_transaction
    def _all_heat_integration(
        self,
        *,
        zone: str | Zone | None = None,
        include_subzones: bool = True,
        period_id: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> TargetOutput:
        runtime, config_overrides = self._runtime(
            options=options,
            period_id=period_id,
        )
        root = self._problem._build_execution_master_zone()
        selected = self._problem._resolve_target_zone(zone, master_zone=root)
        with temporary_zone_configuration(root, config_overrides):
            if include_subzones:
                result = self._problem._run_targeting_for_zone_and_subzones(
                    zone=selected,
                    direct_service_func=direct_heat_integration_service,
                    indirect_service_func=indirect_heat_integration_service,
                    options=runtime,
                )
            else:
                self._problem._execute_targeting(
                    target_id=TargetType.DI.value,
                    application_zone=selected,
                    options=runtime,
                    include_subzones=False,
                    direct_service_func=direct_heat_integration_service,
                    indirect_service_func=indirect_heat_integration_service,
                )
                result = self._problem._results
        self._problem._record_target_run(
            "all_heat_integration",
            options={**config_overrides, **runtime},
            zone_name=zone.name if hasattr(zone, "name") else zone,
            include_subzones=include_subzones,
        )
        return result

    def _hpr(
        self,
        *,
        surface: str,
        cycle: HeatPumpAndRefrigerationCycle,
        is_heat_pump: bool,
        is_utility: bool,
        is_cascade_cycle: bool | None = None,
        zone: str | Zone | None = None,
        include_subzones: bool = False,
        period_id: str | None = None,
        options: Mapping[str, Any] | None = None,
        load_fraction: float | None = None,
        load_duty: float | None = None,
        period_loads: Mapping[str, float] | None = None,
        condensers: int | None = None,
        evaporators: int | None = None,
        compressor_efficiency: float | None = None,
        expander_efficiency: float | None = None,
        minimum_approach_temperature: float | None = None,
        maximum_restarts: int | None = None,
        maximum_iterations: int | None = None,
        maximum_evaluations: int | None = None,
        extra_configuration: Mapping[str, Any] | None = None,
        simulation_backend: str | None = None,
    ) -> BaseTargetModel:
        require_available(f"target.{surface}")
        normalized_backend = (
            None
            if simulation_backend is None
            else normalize_hpr_simulation_backend(simulation_backend)
        )
        if is_cascade_cycle is not None:
            if cycle is HeatPumpAndRefrigerationCycle.CascadeCarnot:
                cycle = (
                    HeatPumpAndRefrigerationCycle.CascadeCarnot
                    if is_cascade_cycle
                    else HeatPumpAndRefrigerationCycle.ParallelCarnot
                )
            elif cycle is HeatPumpAndRefrigerationCycle.CascadeVapourComp:
                cycle = (
                    HeatPumpAndRefrigerationCycle.CascadeVapourComp
                    if is_cascade_cycle
                    else HeatPumpAndRefrigerationCycle.ParallelVapourComp
                )
        configuration = {"HPR_TYPE": cycle.value}
        configuration.update(
            _load_options(
                load_fraction=load_fraction,
                load_duty=load_duty,
                period_loads=period_loads,
            )
        )
        for key, value in (
            ("HPR_N_COND", condensers),
            ("HPR_N_EVAP", evaporators),
            ("HPR_ETA_COMP", compressor_efficiency),
            ("HPR_ETA_EXP", expander_efficiency),
            ("HPR_DT_CONT", minimum_approach_temperature),
            ("HPR_MAX_MULTISTART", maximum_restarts),
        ):
            _set_if_not_none(configuration, key, value)
        configuration.update(dict(extra_configuration or {}))
        direct_service = (
            direct_heat_pump_service if is_heat_pump else direct_refrigeration_service
        )
        indirect_service = (
            indirect_heat_pump_service
            if is_heat_pump
            else indirect_refrigeration_service
        )
        target_id = (
            TargetType.IHP.value
            if is_heat_pump and is_utility
            else TargetType.IR.value
            if not is_heat_pump and is_utility
            else TargetType.DHP.value
            if is_heat_pump
            else TargetType.DR.value
        )
        runtime_options = dict(options or {})
        if maximum_iterations is not None:
            runtime_options["maximum_iterations"] = maximum_iterations
        if maximum_evaluations is not None:
            runtime_options["maximum_evaluations"] = maximum_evaluations
        if {
            "maximum_iterations",
            "maximum_evaluations",
        } & runtime_options.keys():
            budget = HPRSearchBudget(
                maximum_iterations=runtime_options.get("maximum_iterations", 300),
                maximum_evaluations=runtime_options.get(
                    "maximum_evaluations", 1_000_000
                ),
            )
            runtime_options.update(
                maximum_iterations=budget.maximum_iterations,
                maximum_evaluations=budget.maximum_evaluations,
            )
        if normalized_backend is not None:
            runtime_options["simulation_backend"] = normalized_backend
        return self._execute(
            surface=surface,
            target_id=target_id,
            zone=zone,
            options=runtime_options,
            configuration=configuration,
            include_subzones=include_subzones,
            period_id=period_id,
            direct_service=None if is_utility else direct_service,
            indirect_service=indirect_service if is_utility else None,
        )

    def carnot_heat_pump(
        self,
        *,
        is_utility_heat_pump: bool = False,
        is_cascade_cycle: bool = True,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        period_loads=None,
        condensers=None,
        evaporators=None,
        compressor_efficiency=None,
        expander_efficiency=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        return self._hpr(
            surface="carnot_heat_pump",
            cycle=HeatPumpAndRefrigerationCycle.CascadeCarnot,
            is_heat_pump=True,
            is_utility=is_utility_heat_pump,
            is_cascade_cycle=is_cascade_cycle,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            period_loads=period_loads,
            condensers=condensers,
            evaporators=evaporators,
            compressor_efficiency=compressor_efficiency,
            expander_efficiency=expander_efficiency,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
        )

    def carnot_refrigeration(
        self,
        *,
        is_utility_refrigeration: bool = False,
        is_cascade_cycle: bool = True,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        period_loads=None,
        condensers=None,
        evaporators=None,
        compressor_efficiency=None,
        expander_efficiency=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        return self._hpr(
            surface="carnot_refrigeration",
            cycle=HeatPumpAndRefrigerationCycle.CascadeCarnot,
            is_heat_pump=False,
            is_utility=is_utility_refrigeration,
            is_cascade_cycle=is_cascade_cycle,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            period_loads=period_loads,
            condensers=condensers,
            evaporators=evaporators,
            compressor_efficiency=compressor_efficiency,
            expander_efficiency=expander_efficiency,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
        )

    def vapour_compression_heat_pump(
        self,
        *,
        is_utility_heat_pump: bool = False,
        is_cascade_cycle: bool = True,
        refrigerants=None,
        initialize_from_carnot=None,
        sort_refrigerants=None,
        allow_integrated_expander=None,
        simulation_backend="coolprop",
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        period_loads=None,
        condensers=None,
        evaporators=None,
        compressor_efficiency=None,
        expander_efficiency=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        simulation_backend = normalize_hpr_simulation_backend(simulation_backend)
        extra = {}
        for key, value in (
            ("HPR_REFRIGERANTS", refrigerants),
            ("HPR_INITIALISE_SIMULATED_CYCLE", initialize_from_carnot),
            ("HPR_REFRIGERANT_SORT_ENABLED", sort_refrigerants),
            ("HPR_INTEGRATED_EXPANDER_ENABLED", allow_integrated_expander),
        ):
            _set_if_not_none(extra, key, value)
        return self._hpr(
            surface="vapour_compression_heat_pump",
            cycle=HeatPumpAndRefrigerationCycle.CascadeVapourComp,
            is_heat_pump=True,
            is_utility=is_utility_heat_pump,
            is_cascade_cycle=is_cascade_cycle,
            extra_configuration=extra,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            period_loads=period_loads,
            condensers=condensers,
            evaporators=evaporators,
            compressor_efficiency=compressor_efficiency,
            expander_efficiency=expander_efficiency,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
            simulation_backend=simulation_backend,
        )

    def vapour_compression_refrigeration(
        self,
        *,
        is_utility_refrigeration: bool = False,
        is_cascade_cycle: bool = True,
        refrigerants=None,
        initialize_from_carnot=None,
        sort_refrigerants=None,
        allow_integrated_expander=None,
        simulation_backend="coolprop",
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        period_loads=None,
        condensers=None,
        evaporators=None,
        compressor_efficiency=None,
        expander_efficiency=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        simulation_backend = normalize_hpr_simulation_backend(simulation_backend)
        extra = {}
        for key, value in (
            ("HPR_REFRIGERANTS", refrigerants),
            ("HPR_INITIALISE_SIMULATED_CYCLE", initialize_from_carnot),
            ("HPR_REFRIGERANT_SORT_ENABLED", sort_refrigerants),
            ("HPR_INTEGRATED_EXPANDER_ENABLED", allow_integrated_expander),
        ):
            _set_if_not_none(extra, key, value)
        return self._hpr(
            surface="vapour_compression_refrigeration",
            cycle=HeatPumpAndRefrigerationCycle.CascadeVapourComp,
            is_heat_pump=False,
            is_utility=is_utility_refrigeration,
            is_cascade_cycle=is_cascade_cycle,
            extra_configuration=extra,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            period_loads=period_loads,
            condensers=condensers,
            evaporators=evaporators,
            compressor_efficiency=compressor_efficiency,
            expander_efficiency=expander_efficiency,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
            simulation_backend=simulation_backend,
        )

    def brayton_heat_pump(
        self,
        *,
        is_utility_heat_pump: bool = False,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        compressor_efficiency=None,
        expander_efficiency=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        return self._hpr(
            surface="brayton_heat_pump",
            cycle=HeatPumpAndRefrigerationCycle.Brayton,
            is_heat_pump=True,
            is_utility=is_utility_heat_pump,
            is_cascade_cycle=None,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            compressor_efficiency=compressor_efficiency,
            expander_efficiency=expander_efficiency,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
        )

    def brayton_refrigeration(
        self,
        *,
        is_utility_refrigeration: bool = False,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        compressor_efficiency=None,
        expander_efficiency=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        return self._hpr(
            surface="brayton_refrigeration",
            cycle=HeatPumpAndRefrigerationCycle.Brayton,
            is_heat_pump=False,
            is_utility=is_utility_refrigeration,
            is_cascade_cycle=None,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            compressor_efficiency=compressor_efficiency,
            expander_efficiency=expander_efficiency,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
        )

    def mvr_heat_pump(
        self,
        *,
        is_utility_heat_pump: bool = False,
        mvr_fluids=None,
        mvr_compressor_efficiency=None,
        mvr_stages=None,
        motor_efficiency=None,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        load_fraction=None,
        load_duty=None,
        period_loads=None,
        condensers=None,
        evaporators=None,
        minimum_approach_temperature=None,
        maximum_restarts=None,
        maximum_iterations=None,
        maximum_evaluations=None,
    ):
        extra = {}
        for key, value in (
            ("HPR_MVR_FLUIDS", mvr_fluids),
            ("HPR_MVR_ETA_COMP", mvr_compressor_efficiency),
            ("HPR_MVR_COUNT", mvr_stages),
            ("HPR_MVR_ETA_MOTOR", motor_efficiency),
        ):
            _set_if_not_none(extra, key, value)
        return self._hpr(
            surface="mvr_heat_pump",
            cycle=HeatPumpAndRefrigerationCycle.VapourCompMVR,
            is_heat_pump=True,
            is_utility=is_utility_heat_pump,
            is_cascade_cycle=None,
            extra_configuration=extra,
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            load_fraction=load_fraction,
            load_duty=load_duty,
            period_loads=period_loads,
            condensers=condensers,
            evaporators=evaporators,
            minimum_approach_temperature=minimum_approach_temperature,
            maximum_restarts=maximum_restarts,
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
        )

    def _orc(
        self,
        *,
        surface: str,
        model: str,
        zone,
        include_subzones,
        period_id,
        options,
        configuration: dict[str, Any],
        maximum_iterations,
    ):
        require_available(f"target.{surface}")
        runtime_options = dict(options or {})
        configuration = {"ORC_MODEL": model, **configuration}
        if maximum_iterations is not None:
            if int(maximum_iterations) < 1:
                raise ValueError("maximum_iterations must be at least 1.")
            runtime_options["maximum_iterations"] = int(maximum_iterations)
        return self._execute(
            surface=surface,
            target_id=TargetType.DORC.value,
            zone=zone,
            options=runtime_options,
            configuration=configuration,
            include_subzones=include_subzones,
            period_id=period_id,
            direct_service=direct_orc_service,
        )

    @staticmethod
    def _orc_configuration(**values) -> dict[str, Any]:
        keys = {
            "stages": "ORC_N_STAGES",
            "second_law_efficiency": "ORC_ETA_II_CARNOT",
            "condensing_temperature": "ORC_T_COND",
            "minimum_lift": "ORC_MIN_LIFT",
            "minimum_approach_temperature": "ORC_DT_CONT",
            "load_fraction": "ORC_LOAD_FRACTION",
            "maximum_restarts": "ORC_MAX_MULTISTART",
            "fluids": "ORC_FLUIDS",
            "turbine_efficiency": "ORC_ETA_TURBINE",
            "pump_efficiency": "ORC_ETA_PUMP",
            "maximum_superheat": "ORC_MAX_SUPERHEAT",
            "recuperator": "ORC_RECUPERATOR_ENABLED",
            "recuperator_approach_temperature": "ORC_DT_RECUPERATOR",
        }
        configuration: dict[str, Any] = {}
        for name, value in values.items():
            if name == "fluids" and isinstance(value, str):
                value = [value]
            _set_if_not_none(configuration, keys[name], value)
        return configuration

    def carnot_orc(
        self,
        *,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        stages=None,
        second_law_efficiency=None,
        condensing_temperature=None,
        minimum_lift=None,
        minimum_approach_temperature=None,
        load_fraction=None,
        maximum_restarts=None,
        maximum_iterations=None,
    ):
        """Target an organic Rankine cycle on the surplus below the pinch.

        Parallel ORC units (``stages``, default 1) take heat from the
        process's grand composite curve below the pinch and condense at
        ``condensing_temperature`` (degC). Each unit's power is
        ``second_law_efficiency`` times its Carnot power. The design
        minimises the total annual cost change: annualised ORC capital, less
        the value of the power, plus the change in cooling. Hot utility is
        unchanged. Returns ``None`` when there is no surplus below the pinch.
        """
        return self._orc(
            surface="carnot_orc",
            model="carnot",
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            configuration=self._orc_configuration(
                stages=stages,
                second_law_efficiency=second_law_efficiency,
                condensing_temperature=condensing_temperature,
                minimum_lift=minimum_lift,
                minimum_approach_temperature=minimum_approach_temperature,
                load_fraction=load_fraction,
                maximum_restarts=maximum_restarts,
            ),
            maximum_iterations=maximum_iterations,
        )

    def organic_rankine_cycle(
        self,
        *,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        fluids=None,
        stages=None,
        turbine_efficiency=None,
        pump_efficiency=None,
        maximum_superheat=None,
        recuperator=None,
        recuperator_approach_temperature=None,
        condensing_temperature=None,
        minimum_lift=None,
        minimum_approach_temperature=None,
        load_fraction=None,
        maximum_restarts=None,
        maximum_iterations=None,
    ):
        """Target a simulated (CoolProp) ORC on the surplus below the pinch.

        Like :meth:`carnot_orc`, but each unit is a subcritical Rankine cycle
        with a pump, an evaporator that preheats, evaporates and optionally
        superheats, a turbine, an optional ``recuperator`` and a condenser.
        Each of ``fluids`` (default ``ORC_FLUIDS``) is searched in turn, all
        units using it, starting from the Carnot design, and the cheapest is
        kept. Evaporation stays 5 K below the fluid's critical temperature.
        """
        return self._orc(
            surface="organic_rankine_cycle",
            model="simulated",
            zone=zone,
            include_subzones=include_subzones,
            period_id=period_id,
            options=options,
            configuration=self._orc_configuration(
                fluids=fluids,
                stages=stages,
                turbine_efficiency=turbine_efficiency,
                pump_efficiency=pump_efficiency,
                maximum_superheat=maximum_superheat,
                recuperator=recuperator,
                recuperator_approach_temperature=recuperator_approach_temperature,
                condensing_temperature=condensing_temperature,
                minimum_lift=minimum_lift,
                minimum_approach_temperature=minimum_approach_temperature,
                load_fraction=load_fraction,
                maximum_restarts=maximum_restarts,
            ),
            maximum_iterations=maximum_iterations,
        )

    def heat_exchanger_area_and_cost(
        self,
        *,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        utility_price=None,
        annual_operating_hours=None,
        exchanger_fixed_cost=None,
        area_cost_coefficient=None,
        area_cost_exponent=None,
        discount_rate=None,
        service_life_years=None,
    ):
        configuration = {}
        for key, value in (
            ("COSTING_UTILITY_PRICE", utility_price),
            ("COSTING_ANNUAL_OP_TIME", annual_operating_hours),
            ("COSTING_HX_UNIT_COST", exchanger_fixed_cost),
            ("COSTING_HX_AREA_COEFF", area_cost_coefficient),
            ("COSTING_HX_AREA_EXP", area_cost_exponent),
            ("COSTING_DISCOUNT_RATE", discount_rate),
            ("COSTING_SERVICE_LIFE", service_life_years),
        ):
            _set_if_not_none(configuration, key, value)
        runtime = dict(options or {})
        runtime["_calculate_area_cost"] = True
        return self._execute(
            surface="heat_exchanger_area_and_cost",
            target_id=TargetType.DI.value,
            zone=zone,
            options=runtime,
            configuration=configuration,
            include_subzones=include_subzones,
            period_id=period_id,
            direct_service=area_cost_targeting_service,
        )

    @target_transaction
    def _cogeneration(
        self,
        surface,
        model,
        *,
        efficiency=None,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        base_target=None,
    ):
        configuration = {"POWER_TURB_MODEL": model.value}
        _set_if_not_none(configuration, "POWER_MIN_EFF", efficiency)
        runtime = dict(options or {})
        if base_target is not None:
            runtime["base_target_type"] = validate_base_target(
                self._problem,
                base_target,
                zone=zone,
                period_id=period_id,
                options=options,
            )
        root = self._problem._build_execution_master_zone()
        runtime, option_config = self._runtime(
            options=runtime, period_id=period_id, configuration=configuration
        )
        with temporary_zone_configuration(root, option_config):
            result = self._problem._execute_cogeneration_targeting(
                application_zone=zone,
                options=runtime,
                include_subzones=include_subzones,
                service_func=power_cogeneration_service,
            )
        self._problem._record_target_run(
            surface,
            options={**option_config, **runtime},
            zone_name=zone.name if hasattr(zone, "name") else zone,
            include_subzones=include_subzones,
        )
        return result

    def cogeneration(self, **kwargs):
        return self._cogeneration("cogeneration", TurbineModel.MEDINA_FLORES, **kwargs)

    def sun_smith_cogeneration(self, **kwargs):
        return self._cogeneration(
            "sun_smith_cogeneration", TurbineModel.SUN_SMITH, **kwargs
        )

    def varbanov_cogeneration(self, **kwargs):
        return self._cogeneration(
            "varbanov_cogeneration", TurbineModel.VARBANOV, **kwargs
        )

    def isentropic_cogeneration(self, *, efficiency, **kwargs):
        if not 0.0 < float(efficiency) <= 1.0:
            raise ValueError("efficiency must be greater than 0 and at most 1.")
        return self._cogeneration(
            "isentropic_cogeneration",
            TurbineModel.ISENTROPIC,
            efficiency=efficiency,
            **kwargs,
        )

    @target_transaction
    def exergy(
        self,
        *,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        base_target=None,
    ):
        runtime = dict(options or {})
        if base_target is not None:
            runtime["base_target_type"] = validate_base_target(
                self._problem,
                base_target,
                zone=zone,
                period_id=period_id,
                options=options,
            )
        runtime, config = self._runtime(options=runtime, period_id=period_id)
        root = self._problem._build_execution_master_zone()
        with temporary_zone_configuration(root, config):
            result = self._problem._execute_exergy_targeting(
                application_zone=zone,
                options=runtime,
                include_subzones=include_subzones,
                service_func=exergy_targeting_service,
            )
        self._problem._record_target_run(
            "exergy",
            options={**config, **runtime},
            zone_name=zone.name if hasattr(zone, "name") else zone,
            include_subzones=include_subzones,
        )
        return result

    def energy_transfer(
        self,
        *,
        zone=None,
        include_subzones=False,
        period_id=None,
        options=None,
        base_target=None,
    ):
        runtime = dict(options or {})
        if base_target is not None:
            selected, period_id = resolve_target_selection(
                self._problem,
                base_target,
                zone=zone,
                period_id=period_id,
                options=options,
            )
            zone = selected.address
            runtime["base_target_type"] = validate_base_target(
                self._problem,
                base_target,
                zone=zone,
                period_id=period_id,
                options=options,
            )
        return self._execute(
            surface="energy_transfer",
            target_id=TargetType.ET.value,
            zone=zone,
            options=runtime,
            configuration=None,
            include_subzones=include_subzones,
            period_id=period_id,
            direct_service=energy_transfer_analysis_service,
        )

    def utility_placement(
        self,
        *,
        isothermal: int | None = None,
        sensible: int | None = None,
        zone=None,
        period_ids=None,
        maximum_duties=None,
        options=None,
        base_target: BaseTargetModel | None = None,
    ):
        """Return a solved detached case containing the best utility set."""
        from ...utility_placement import run_problem_utility_placement

        if base_target is not None:
            residual = resolve_hpr_residual_case(
                self._problem,
                base_target,
                zone=zone,
                period_ids=period_ids,
            )
            return residual.target.utility_placement(
                isothermal=isothermal,
                sensible=sensible,
                maximum_duties=maximum_duties,
                options=options,
            )
        return run_problem_utility_placement(
            self._problem,
            isothermal=isothermal,
            sensible=sensible,
            zone=zone,
            period_ids=period_ids,
            maximum_duties=maximum_duties,
            options=options,
        )


class _TargetAccessorDescriptor:
    """Non-data descriptor exposing the explicit target accessor on instances."""

    def __get__(self, obj: "PinchProblem | None", owner=None):
        if obj is None:
            return self
        return _TargetAccessor(obj)
