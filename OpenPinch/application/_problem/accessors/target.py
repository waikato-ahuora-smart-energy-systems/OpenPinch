"""Descriptive targeting workflows owned by :class:`PinchProblem`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping

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

if TYPE_CHECKING:
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
        if isinstance(workers, bool) or int(workers) < 1:
            raise ValueError("workers must be a positive integer.")
        problem = self._target._problem
        period_ids = list(problem.period_ids)
        original_zone = problem._master_zone
        original_results = problem._results
        original_spec = problem._last_target_run_spec
        baseline_zone = deepcopy(original_zone)

        def solve(period_id: str) -> TargetOutput:
            isolated = type(problem)(
                source=problem.to_problem_json(),
                project_name=problem.project_name,
            )
            method = getattr(isolated.target, method_name)
            method(period_id=period_id, **kwargs)
            return TargetOutput.model_validate(
                isolated.results.model_dump(mode="python")
            )

        try:
            if workers == 1:
                outputs = {}
                for period_id in period_ids:
                    problem._master_zone = deepcopy(baseline_zone)
                    getattr(problem.target, method_name)(period_id=period_id, **kwargs)
                    outputs[period_id] = TargetOutput.model_validate(
                        problem.results.model_dump(mode="python")
                    )
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    solved = executor.map(solve, period_ids)
                    outputs = dict(zip(period_ids, solved, strict=True))
            problem._period_results = outputs
            return outputs
        finally:
            problem._master_zone = original_zone
            problem._results = original_results
            problem._last_target_run_spec = original_spec

    def direct_heat_integration(self, *, workers: int = 1, **kwargs):
        return self._run("direct_heat_integration", workers=workers, kwargs=kwargs)

    def indirect_heat_integration(self, *, workers: int = 1, **kwargs):
        return self._run("indirect_heat_integration", workers=workers, kwargs=kwargs)

    def total_site_heat_integration(self, *, workers: int = 1, **kwargs):
        return self._run("total_site_heat_integration", workers=workers, kwargs=kwargs)

    def all_heat_integration(self, *, workers: int = 1, **kwargs):
        return self._run("all_heat_integration", workers=workers, kwargs=kwargs)

    def heat_exchanger_area_and_cost(self, *, workers: int = 1, **kwargs):
        return self._run("heat_exchanger_area_and_cost", workers=workers, kwargs=kwargs)

    def carnot_heat_pump(self, *, workers: int = 1, **kwargs):
        return self._run("carnot_heat_pump", workers=workers, kwargs=kwargs)

    def carnot_refrigeration(self, *, workers: int = 1, **kwargs):
        return self._run("carnot_refrigeration", workers=workers, kwargs=kwargs)

    def vapour_compression_heat_pump(self, *, workers: int = 1, **kwargs):
        return self._run("vapour_compression_heat_pump", workers=workers, kwargs=kwargs)

    def vapour_compression_refrigeration(self, *, workers: int = 1, **kwargs):
        return self._run(
            "vapour_compression_refrigeration", workers=workers, kwargs=kwargs
        )

    def mvr_heat_pump(self, *, workers: int = 1, **kwargs):
        return self._run("mvr_heat_pump", workers=workers, kwargs=kwargs)

    def cogeneration(self, *, workers: int = 1, **kwargs):
        return self._run("cogeneration", workers=workers, kwargs=kwargs)

    def sun_smith_cogeneration(self, *, workers: int = 1, **kwargs):
        return self._run("sun_smith_cogeneration", workers=workers, kwargs=kwargs)

    def varbanov_cogeneration(self, *, workers: int = 1, **kwargs):
        return self._run("varbanov_cogeneration", workers=workers, kwargs=kwargs)

    def isentropic_cogeneration(self, *, workers: int = 1, **kwargs):
        return self._run("isentropic_cogeneration", workers=workers, kwargs=kwargs)

    def exergy(self, *, workers: int = 1, **kwargs):
        return self._run("exergy", workers=workers, kwargs=kwargs)

    def energy_transfer(self, *, workers: int = 1, **kwargs):
        return self._run("energy_transfer", workers=workers, kwargs=kwargs)


class _TargetAccessor:
    """Explicit, discoverable targeting workflows for one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    @property
    def all_periods(self) -> _AllPeriodsTargetAccessor:
        return _AllPeriodsTargetAccessor(self)

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
    ) -> BaseTargetModel:
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
            result = self._problem._run_targeting_for_zone_and_subzones(
                zone=selected,
                direct_service_func=direct_heat_integration_service,
                indirect_service_func=indirect_heat_integration_service,
                options=runtime,
            )
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
        extra_configuration: Mapping[str, Any] | None = None,
    ) -> BaseTargetModel:
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
        return self._execute(
            surface=surface,
            target_id=target_id,
            zone=zone,
            options=options,
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
    ):
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
    ):
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
            runtime["base_target_type"] = getattr(base_target, "type", None)
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
            runtime["base_target_type"] = getattr(base_target, "type", None)
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
            runtime["base_target_type"] = getattr(base_target, "type", None)
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


class _TargetAccessorDescriptor:
    """Non-data descriptor exposing the explicit target accessor on instances."""

    def __get__(self, obj: "PinchProblem | None", owner=None):
        if obj is None:
            return self
        return _TargetAccessor(obj)
