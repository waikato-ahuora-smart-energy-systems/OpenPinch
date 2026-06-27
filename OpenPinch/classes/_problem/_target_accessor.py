from typing import TYPE_CHECKING, Any, Optional

from ...lib.enums import TT
from ...lib.schemas.targets import BaseTargetModel
from ...services import (
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
from ._target_plan import build_targeting_plan

if TYPE_CHECKING:
    from ..pinch_problem import PinchProblem


class _TargetAccessor:
    """Callable targeting helper that also exposes named targeting workflows."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def _record_target_run(self, *args, **kwargs) -> None:
        recorder = getattr(self._problem, "_record_target_run", None)
        if callable(recorder):
            recorder(*args, **kwargs)

    def __call__(
        self,
        *,
        options: Optional[dict[str, Any]] = None,
        period_id: Optional[str] = None,
    ):
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        execution_master_zone = self._problem._build_execution_master_zone()
        plan = build_targeting_plan(execution_master_zone.config)
        result = self._problem._run_targeting_for_zone_and_subzones(
            zone=execution_master_zone,
            direct_service_func=plan.composite_direct_service(),
            indirect_service_func=plan.composite_indirect_service(),
            options=runtime_options,
        )
        self._record_target_run("default", options=runtime_options)
        return result

    def direct_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run direct integration targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.DI.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=direct_heat_integration_service,
        )
        self._record_target_run(
            "direct_heat_integration",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def indirect_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run indirect / Total Site targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.TS.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_heat_integration_service,
        )
        self._record_target_run(
            "indirect_heat_integration",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def direct_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run direct Heat Pump targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.DHP.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=direct_heat_pump_service,
        )
        self._record_target_run(
            "direct_heat_pump",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def indirect_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run indirect Heat Pump targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.IHP.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_heat_pump_service,
        )
        self._record_target_run(
            "indirect_heat_pump",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def direct_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run direct refrigeration targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.DR.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=direct_refrigeration_service,
        )
        self._record_target_run(
            "direct_refrigeration",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def indirect_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run indirect refrigeration targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.IR.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_refrigeration_service,
        )
        self._record_target_run(
            "indirect_refrigeration",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def cogeneration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """
        Run cogeneration on `TS -> IHP -> IR -> DHP -> DR -> DI`
        unless overridden.
        """
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_cogeneration_targeting(
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            service_func=power_cogeneration_service,
        )
        self._record_target_run(
            "cogeneration",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def area_cost(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run area and capital-cost targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.DI.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=area_cost_targeting_service,
        )
        self._record_target_run(
            "area_cost",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def exergy(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run exergy targeting on the first compatible base target family."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_exergy_targeting(
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            service_func=exergy_targeting_service,
        )
        self._record_target_run(
            "exergy",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result

    def energy_transfer(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        period_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Create energy-transfer diagram and heat-surplus/deficit outputs."""
        runtime_options = dict(options or {})
        if period_id is not None:
            runtime_options["period_id"] = period_id
        result = self._problem._execute_targeting(
            target_id=TT.ET.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=energy_transfer_analysis_service,
        )
        self._record_target_run(
            "energy_transfer",
            options=runtime_options,
            zone_name=zone_name,
            include_subzones=include_subzones,
        )
        return result


class _TargetAccessorDescriptor:
    """Non-data descriptor exposing a callable target accessor on instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _TargetAccessor(obj)
