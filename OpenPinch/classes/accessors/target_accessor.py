from typing import Any, Optional, TYPE_CHECKING

from ...lib.enums import TT
from ...lib.schemas.targets import BaseTargetModel
from ...services import (
    area_cost_targeting_service,
    direct_heat_integration_service,
    direct_heat_pump_service,
    direct_refrigeration_service,
    indirect_heat_integration_service,
    indirect_heat_pump_service,
    indirect_refrigeration_service,
    power_cogeneration_service,
)

if TYPE_CHECKING:
    from ...classes.pinch_problem import PinchProblem


class _TargetAccessor:
    """Callable targeting helper that also exposes named targeting workflows."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def __call__(self):
        return self._problem._run_targeting_for_zone_and_subzones(
            zone=self._problem._master_zone,
            direct_service_func=direct_heat_integration_service,
            indirect_service_func=indirect_heat_integration_service,
        )

    def direct_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run direct integration targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.DI.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            direct_service_func=direct_heat_integration_service,
        )

    def indirect_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run indirect / total-site targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.TS.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_heat_integration_service,
        )

    def direct_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run direct heat-pump targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.DHP.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            direct_service_func=direct_heat_pump_service,
        )

    def indirect_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run indirect heat-pump targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.IHP.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_heat_pump_service,
        )

    def direct_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run direct refrigeration targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.DR.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            direct_service_func=direct_refrigeration_service,
        )

    def indirect_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run indirect refrigeration targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.IR.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_refrigeration_service,
        )

    def cogeneration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run turbine cogeneration post-processing on a compatible target."""
        target_id = TT.DI.value
        if options and "base_target_type" in options:
            target_id = str(options["base_target_type"])
        return self._problem._execute_targeting(
            target_id=target_id,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            direct_service_func=power_cogeneration_service,
        )

    def area_cost(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
    ) -> BaseTargetModel:
        """Run area and capital-cost targeting for one zone or sub-tree."""
        return self._problem._execute_targeting(
            target_id=TT.DI.value,
            application_zone=zone_name,
            options=options,
            include_subzones=include_subzones,
            direct_service_func=area_cost_targeting_service,
        )


class _TargetAccessorDescriptor:
    """Non-data descriptor exposing a callable target accessor on instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _TargetAccessor(obj)
