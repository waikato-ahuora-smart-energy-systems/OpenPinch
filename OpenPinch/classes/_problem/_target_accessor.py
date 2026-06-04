from typing import TYPE_CHECKING, Any, Optional

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
    from ..pinch_problem import PinchProblem


class _TargetAccessor:
    """Callable targeting helper that also exposes named targeting workflows."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def __call__(
        self, *, options: Optional[dict[str, Any]] = {}, state_id: Optional[str] = None
    ):
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._run_targeting_for_zone_and_subzones(
            zone=None,
            direct_service_func=direct_heat_integration_service,
            indirect_service_func=indirect_heat_integration_service,
            options=runtime_options,
        )

    def direct_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run direct integration targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.DI.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=direct_heat_integration_service,
        )

    def indirect_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run indirect / Total Site targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.TS.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_heat_integration_service,
        )

    def direct_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run direct Heat Pump targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.DHP.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=direct_heat_pump_service,
        )

    def indirect_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run indirect Heat Pump targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.IHP.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_heat_pump_service,
        )

    def direct_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run direct refrigeration targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.DR.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=direct_refrigeration_service,
        )

    def indirect_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run indirect refrigeration targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.IR.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            indirect_service_func=indirect_refrigeration_service,
        )

    def cogeneration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run cogeneration on `TS -> IHP -> IR -> DHP -> DR -> DI` unless overridden."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_cogeneration_targeting(
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            service_func=power_cogeneration_service,
        )

    def area_cost(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        include_subzones: bool = False,
        state_id: Optional[str] = None,
    ) -> BaseTargetModel:
        """Run area and capital-cost targeting for one zone or sub-tree."""
        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return self._problem._execute_targeting(
            target_id=TT.DI.value,
            application_zone=zone_name,
            options=runtime_options,
            include_subzones=include_subzones,
            direct_service_func=area_cost_targeting_service,
        )


class _TargetAccessorDescriptor:
    """Non-data descriptor exposing a callable target accessor on instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _TargetAccessor(obj)
