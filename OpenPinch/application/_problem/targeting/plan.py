"""Targeting selector registry for the default problem targeting entry point."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ....domain.enums import TT, ZT
from ...targeting import (
    area_cost_targeting_service,
    direct_heat_integration_service,
    direct_heat_pump_service,
    direct_refrigeration_service,
    exergy_targeting_service,
    indirect_heat_integration_service,
    indirect_heat_pump_service,
    indirect_refrigeration_service,
    power_cogeneration_service,
)

TargetingServiceSlot = Literal["direct", "indirect"]


@dataclass(frozen=True)
class TargetingMethodSpec:
    """Describe one flat targeting selector and its runtime dispatch service."""

    selector: str
    service: Callable
    target_type: str
    slot: TargetingServiceSlot
    prerequisites: tuple[str, ...]
    zone_applicability: tuple[str, ...]
    execution_order: int


@dataclass(frozen=True)
class TargetingPlan:
    """Ordered targeting services selected from ``config.targeting``."""

    specs: tuple[TargetingMethodSpec, ...]

    @property
    def direct_services(self) -> tuple[Callable, ...]:
        return _unique_services(
            spec.service for spec in self.specs if spec.slot == "direct"
        )

    @property
    def indirect_services(self) -> tuple[Callable, ...]:
        return _unique_services(
            spec.service for spec in self.specs if spec.slot == "indirect"
        )

    def composite_direct_service(self) -> Callable | None:
        return _compose_services(spec for spec in self.specs if spec.slot == "direct")

    def composite_indirect_service(self) -> Callable | None:
        return _compose_services(spec for spec in self.specs if spec.slot == "indirect")


@dataclass(frozen=True)
class _TargetRunSpec:
    """Immutable targeting intent that can be replayed for another period."""

    surface: str
    options: dict[str, object]
    zone_name: str | None = None
    include_subzones: bool = False


TARGETING_METHOD_SPECS: tuple[TargetingMethodSpec, ...] = (
    TargetingMethodSpec(
        selector="TARGETING_DIRECT_SITE_ENABLED",
        service=direct_heat_integration_service,
        target_type=TT.DI.value,
        slot="direct",
        prerequisites=(),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=10,
    ),
    TargetingMethodSpec(
        selector="TARGETING_DIRECT_OPERATION_ENABLED",
        service=direct_heat_integration_service,
        target_type=TT.DI.value,
        slot="direct",
        prerequisites=(),
        zone_applicability=(ZT.O.value,),
        execution_order=20,
    ),
    TargetingMethodSpec(
        selector="TARGETING_INDIRECT_PROCESS_ENABLED",
        service=indirect_heat_integration_service,
        target_type=TT.TS.value,
        slot="indirect",
        prerequisites=(TT.DI.value,),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=30,
    ),
    TargetingMethodSpec(
        selector="TARGETING_PROCESS_HP_ENABLED",
        service=direct_heat_pump_service,
        target_type=TT.DHP.value,
        slot="direct",
        prerequisites=(TT.DI.value,),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=40,
    ),
    TargetingMethodSpec(
        selector="TARGETING_PROCESS_RFRG_ENABLED",
        service=direct_refrigeration_service,
        target_type=TT.DR.value,
        slot="direct",
        prerequisites=(TT.DI.value,),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=50,
    ),
    TargetingMethodSpec(
        selector="TARGETING_UTILITY_HP_ENABLED",
        service=indirect_heat_pump_service,
        target_type=TT.IHP.value,
        slot="indirect",
        prerequisites=(TT.TS.value,),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=60,
    ),
    TargetingMethodSpec(
        selector="TARGETING_UTILITY_RFRG_ENABLED",
        service=indirect_refrigeration_service,
        target_type=TT.IR.value,
        slot="indirect",
        prerequisites=(TT.TS.value,),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=70,
    ),
    TargetingMethodSpec(
        selector="TARGETING_TURBINE_ENABLED",
        service=power_cogeneration_service,
        target_type="Cogeneration",
        slot="direct",
        prerequisites=(TT.DI.value, TT.TS.value),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=80,
    ),
    TargetingMethodSpec(
        selector="TARGETING_EXERGY_ENABLED",
        service=exergy_targeting_service,
        target_type="Exergy Analysis",
        slot="direct",
        prerequisites=(TT.DI.value, TT.TS.value),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=90,
    ),
    TargetingMethodSpec(
        selector="TARGETING_AREA_COST_ENABLED",
        service=area_cost_targeting_service,
        target_type=TT.DI.value,
        slot="direct",
        prerequisites=(TT.DI.value,),
        zone_applicability=(ZT.S.value, ZT.P.value),
        execution_order=100,
    ),
)


def build_targeting_plan(config) -> TargetingPlan:
    """Build the default service plan from ``Configuration.targeting`` flags."""
    selected = [
        spec
        for spec in TARGETING_METHOD_SPECS
        if _selector_enabled(config, spec.selector)
    ]
    return TargetingPlan(
        specs=tuple(sorted(selected, key=lambda spec: spec.execution_order))
    )


def _selector_enabled(config, selector: str) -> bool:
    field_name = selector.removeprefix("TARGETING_").removesuffix("_ENABLED").lower()
    return bool(getattr(config.targeting, f"{field_name}_enabled", False))


def _unique_services(services) -> tuple[Callable, ...]:
    unique: list[Callable] = []
    seen: set[int] = set()
    for service in services:
        marker = id(service)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(service)
    return tuple(unique)


def _compose_services(specs) -> Callable | None:
    selected = tuple(specs)
    if not selected:
        return None

    def _composite(zone, args=None):
        zone_type = getattr(zone, "type", None)
        invoked_services: set[int] = set()
        for spec in selected:
            if zone_type not in spec.zone_applicability:
                continue
            marker = id(spec.service)
            if marker in invoked_services:
                continue
            invoked_services.add(marker)
            spec.service(zone, args)
        return zone

    _composite._openpinch_zone_applicability = frozenset(
        zone_type for spec in selected for zone_type in spec.zone_applicability
    )
    return _composite
