"""Closed capability declarations for named analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ....domain.configuration_fields import USER_CONFIG_FIELD_SPECS
from ....domain.enums import ZoneType

PROCESS_SCOPES = (ZoneType.S.value, ZoneType.P.value, ZoneType.O.value)
AGGREGATE_SCOPES = (
    ZoneType.R.value,
    ZoneType.C.value,
    ZoneType.S.value,
    ZoneType.P.value,
)


@dataclass(frozen=True)
class AnalysisMethodSpec:
    scopes: tuple[str, ...]
    prerequisites: tuple[str, ...]
    period_behavior: str
    configuration_fields: tuple[str, ...]
    optional_dependencies: tuple[str, ...]
    result_adapter: str
    available: bool = True


def _spec(
    *,
    scopes=PROCESS_SCOPES,
    prerequisites=(),
    periods="independent",
    dependencies=(),
    adapter="thermal",
    available=True,
):
    return AnalysisMethodSpec(
        scopes,
        prerequisites,
        periods,
        tuple(sorted(USER_CONFIG_FIELD_SPECS)),
        dependencies,
        adapter,
        available,
    )


METHOD_CATALOG = MappingProxyType(
    {
        "target.direct_heat_integration": _spec(),
        "target.indirect_heat_integration": _spec(
            scopes=AGGREGATE_SCOPES, prerequisites=("direct_heat_integration",)
        ),
        "target.total_site_heat_integration": _spec(
            scopes=(ZoneType.S.value,), prerequisites=("direct_heat_integration",)
        ),
        "target.all_heat_integration": _spec(
            scopes=AGGREGATE_SCOPES + (ZoneType.O.value,)
        ),
        "target.heat_exchanger_area_and_cost": _spec(
            prerequisites=("direct_heat_integration",)
        ),
        "target.heat_recovery_dt_min": _spec(
            periods="nonmutating", adapter="inverse_dt_min"
        ),
        **{
            f"target.{name}": _spec(
                scopes=AGGREGATE_SCOPES + (ZoneType.O.value,),
                prerequisites=("direct_heat_integration", "indirect_heat_integration"),
                dependencies=("tespy (explicit backend only)",)
                if "vapour" in name
                else (),
            )
            for name in (
                "carnot_heat_pump",
                "carnot_refrigeration",
                "vapour_compression_heat_pump",
                "vapour_compression_refrigeration",
                "mvr_heat_pump",
            )
        },
        **{
            f"target.{name}": _spec(available=False, periods="unsupported")
            for name in ("brayton_heat_pump", "brayton_refrigeration")
        },
        **{
            f"target.{name}": _spec(
                scopes=AGGREGATE_SCOPES + (ZoneType.O.value,),
                prerequisites=("compatible_thermal_target",),
            )
            for name in (
                "cogeneration",
                "sun_smith_cogeneration",
                "varbanov_cogeneration",
                "isentropic_cogeneration",
                "exergy",
                "energy_transfer",
            )
        },
        "target.utility_placement": _spec(
            scopes=AGGREGATE_SCOPES + (ZoneType.O.value,),
            periods="shared",
            prerequisites=("direct_heat_integration",),
            adapter="placement",
        ),
        "target.hpr_performance_map": _spec(
            prerequisites=("scalar_hpr_target",),
            periods="nonmutating",
            dependencies=("tespy (explicit backend only)",),
            adapter="hpr_map",
        ),
        **{
            f"design.{name}": _spec(
                prerequisites=("direct_heat_integration",),
                periods="shared"
                if name == "multiperiod_heat_exchanger_network"
                else "specialized",
                dependencies=("synthesis", "external solver"),
                adapter="hen",
            )
            for name in (
                "heat_exchanger_network",
                "enhanced_heat_exchanger_network",
                "multiperiod_heat_exchanger_network",
                "network_evolution",
                "open_hens",
                "pinch_design",
                "thermal_derivative",
            )
        },
    }
)


def require_available(method: str) -> AnalysisMethodSpec:
    spec = METHOD_CATALOG[method]
    if not spec.available:
        raise NotImplementedError(
            "Brayton targeting is unavailable pending solver repair."
        )
    return spec
