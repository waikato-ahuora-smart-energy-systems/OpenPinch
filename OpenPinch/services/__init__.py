"""Public service-layer entry points and reusable targeting helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

from .common.capital_cost_and_area_targeting import (
    get_area_targets,
    get_capital_cost_targets,
)
from .common.graph_data import get_output_graph_data
from .common.utility_targeting import get_utility_targets

if TYPE_CHECKING:
    from types import ModuleType

    from ..classes.zone import Zone

__all__ = [
    "data_preprocessing_service",
    "direct_heat_integration_service",
    "exergy_targeting_service",
    "indirect_heat_integration_service",
    "direct_heat_pump_service",
    "indirect_heat_pump_service",
    "direct_refrigeration_service",
    "indirect_refrigeration_service",
    "power_cogeneration_service",
    "area_cost_targeting_service",
    "energy_transfer_analysis_service",
    "get_area_targets",
    "get_capital_cost_targets",
    "get_output_graph_data",
    "get_utility_targets",
]


@lru_cache(maxsize=1)
def _load_services_entry_module() -> "ModuleType":
    """Load orchestration services lazily to avoid package import cycles."""
    return import_module(".services_entry", __name__)


def data_preprocessing_service(
    input_data: Any,
    project_name: str = "Site",
) -> "Zone":
    """Validate raw input data and construct the in-memory zone tree."""
    return _load_services_entry_module().data_preprocessing_service(
        input_data=input_data,
        project_name=project_name,
    )


def direct_heat_integration_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run direct heat integration targeting for a prepared zone."""
    return _load_services_entry_module().direct_heat_integration_service(zone, args)


def exergy_targeting_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run exergy enrichment on one compatible target family."""
    return _load_services_entry_module().exergy_targeting_service(zone, args)


def indirect_heat_integration_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run indirect heat integration targeting for an aggregated zone."""
    return _load_services_entry_module().indirect_heat_integration_service(zone, args)


def direct_heat_pump_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run direct heat pump targeting for a prepared zone."""
    return _load_services_entry_module().direct_heat_pump_service(zone, args)


def indirect_heat_pump_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run indirect heat pump targeting for an aggregated zone."""
    return _load_services_entry_module().indirect_heat_pump_service(zone, args)


def direct_refrigeration_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run direct refrigeration targeting for a prepared zone."""
    return _load_services_entry_module().direct_refrigeration_service(zone, args)


def indirect_refrigeration_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run indirect refrigeration targeting for an aggregated zone."""
    return _load_services_entry_module().indirect_refrigeration_service(zone, args)


def power_cogeneration_service(zone: "Zone", args: dict = None) -> "Zone":
    """Run turbine cogeneration targeting for a prepared zone."""
    return _load_services_entry_module().power_cogeneration_service(zone, args)


def area_cost_targeting_service(zone: "Zone", args: dict = None) -> "Zone":
    """Recompute direct integration targets with area/cost targeting enabled."""
    return _load_services_entry_module().area_cost_targeting_service(zone, args)


def energy_transfer_analysis_service(zone: "Zone", args: dict = None) -> "Zone":
    """Create energy-transfer diagram and surplus/deficit table outputs."""
    return _load_services_entry_module().energy_transfer_analysis_service(zone, args)
