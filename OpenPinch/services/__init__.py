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
from .input_data_processing.data_preparation import prepare_problem

if TYPE_CHECKING:
    from types import ModuleType

    from ..classes.zone import Zone

__all__ = [
    "data_preprocessing_service",
    "direct_heat_integration_service",
    "get_area_targets",
    "get_capital_cost_targets",
    "get_output_graph_data",
    "get_utility_targets",
    "indirect_heat_integration_service",
    "prepare_problem",
]

@lru_cache(maxsize=1)
def _load_services_entry_module() -> "ModuleType":
    """Load orchestration services lazily to avoid package import cycles."""
    return import_module(".services_entry", __name__)


def data_preprocessing_service(
    input_data: Any,
    project_name: str = "Site",
) -> "Zone":
    """Validate raw input payloads and construct the in-memory zone tree."""
    return _load_services_entry_module().data_preprocessing_service(
        input_data=input_data,
        project_name=project_name,
    )


def direct_heat_integration_service(zone: "Zone") -> "Zone":
    """Run direct heat integration targeting for a prepared zone."""
    return _load_services_entry_module().direct_heat_integration_service(zone)


def indirect_heat_integration_service(zone: "Zone") -> "Zone":
    """Run indirect heat integration targeting for an aggregated zone."""
    return _load_services_entry_module().indirect_heat_integration_service(zone)
