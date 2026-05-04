"""Service-layer orchestration helpers for prepared OpenPinch workflows."""

from typing import Any

from ..classes.zone import Zone
from ..lib.schema import TargetInput
from .direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from .indirect_heat_integration.indirect_integration_entry import (
    compute_indirect_integration_targets,
)
from .input_data_processing.data_preparation import prepare_problem

__all__ = [
    "data_preprocessing_service",
    "direct_heat_integration_service",
    "indirect_heat_integration_service",
]


def data_preprocessing_service(
    input_data: Any,
    project_name: str = "Site",
) -> Zone:
    """Validate raw input payloads and construct the in-memory zone tree."""
    input_data = TargetInput.model_validate(input_data)
    return prepare_problem(
        project_name=project_name,
        streams=input_data.streams,
        utilities=input_data.utilities,
        options=input_data.options,
        zone_tree=input_data.zone_tree,
    )


def direct_heat_integration_service(zone: Zone) -> Zone:
    """Run direct heat integration targeting for a prepared zone."""
    return compute_direct_integration_targets(zone)


def indirect_heat_integration_service(zone: Zone) -> Zone:
    """Run indirect heat integration targeting for a prepared zone."""
    return compute_indirect_integration_targets(zone)
