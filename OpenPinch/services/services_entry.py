"""Service-layer orchestration helpers for prepared OpenPinch workflows."""

from typing import Any

from ..classes.zone import Zone
from ..lib.schema import TargetInput
from .direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from .indirect_heat_integration.indirect_integration_entry import (
    compute_total_subzone_utility_targets,
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
    target = compute_direct_integration_targets(zone)
    zone.add_target(target)
    return zone


def indirect_heat_integration_service(zone: Zone) -> Zone:
    """Run indirect heat integration targeting for a prepared zone."""
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    tz_target = compute_total_subzone_utility_targets(zone)
    zone.add_target(tz_target)
    ts_target = compute_indirect_integration_targets(zone)
    zone.add_target(ts_target)
    return zone
