"""Service-layer orchestration helpers for prepared OpenPinch workflows."""

from typing import Any

from ..classes.zone import Zone
from ..lib.schema import TargetInput
from ..lib.enums import TT
from .direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from .indirect_heat_integration.indirect_integration_entry import (
    compute_total_subzone_utility_targets,
    compute_indirect_integration_targets,
)
from .input_data_processing.data_preparation import prepare_problem
from .heat_pump_integration.heat_pump_and_refrigeration_entry import (
    compute_direct_heat_pump_or_refrigeration_target,
    compute_indirect_heat_pump_or_refrigeration_target,
)

__all__ = [
    "data_preprocessing_service",
    "direct_heat_integration_service",
    "indirect_heat_integration_service",
    "direct_heat_pump_service",
    "indirect_heat_pump_service",
    "direct_refrigeration_service",
    "indirect_refrigeration_service",
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
    zone.add_target(compute_direct_integration_targets(zone))
    return zone


def indirect_heat_integration_service(zone: Zone) -> Zone:
    """Run indirect heat integration targeting for a prepared zone."""
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    zone.add_target(compute_total_subzone_utility_targets(zone))
    zone.add_target(compute_indirect_integration_targets(zone))
    return zone


def direct_heat_pump_service(zone: Zone) -> Zone:
    if not TT.DI.value in zone.targets:
        direct_heat_integration_service(zone)
    if zone.config.DO_PROCESS_HP_TARGETING:
        zone.add_target(compute_direct_heat_pump_or_refrigeration_target(zone, is_heat_pumping=True))
    return zone


def indirect_heat_pump_service(zone: Zone) -> Zone:
    if not TT.TS.value in zone.targets:
        indirect_heat_integration_service(zone)
    if zone.config.DO_UTILITY_HP_TARGETING:
        zone.add_target(compute_indirect_heat_pump_or_refrigeration_target(zone, is_heat_pumping=True))
    return zone


def direct_refrigeration_service(zone: Zone) -> Zone:
    if not TT.DI.value in zone.targets:
        direct_heat_integration_service(zone)
    if zone.config.DO_PROCESS_RFRG_TARGETING:
        zone.add_target(compute_direct_heat_pump_or_refrigeration_target(zone, is_heat_pumping=False))
    return zone


def indirect_refrigeration_service(zone: Zone) -> Zone:
    if not TT.TS.value in zone.targets:
        indirect_heat_integration_service(zone)
    if zone.config.DO_UTILITY_RFRG_TARGETING:
        zone.add_target(compute_indirect_heat_pump_or_refrigeration_target(zone, is_heat_pumping=False))
    return zone
