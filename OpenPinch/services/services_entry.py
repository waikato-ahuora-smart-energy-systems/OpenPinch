"""Service-layer orchestration helpers for prepared OpenPinch workflows."""

from typing import Any

from ..classes.zone import Zone
from ..lib.schemas.io import TargetInput
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
from .power_cogeneration_analysis import get_power_cogeneration_above_pinch

__all__ = [
    "data_preprocessing_service",
    "direct_heat_integration_service",
    "indirect_heat_integration_service",
    "direct_heat_pump_service",
    "indirect_heat_pump_service",
    "direct_refrigeration_service",
    "indirect_refrigeration_service",
    "power_cogeneration_service",
    "area_cost_targeting_service",
]


def _apply_zone_config_overrides(zone: Zone, args: dict | None) -> None:
    """Apply supported runtime option overrides onto the selected zone config."""
    if not isinstance(args, dict):
        return

    for key, value in args.items():
        if key == "base_target_type":
            continue
        if not hasattr(zone.config, key):
            continue
        if key == "REFRIGERANTS" and isinstance(value, str):
            value = value.replace(";", ",").split(",")
        setattr(zone.config, key, value)


def data_preprocessing_service(
    input_data: TargetInput,
    project_name: str = "Site",
) -> Zone:
    """Validate raw input payloads and construct the in-memory zone tree."""
    return prepare_problem(
        project_name=project_name,
        streams=input_data.streams,
        utilities=input_data.utilities,
        options=input_data.options,
        zone_tree=input_data.zone_tree,
    )


def direct_heat_integration_service(zone: Zone, args: dict = {}) -> Zone:
    """Run direct heat integration targeting for a prepared zone."""
    zone.add_target(compute_direct_integration_targets(zone))
    return zone


def indirect_heat_integration_service(zone: Zone, args: dict = {}) -> Zone:
    """Run indirect heat integration targeting for a prepared zone."""
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    zone.add_target(compute_total_subzone_utility_targets(zone))
    zone.add_target(compute_indirect_integration_targets(zone))
    return zone


def direct_heat_pump_service(zone: Zone, args: dict = {}) -> Zone:
    _apply_zone_config_overrides(zone, args)
    if TT.DI.value not in zone.targets:
        direct_heat_integration_service(zone)
    zone.add_target(
        compute_direct_heat_pump_or_refrigeration_target(
            zone,
            is_heat_pumping=True,
        )
    )
    return zone


def indirect_heat_pump_service(zone: Zone, args: dict = {}) -> Zone:
    _apply_zone_config_overrides(zone, args)
    if TT.TS.value not in zone.targets:
        indirect_heat_integration_service(zone)
    zone.add_target(
        compute_indirect_heat_pump_or_refrigeration_target(
            zone,
            is_heat_pumping=True,
        )
    )
    return zone


def direct_refrigeration_service(zone: Zone, args: dict = {}) -> Zone:
    _apply_zone_config_overrides(zone, args)
    if TT.DI.value not in zone.targets:
        direct_heat_integration_service(zone)
    zone.add_target(
        compute_direct_heat_pump_or_refrigeration_target(
            zone,
            is_heat_pumping=False,
        )
    )
    return zone


def indirect_refrigeration_service(zone: Zone, args: dict = {}) -> Zone:
    _apply_zone_config_overrides(zone, args)
    if TT.TS.value not in zone.targets:
        indirect_heat_integration_service(zone)
    zone.add_target(
        compute_indirect_heat_pump_or_refrigeration_target(
            zone,
            is_heat_pumping=False,
        )
    )
    return zone


def power_cogeneration_service(zone: Zone, args: dict = {}) -> Zone:
    _apply_zone_config_overrides(zone, args)
    target_type = [
        TT.IHP.value,
        TT.IR.value,
        TT.TS.value,
        TT.DHP.value,
        TT.DR.value,
        TT.DI.value,
    ]
    if not(isinstance(args, dict)):
        args = {}
    if "base_target_type" in args:
        target_type = [str(args["base_target_type"])]
    if len(zone.targets) == 0:
        direct_heat_integration_service(zone)
    for tt in target_type:
        if tt in zone.targets: 
            get_power_cogeneration_above_pinch(zone.targets[tt])
            return zone
    raise ValueError("Load data before running pinch analysis services.")


def area_cost_targeting_service(zone: Zone, args: dict = {}) -> Zone:
    direct_heat_integration_service(zone)
    return zone
