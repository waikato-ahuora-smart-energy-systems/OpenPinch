"""Service-layer orchestration helpers for prepared OpenPinch workflows."""

from __future__ import annotations

from ..classes.zone import Zone
from ..lib.enums import TT
from ..lib.schemas.io import TargetInput
from .common.service_orchestration import (
    apply_zone_config_overrides,
    record_selected_state,
    target_matches_requested_state,
)
from .direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from .energy_transfer_analysis.energy_transfer_entry import (
    compute_energy_transfer_target,
    run_energy_transfer_analysis_service,
)
from .exergy_analysis import (
    apply_exergy_if_enabled,
    apply_exergy_targeting,
    run_exergy_targeting_service,
)
from .heat_pump_integration.heat_pump_and_refrigeration_entry import (
    compute_direct_heat_pump_or_refrigeration_target,
    compute_indirect_heat_pump_or_refrigeration_target,
)
from .indirect_heat_integration.indirect_integration_entry import (
    compute_indirect_integration_targets,
    compute_total_subzone_utility_targets,
)
from .input_data_processing.data_preparation import prepare_problem
from .power_cogeneration import (
    get_power_cogeneration_above_pinch,
    run_power_cogeneration_service,
)

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
]


def data_preprocessing_service(
    input_data: TargetInput,
    project_name: str = "Site",
) -> Zone:
    """Validate raw input data and construct the in-memory zone tree."""
    return prepare_problem(
        project_name=project_name,
        streams=input_data.streams,
        utilities=input_data.utilities,
        options=input_data.options,
        zone_tree=input_data.zone_tree,
    )


def direct_heat_integration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run direct heat integration targeting for a prepared zone."""
    apply_zone_config_overrides(zone, args)
    record_selected_state(zone, args)
    target = compute_direct_integration_targets(zone, args)
    zone.add_target(
        apply_exergy_if_enabled(target, zone, apply_func=apply_exergy_targeting)
    )
    return zone


def indirect_heat_integration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run indirect heat integration targeting for a prepared zone."""
    apply_zone_config_overrides(zone, args)
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    record_selected_state(zone, args)
    zone.add_target(compute_total_subzone_utility_targets(zone, args))
    target = compute_indirect_integration_targets(zone, args)
    if target is not None:
        zone.add_target(
            apply_exergy_if_enabled(target, zone, apply_func=apply_exergy_targeting)
        )
    return zone


def direct_heat_pump_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run direct Heat Pump targeting after ensuring a base DI target exists."""
    apply_zone_config_overrides(zone, args)
    record_selected_state(zone, args)
    if not target_matches_requested_state(
        zone.targets.get(TT.DI.value),
        args=args,
        state_ids=zone.state_ids,
    ):
        direct_heat_integration_service(zone, args)
    target = compute_direct_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args=args,
    )
    if target is None:
        zone.targets.pop(TT.DHP.value, None)
    else:
        zone.add_target(
            apply_exergy_if_enabled(target, zone, apply_func=apply_exergy_targeting)
        )
    return zone


def indirect_heat_pump_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run indirect Heat Pump targeting after ensuring a base TS target exists."""
    apply_zone_config_overrides(zone, args)
    record_selected_state(zone, args)
    if not target_matches_requested_state(
        zone.targets.get(TT.TS.value),
        args=args,
        state_ids=zone.state_ids,
    ):
        indirect_heat_integration_service(zone, args)
    target = compute_indirect_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args=args,
    )
    if target is None:
        zone.targets.pop(TT.IHP.value, None)
    else:
        zone.add_target(
            apply_exergy_if_enabled(target, zone, apply_func=apply_exergy_targeting)
        )
    return zone


def direct_refrigeration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run direct refrigeration targeting after ensuring a base DI target exists."""
    apply_zone_config_overrides(zone, args)
    record_selected_state(zone, args)
    if not target_matches_requested_state(
        zone.targets.get(TT.DI.value),
        args=args,
        state_ids=zone.state_ids,
    ):
        direct_heat_integration_service(zone, args)
    target = compute_direct_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=False,
        args=args,
    )
    if target is None:
        zone.targets.pop(TT.DR.value, None)
    else:
        zone.add_target(target)
    return zone


def indirect_refrigeration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run indirect refrigeration targeting after ensuring a base TS target exists."""
    apply_zone_config_overrides(zone, args)
    record_selected_state(zone, args)
    if not target_matches_requested_state(
        zone.targets.get(TT.TS.value),
        args=args,
        state_ids=zone.state_ids,
    ):
        indirect_heat_integration_service(zone, args)
    target = compute_indirect_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=False,
        args=args,
    )
    if target is None:
        zone.targets.pop(TT.IR.value, None)
    else:
        zone.add_target(target)
    return zone


def power_cogeneration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Post-process one compatible target in
    TS -> IHP -> IR -> DHP -> DR -> DI order."""
    return run_power_cogeneration_service(
        zone,
        args,
        refresh_services={
            TT.DI.value: direct_heat_integration_service,
            TT.TS.value: indirect_heat_integration_service,
            TT.DHP.value: direct_heat_pump_service,
            TT.DR.value: direct_refrigeration_service,
            TT.IHP.value: indirect_heat_pump_service,
            TT.IR.value: indirect_refrigeration_service,
        },
        cogeneration_func=get_power_cogeneration_above_pinch,
    )


def exergy_targeting_service(zone: Zone, args: dict | None = None) -> Zone:
    """Enrich the first compatible existing target family with exergy outputs."""
    return run_exergy_targeting_service(
        zone,
        args,
        apply_func=apply_exergy_targeting,
    )


def area_cost_targeting_service(zone: Zone, args: dict | None = None) -> Zone:
    """Refresh direct integration targets before area and cost reporting."""
    apply_zone_config_overrides(zone, args)
    direct_heat_integration_service(zone, args)
    return zone


def energy_transfer_analysis_service(zone: Zone, args: dict | None = None) -> Zone:
    """Create the energy-transfer diagram and heat-surplus/deficit table."""
    return run_energy_transfer_analysis_service(
        zone,
        args,
        refresh_services={
            TT.DI.value: direct_heat_integration_service,
            TT.TS.value: indirect_heat_integration_service,
        },
        compute_func=compute_energy_transfer_target,
    )
