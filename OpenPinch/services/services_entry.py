"""Service-layer orchestration helpers for prepared OpenPinch workflows."""

from ..classes.zone import Zone
from ..lib.enums import TT
from ..lib.schemas.io import TargetInput
from ..utils.miscellaneous import get_state_index
from .direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
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
from .power_cogeneration import get_power_cogeneration_above_pinch

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

_COGENERATION_TARGET_ORDER = (
    TT.TS.value,
    TT.IHP.value,
    TT.IR.value,
    TT.DHP.value,
    TT.DR.value,
    TT.DI.value,
)


def _record_selected_state(zone: Zone, args: dict | None) -> tuple[int, str | None]:
    """Persist the selected state metadata on a prepared zone."""
    idx, sid = get_state_index(state_ids=getattr(zone, "state_ids", None), args=args)
    zone._selected_state_id = sid
    zone._selected_state_idx = idx
    return idx, sid


def _target_matches_requested_state(
    target,
    *,
    args: dict | None,
    state_ids,
) -> bool:
    """Return ``True`` when an existing target was solved for the requested state."""
    if target is None:
        return False

    idx, sid = get_state_index(state_ids=state_ids, args=args)
    target_idx = getattr(target, "state_idx", None)
    if target_idx is not None:
        return int(target_idx) == idx

    target_sid = getattr(target, "state_id", None)
    if target_sid is not None or sid is not None:
        return target_sid == sid

    if not isinstance(args, dict):
        return True
    return "idx" not in args and "state_id" not in args


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


def _normalize_cogeneration_base_target_type(
    base_target_type: object | None,
) -> str | None:
    """Validate an explicit cogeneration base target override."""
    if base_target_type is None:
        return None

    normalized = str(base_target_type)
    if normalized not in _COGENERATION_TARGET_ORDER:
        supported = ", ".join(_COGENERATION_TARGET_ORDER)
        raise ValueError(
            "Unsupported cogeneration base_target_type "
            f"{normalized!r}. Supported types: {supported}."
        )
    return normalized


def _get_cogeneration_candidate_order(
    base_target_type: str | None,
) -> tuple[str, ...]:
    """Return the exact cogeneration target search order for this call."""
    if base_target_type is not None:
        return (base_target_type,)
    return _COGENERATION_TARGET_ORDER


def _get_cogeneration_refresh_services():
    """Map compatible cogeneration target families to their prerequisite service."""
    return {
        TT.DI.value: direct_heat_integration_service,
        TT.TS.value: indirect_heat_integration_service,
        TT.DHP.value: direct_heat_pump_service,
        TT.DR.value: direct_refrigeration_service,
        TT.IHP.value: indirect_heat_pump_service,
        TT.IR.value: indirect_refrigeration_service,
    }


def _ensure_cogeneration_target(
    zone: Zone,
    *,
    target_type: str,
    refresh_args: dict | None,
    compare_args: dict | None,
):
    """Ensure one compatible target family exists for the requested state."""
    target = zone.targets.get(target_type)
    if _target_matches_requested_state(
        target,
        args=compare_args,
        state_ids=getattr(zone, "state_ids", None),
    ):
        return target

    refresh_service = _get_cogeneration_refresh_services().get(target_type)
    if refresh_service is None:
        return None

    refresh_service(zone, refresh_args)
    refreshed_target = zone.targets.get(target_type)
    if _target_matches_requested_state(
        refreshed_target,
        args=compare_args,
        state_ids=getattr(zone, "state_ids", None),
    ):
        return refreshed_target
    return None


def _format_cogeneration_state_suffix(args: dict | None) -> str:
    """Render the selected state into cogeneration error messages."""
    if not isinstance(args, dict):
        return ""
    if args.get("state_id") is not None:
        return f" for state_id {str(args['state_id'])!r}"
    if args.get("idx") is not None:
        return f" for idx {int(args['idx'])}"
    return ""


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
    _apply_zone_config_overrides(zone, args)
    _record_selected_state(zone, args)
    zone.add_target(compute_direct_integration_targets(zone, args))
    return zone


def indirect_heat_integration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run indirect heat integration targeting for a prepared zone."""
    _apply_zone_config_overrides(zone, args)
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    _record_selected_state(zone, args)
    zone.add_target(compute_total_subzone_utility_targets(zone, args))
    zone.add_target(compute_indirect_integration_targets(zone, args))
    return zone


def direct_heat_pump_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run direct Heat Pump targeting after ensuring a base DI target exists."""
    _apply_zone_config_overrides(zone, args)
    _record_selected_state(zone, args)
    if not _target_matches_requested_state(
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
        zone.add_target(target)
    return zone


def indirect_heat_pump_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run indirect Heat Pump targeting after ensuring a base TS target exists."""
    _apply_zone_config_overrides(zone, args)
    _record_selected_state(zone, args)
    if not _target_matches_requested_state(
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
        zone.add_target(target)
    return zone


def direct_refrigeration_service(zone: Zone, args: dict | None = None) -> Zone:
    """Run direct refrigeration targeting after ensuring a base DI target exists."""
    _apply_zone_config_overrides(zone, args)
    _record_selected_state(zone, args)
    if not _target_matches_requested_state(
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
    _apply_zone_config_overrides(zone, args)
    _record_selected_state(zone, args)
    if not _target_matches_requested_state(
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
    _apply_zone_config_overrides(zone, args)
    runtime_args = dict(args or {})
    explicit_target_type = _normalize_cogeneration_base_target_type(
        runtime_args.get("base_target_type")
    )
    idx, sid = _record_selected_state(zone, runtime_args)
    runtime_args["idx"] = idx
    if sid is not None:
        runtime_args["state_id"] = sid
    compare_args = dict(args or {}) if isinstance(args, dict) else {}
    zone._selected_cogeneration_target_type = None

    for target_type in _get_cogeneration_candidate_order(explicit_target_type):
        target = _ensure_cogeneration_target(
            zone,
            target_type=target_type,
            refresh_args=runtime_args,
            compare_args=compare_args,
        )
        if target is None:
            if explicit_target_type is not None:
                raise RuntimeError(
                    "Cogeneration could not produce target "
                    f"{target_type!r} for zone {zone.name!r}"
                    f"{_format_cogeneration_state_suffix(runtime_args)}."
                )
            continue

        get_power_cogeneration_above_pinch(target, args=runtime_args)
        zone._selected_cogeneration_target_type = target_type
        return zone

    raise RuntimeError(
        "Cogeneration could not find a compatible target for zone "
        f"{zone.name!r}{_format_cogeneration_state_suffix(runtime_args)} "
        f"using implicit order {' -> '.join(_COGENERATION_TARGET_ORDER)}."
    )


def area_cost_targeting_service(zone: Zone, args: dict | None = None) -> Zone:
    """Refresh direct integration targets before area and cost reporting."""
    _apply_zone_config_overrides(zone, args)
    direct_heat_integration_service(zone, args)
    return zone
