"""Recursive dispatch helpers for zone-scale targeting and result extraction."""

import inspect
from typing import Callable, List

from ..classes.stream import Stream
from ..classes.stream_collection import StreamCollection
from ..classes.zone import Zone
from ..lib.enums import ZT
from ..services.common.graph_data import get_output_graph_data

__all__ = [
    "get_targets_for_zone_and_sub_zones",
    "extract_results",
]


def get_targets_for_zone_and_sub_zones(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Dispatch a prepared zone tree to the appropriate targeting routine."""
    handler = _TARGET_HANDLERS.get(zone.type)
    if handler is None:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")
    return handler(zone, direct_service_func, indirect_service_func, args)


def extract_results(zone: Zone, state_id: str | None = None) -> dict:
    """Serialise solved targets, generated utilities, and graph payloads."""
    return {
        "name": zone.name,
        "state_id": state_id,
        "targets": _get_report(zone, state_id=state_id),
        "utilities": _get_utilities(zone),
        "graphs": get_output_graph_data(zone),
    }


################################################################################
# Helper functions
################################################################################


def _invoke_service(service_func: Callable, zone: Zone, args: dict | None = None):
    """Call a targeting callback with backward-compatible argument handling."""
    if not isinstance(service_func, Callable):
        return None

    if args is None:
        return service_func(zone)

    try:
        signature = inspect.signature(service_func)
    except TypeError, ValueError:
        return service_func(zone, args)

    params = list(signature.parameters.values())
    accepts_varargs = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL for param in params
    )
    positional_params = [
        param
        for param in params
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    if accepts_varargs or len(positional_params) >= 2:
        return service_func(zone, args)
    return service_func(zone)


def _invoke_target_handler(
    handler: Callable,
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Call nested target handlers without forcing an ``args`` keyword."""
    if args is None:
        return handler(
            zone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
        )
    return handler(
        zone,
        direct_service_func=direct_service_func,
        indirect_service_func=indirect_service_func,
        args=args,
    )


def _get_unit_operation_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Populate a ``Zone`` with detailed unit operation-level pinch targets."""
    if zone.config.DO_DIRECT_OPERATION_TARGETING:
        if len(zone.subzones) > 0:
            for subzone in zone.subzones.values():
                if subzone.type == ZT.O.value:
                    if subzone.config.DO_DIRECT_OPERATION_TARGETING:
                        if isinstance(direct_service_func, Callable):
                            _invoke_service(direct_service_func, subzone, args)
                else:
                    raise ValueError(
                        "Invalid zone nesting. Unit operation zones can only "
                        "contain other operation zones."
                    )

        if isinstance(direct_service_func, Callable):
            _invoke_service(direct_service_func, zone, args)

    return zone


def _get_process_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Populate a ``Zone`` with detailed process-level pinch targets."""

    if len(zone.subzones) > 0:
        for subzone in zone.subzones.values():
            if subzone.type == ZT.O.value:
                subzone = _invoke_target_handler(
                    _get_unit_operation_targets,
                    subzone,
                    direct_service_func=direct_service_func,
                    indirect_service_func=indirect_service_func,
                    args=args,
                )
            elif subzone.type == ZT.P.value:
                subzone = _invoke_target_handler(
                    _get_process_targets,
                    subzone,
                    direct_service_func=direct_service_func,
                    indirect_service_func=indirect_service_func,
                    args=args,
                )
            else:
                raise ValueError(
                    "Invalid zone nesting. Process zones can only contain "
                    "other process zones and operation zones."
                )

        if zone.config.DO_INDIRECT_PROCESS_TARGETING:
            if isinstance(indirect_service_func, Callable):
                _invoke_service(indirect_service_func, zone, args)

    if isinstance(direct_service_func, Callable):
        _invoke_service(direct_service_func, zone, args)

    return zone


def _get_site_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Targets heat integration using Total Site Anlysis,
    by systematically analysing individual zones and then performing
    site-level indirect integration through the utility system.
    """

    # Totally integrated analysis for a site zone
    if isinstance(direct_service_func, Callable):
        _invoke_service(direct_service_func, zone, args)

    # Targets sub-zone energy requirements
    if len(zone.subzones) > 0:
        for subzone in zone.subzones.values():
            if subzone.type == ZT.O.value:
                _invoke_target_handler(
                    _get_unit_operation_targets,
                    subzone,
                    direct_service_func=direct_service_func,
                    indirect_service_func=indirect_service_func,
                    args=args,
                )
            elif subzone.type == ZT.P.value:
                _invoke_target_handler(
                    _get_process_targets,
                    subzone,
                    direct_service_func=direct_service_func,
                    indirect_service_func=indirect_service_func,
                    args=args,
                )
            elif subzone.type == ZT.S.value:
                _invoke_target_handler(
                    _get_site_targets,
                    subzone,
                    direct_service_func=direct_service_func,
                    indirect_service_func=indirect_service_func,
                    args=args,
                )
            else:
                raise ValueError(
                    "Invalid zone nesting. Sites zones can only contain "
                    "site, process, and operation zones."
                )

        # Calculates indirect targets based on different approaches
        if isinstance(indirect_service_func, Callable):
            _invoke_service(indirect_service_func, zone, args)

    return zone


def _get_community_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Targets a Community Zone."""
    for subzone in zone.subzones.values():
        subzone = _invoke_target_handler(
            _get_site_targets,
            subzone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            args=args,
        )
    return zone


def _get_regional_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Targets a Regional Zone."""
    for subzone in zone.subzones.values():
        subzone = _invoke_target_handler(
            _get_community_targets,
            subzone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            args=args,
        )
    return zone


def _get_report(zone: Zone, state_id: str | None = None) -> dict:
    """Creates the database summary of zone targets."""
    targets: List[dict] = []

    for target in zone.targets.values():
        target_payload = target.serialize_json()
        if state_id is not None:
            target_payload["state_id"] = state_id
        targets.append(target_payload)

    if len(zone.subzones) > 0:
        for subzone in zone.subzones.values():
            targets.extend(_get_report(subzone, state_id=state_id))

    return targets


def _get_utilities(zone: Zone) -> StreamCollection:
    """Gets a list of any default utilities generated during the analysis."""
    utilities: StreamCollection = zone.hot_utilities + zone.cold_utilities
    default_hu: Stream = next((u for u in utilities if u.name == "HU"), None)
    default_cu: Stream = next((u for u in utilities if u.name == "CU"), None)
    return [default_hu, default_cu]


_TARGET_HANDLERS = {
    ZT.R.value: _get_regional_targets,
    ZT.C.value: _get_community_targets,
    ZT.S.value: _get_site_targets,
    ZT.P.value: _get_process_targets,
    ZT.O.value: _get_unit_operation_targets,
}
