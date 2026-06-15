"""Recursive dispatch helpers for zone-scale targeting across subzones."""

from __future__ import annotations

from typing import Callable

from ...lib.enums import ZT
from ..zone import Zone

__all__ = ["run_targeting_for_zone_and_subzones"]


def run_targeting_for_zone_and_subzones(
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


################################################################################
# Helper functions
################################################################################


def _invoke_service(service_func: Callable, zone: Zone, args: dict | None = None):
    """Call a targeting callback with runtime options."""
    if not callable(service_func):
        return None
    return service_func(zone, args)


def _invoke_target_handler(
    handler: Callable,
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Call nested target handlers with runtime options."""
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
                        if callable(direct_service_func):
                            _invoke_service(direct_service_func, subzone, args)
                else:
                    raise ValueError(
                        "Invalid zone nesting. Unit operation zones can only "
                        "contain other operation zones."
                    )

        if callable(direct_service_func):
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
            if callable(indirect_service_func):
                _invoke_service(indirect_service_func, zone, args)

    if callable(direct_service_func):
        _invoke_service(direct_service_func, zone, args)

    return zone


def _get_site_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Run site targeting over the full nested zone hierarchy."""

    if callable(direct_service_func):
        _invoke_service(direct_service_func, zone, args)

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

        if callable(indirect_service_func):
            _invoke_service(indirect_service_func, zone, args)

    return zone


def _get_community_targets(
    zone: Zone,
    direct_service_func: Callable = None,
    indirect_service_func: Callable = None,
    args: dict | None = None,
):
    """Target a community zone by dispatching each site child."""
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
    """Target a regional zone by dispatching each community child."""
    for subzone in zone.subzones.values():
        subzone = _invoke_target_handler(
            _get_community_targets,
            subzone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            args=args,
        )
    return zone


_TARGET_HANDLERS = {
    ZT.R.value: _get_regional_targets,
    ZT.C.value: _get_community_targets,
    ZT.S.value: _get_site_targets,
    ZT.P.value: _get_process_targets,
    ZT.O.value: _get_unit_operation_targets,
}
