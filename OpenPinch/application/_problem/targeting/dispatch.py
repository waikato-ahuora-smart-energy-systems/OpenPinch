"""Dependency-aware targeting traversal derived from zone structure."""

from __future__ import annotations

from collections.abc import Callable

from ....domain.enums import ZoneType
from ....domain.zone import Zone

__all__ = ["run_targeting_for_zone_and_subzones"]

_DIRECT_ZONE_TYPES = frozenset({ZoneType.S.value, ZoneType.P.value, ZoneType.O.value})
_INDIRECT_ZONE_TYPES = frozenset(
    {ZoneType.R.value, ZoneType.C.value, ZoneType.S.value, ZoneType.P.value}
)
_KNOWN_ZONE_TYPES = _DIRECT_ZONE_TYPES | _INDIRECT_ZONE_TYPES | {ZoneType.U.value}


def run_targeting_for_zone_and_subzones(
    zone: Zone,
    direct_service_func: Callable | None = None,
    indirect_service_func: Callable | None = None,
    args: dict | None = None,
):
    """Run child prerequisites before direct and aggregate target services."""
    if zone.type not in _KNOWN_ZONE_TYPES:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")

    child_args = args
    if args is not None and "base_target_type" in args:
        child_args = {
            key: value for key, value in args.items() if key != "base_target_type"
        }
    for subzone in zone.subzones.values():
        run_targeting_for_zone_and_subzones(
            subzone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            args=child_args,
        )

    if callable(direct_service_func) and zone.type in _DIRECT_ZONE_TYPES:
        direct_service_func(zone, args)
    if callable(indirect_service_func) and zone.type in _INDIRECT_ZONE_TYPES:
        indirect_service_func(zone, args)
    return zone
