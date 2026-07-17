"""Dependency-aware targeting traversal derived from zone structure."""

from __future__ import annotations

from collections.abc import Callable

from ....domain.enums import ZT
from ....domain.zone import Zone

__all__ = ["run_targeting_for_zone_and_subzones"]

_DIRECT_ZONE_TYPES = frozenset({ZT.S.value, ZT.P.value, ZT.O.value})
_INDIRECT_ZONE_TYPES = frozenset({ZT.R.value, ZT.C.value, ZT.S.value, ZT.P.value})
_KNOWN_ZONE_TYPES = _DIRECT_ZONE_TYPES | _INDIRECT_ZONE_TYPES | {ZT.U.value}


def run_targeting_for_zone_and_subzones(
    zone: Zone,
    direct_service_func: Callable | None = None,
    indirect_service_func: Callable | None = None,
    args: dict | None = None,
):
    """Run child prerequisites before direct and aggregate target services."""
    if zone.type not in _KNOWN_ZONE_TYPES:
        raise ValueError("No valid zone passed into OpenPinch for analysis.")

    for subzone in zone.subzones.values():
        run_targeting_for_zone_and_subzones(
            subzone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            args=args,
        )

    if callable(direct_service_func) and zone.type in _DIRECT_ZONE_TYPES:
        direct_service_func(zone, args)
    if callable(indirect_service_func) and zone.type in _INDIRECT_ZONE_TYPES:
        indirect_service_func(zone, args)
    return zone
