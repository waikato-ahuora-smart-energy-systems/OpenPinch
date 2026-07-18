"""Base-target selection for energy-transfer analysis."""

from __future__ import annotations

from typing import Any

from ...domain.enums import TargetType
from ..targeting.context import target_matches_requested_period

ENERGY_TRANSFER_TARGET_ORDER = (TargetType.II.value, TargetType.DI.value)


def normalize_base_target_type(base_target_type: object | None) -> str | None:
    """Validate an explicit energy-transfer base-target override."""
    if base_target_type is None:
        return None
    normalized = str(base_target_type)
    if normalized not in ENERGY_TRANSFER_TARGET_ORDER:
        supported = ", ".join(ENERGY_TRANSFER_TARGET_ORDER)
        raise ValueError(
            "Unsupported energy-transfer base_target_type "
            f"{normalized!r}. Supported types: {supported}."
        )
    return normalized


def candidate_order(zone, base_target_type: str | None) -> tuple[str, ...]:
    """Return the base-target search order for one zone."""
    if base_target_type is not None:
        return (base_target_type,)
    if TargetType.II.value in zone.targets or zone.subzones:
        return ENERGY_TRANSFER_TARGET_ORDER
    return (TargetType.DI.value,)


def ensure_base_target(
    zone,
    *,
    target_type: str,
    refresh_args: dict | None,
    compare_args: dict | None,
    refresh_services: dict[str, Any],
):
    """Return a matching target, refreshing the required hierarchy when stale."""
    target = zone.targets.get(target_type)
    if target_matches_requested_period(
        target,
        args=compare_args,
        period_ids=getattr(zone, "period_ids", None),
    ):
        return target
    refresh_service = refresh_services.get(target_type)
    if refresh_service is None:
        return None
    if target_type == TargetType.II.value:
        if not zone.subzones:
            return None
        direct_target = zone.targets.get(TargetType.DI.value)
        if not target_matches_requested_period(
            direct_target,
            args=compare_args,
            period_ids=getattr(zone, "period_ids", None),
        ):
            refresh_services[TargetType.DI.value](zone, refresh_args)
        for subzone in zone.subzones.values():
            subtarget = subzone.targets.get(TargetType.DI.value)
            if not target_matches_requested_period(
                subtarget,
                args=compare_args,
                period_ids=getattr(subzone, "period_ids", None),
            ):
                refresh_services[TargetType.DI.value](subzone, refresh_args)
    refresh_service(zone, refresh_args)
    refreshed_target = zone.targets.get(target_type)
    if target_matches_requested_period(
        refreshed_target,
        args=compare_args,
        period_ids=getattr(zone, "period_ids", None),
    ):
        return refreshed_target
    return None


def source_targets(zone, base_target_type: str) -> list:
    """Return operation-level targets contributing to one diagram."""
    if base_target_type == TargetType.II.value and zone.subzones:
        return [
            {"name": source_name, "target": source_zone.targets[TargetType.DI.value]}
            for source_name, source_zone in iter_source_zones(zone)
            if TargetType.DI.value in source_zone.targets
        ]
    target = zone.targets.get(base_target_type)
    return [target] if target is not None else []


def iter_source_zones(zone):
    """Yield immediate child zones whose GCCs form one Total Site layer."""
    prefix = str(zone.name)
    for subzone in zone.subzones.values():
        yield f"{prefix}/{subzone.name}", subzone
