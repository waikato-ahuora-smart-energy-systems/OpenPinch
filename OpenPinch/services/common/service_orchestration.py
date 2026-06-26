"""Shared helpers for service-layer orchestration."""

from __future__ import annotations

from ...classes.zone import Zone
from .miscellaneous import get_period_index


def record_selected_period(zone: Zone, args: dict | None) -> tuple[int, str | None]:
    """Persist the selected period metadata on a prepared zone."""
    idx, sid = get_period_index(period_ids=getattr(zone, "period_ids", None), args=args)
    zone._selected_period_id = sid
    zone._selected_period_idx = idx
    return idx, sid


def target_matches_requested_period(
    target,
    *,
    args: dict | None,
    period_ids,
) -> bool:
    """Return ``True`` when an existing target was solved for the requested period."""
    if target is None:
        return False

    idx, sid = get_period_index(period_ids=period_ids, args=args)
    target_idx = getattr(target, "period_idx", None)
    if target_idx is not None:
        return int(target_idx) == idx

    target_sid = getattr(target, "period_id", None)
    if target_sid is not None or sid is not None:
        return target_sid == sid

    if not isinstance(args, dict):
        return True
    return "period_idx" not in args and "period_id" not in args


def apply_zone_config_overrides(zone: Zone, args: dict | None) -> None:
    """Reject broad runtime config overrides at service boundaries."""
    if not isinstance(args, dict):
        return

    allowed_runtime_keys = {"period_idx", "period_id", "base_target_type"}
    invalid_keys = sorted(
        str(key) for key in args if str(key) not in allowed_runtime_keys
    )
    if invalid_keys:
        raise ValueError(
            "Runtime options may only contain execution context keys: "
            + ", ".join(sorted(allowed_runtime_keys))
            + ". Invalid key(s): "
            + ", ".join(invalid_keys)
            + "."
        )

    for key, value in args.items():
        if str(key) in allowed_runtime_keys:
            continue


def format_selected_period_suffix(args: dict | None) -> str:
    """Render the selected period into service error messages."""
    if not isinstance(args, dict):
        return ""
    if args.get("period_id") is not None:
        return f" for period_id {str(args['period_id'])!r}"
    if args.get("period_idx") is not None:
        return f" for period_idx {int(args['period_idx'])}"
    return ""
