"""Shared helpers for service-layer orchestration."""

from __future__ import annotations

from ...classes.zone import Zone
from .miscellaneous import get_state_index


def record_selected_state(zone: Zone, args: dict | None) -> tuple[int, str | None]:
    """Persist the selected state metadata on a prepared zone."""
    idx, sid = get_state_index(state_ids=getattr(zone, "state_ids", None), args=args)
    zone._selected_state_id = sid
    zone._selected_state_idx = idx
    return idx, sid


def target_matches_requested_state(
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


def apply_zone_config_overrides(zone: Zone, args: dict | None) -> None:
    """Reject broad runtime config overrides at service boundaries."""
    if not isinstance(args, dict):
        return

    allowed_runtime_keys = {"idx", "state_id", "base_target_type"}
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


def format_selected_state_suffix(args: dict | None) -> str:
    """Render the selected state into service error messages."""
    if not isinstance(args, dict):
        return ""
    if args.get("state_id") is not None:
        return f" for state_id {str(args['state_id'])!r}"
    if args.get("idx") is not None:
        return f" for idx {int(args['idx'])}"
    return ""
