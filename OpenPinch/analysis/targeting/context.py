"""Shared helpers for service-layer orchestration."""

from __future__ import annotations

from copy import deepcopy

from ...domain.zone import Zone
from ..numerics import get_period_index


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
    config=None,
) -> bool:
    """Return ``True`` when an existing target was solved for the requested period."""
    if target is None:
        return False
    if config is not None:
        previous = getattr(getattr(target, "config", None), "_values", {})
        current = config._values
        prefixes = ("THERMAL_", "DIRECT_", "INPUT_UNIT_", "OUTPUT_UNIT_", "COSTING_")
        if any(
            previous.get(key) != value
            for key, value in current.items()
            if key.startswith(prefixes)
        ):
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

    allowed_runtime_keys = {
        "period_idx",
        "period_id",
        "base_target_type",
        "_calculate_area_cost",
        "_prepared_direct_profiles",
        "maximum_iterations",
        "maximum_evaluations",
        "simulation_backend",
    }
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


def normalize_base_target_type(
    base_target_type: object | None,
    supported: tuple[str, ...],
    *,
    service: str,
) -> str | None:
    """Validate an explicit base-target override against ``supported``.

    ``service`` names the calling analysis in the error message, e.g.
    ``"exergy"`` renders ``"Unsupported exergy base_target_type ..."``.
    """
    if base_target_type is None:
        return None

    normalized = str(base_target_type)
    if normalized not in supported:
        supported_text = ", ".join(supported)
        raise ValueError(
            f"Unsupported {service} base_target_type "
            f"{normalized!r}. Supported types: {supported_text}."
        )
    return normalized


def prepare_enrichment_run(
    zone: Zone,
    args: dict | None,
    supported: tuple[str, ...],
    *,
    service: str,
) -> tuple[dict, dict, str | None]:
    """Validate args and record the selected period for a target-enrichment service.

    Returns ``(runtime_args, compare_args, explicit_target_type)`` where
    ``runtime_args`` carries the resolved period and ``compare_args`` is a copy of
    the caller's args used to match existing targets.
    """
    apply_zone_config_overrides(zone, args)
    runtime_args = dict(args or {})
    explicit_target_type = normalize_base_target_type(
        runtime_args.get("base_target_type"), supported, service=service
    )
    idx, sid = record_selected_period(zone, runtime_args)
    runtime_args["period_idx"] = idx
    if sid is not None:
        runtime_args["period_id"] = sid
    compare_args = dict(args or {}) if isinstance(args, dict) else {}
    return runtime_args, compare_args, explicit_target_type


def clone_target_with_zone_settings(target, zone: Zone, *, prefix: str):
    """Deep-copy ``target`` (sharing its parent zone) and apply ``zone``'s config
    values whose keys start with ``prefix``.
    """
    target = deepcopy(target, {id(target.parent_zone): target.parent_zone})
    target.config.update_values(
        **{
            key: value
            for key, value in zone.config._values.items()
            if key.startswith(prefix)
        }
    )
    return target
