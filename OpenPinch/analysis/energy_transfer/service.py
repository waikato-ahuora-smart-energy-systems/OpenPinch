"""Application-facing orchestration for energy-transfer analysis."""

from __future__ import annotations

from typing import Any

from ..targeting.context import (
    apply_zone_config_overrides,
    format_selected_period_suffix,
    record_selected_period,
)
from .diagram import compute_energy_transfer_target
from .selection import (
    ENERGY_TRANSFER_TARGET_ORDER,
    candidate_order,
    ensure_base_target,
    normalize_base_target_type,
    source_targets,
)


def run_energy_transfer_analysis_service(
    zone,
    args: dict | None = None,
    *,
    refresh_services: dict[str, Any],
    compute_func=compute_energy_transfer_target,
):
    """Select source targets and attach one energy-transfer result."""
    apply_zone_config_overrides(zone, args)
    runtime_args = dict(args or {})
    explicit_target_type = normalize_base_target_type(
        runtime_args.get("base_target_type")
    )
    index, period_id = record_selected_period(zone, runtime_args)
    runtime_args["period_idx"] = index
    if period_id is not None:
        runtime_args["period_id"] = period_id
    compare_args = dict(args or {}) if isinstance(args, dict) else {}
    zone._selected_energy_transfer_base_target_type = None

    for target_type in candidate_order(zone, explicit_target_type):
        target = ensure_base_target(
            zone,
            target_type=target_type,
            refresh_args=runtime_args,
            compare_args=compare_args,
            refresh_services=refresh_services,
        )
        if target is None:
            if explicit_target_type is not None:
                raise RuntimeError(
                    "Energy transfer analysis could not produce base target "
                    f"{target_type!r} for zone {zone.name!r}"
                    f"{format_selected_period_suffix(runtime_args)}."
                )
            continue
        zone.add_target(
            compute_func(
                target,
                source_targets=source_targets(zone, target_type),
            )
        )
        zone._selected_energy_transfer_base_target_type = target_type
        return zone
    raise RuntimeError(
        "Energy transfer analysis could not find a compatible target for zone "
        f"{zone.name!r}{format_selected_period_suffix(runtime_args)} "
        f"using implicit order {' -> '.join(ENERGY_TRANSFER_TARGET_ORDER)}."
    )


__all__ = ["run_energy_transfer_analysis_service"]
