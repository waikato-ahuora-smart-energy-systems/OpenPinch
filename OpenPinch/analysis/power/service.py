"""Utility routines for estimating turbine cogeneration targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...domain.configuration import T_CRIT, Configuration, tol
from ...domain.enums import CogenerationTarget, TargetType
from ..numerics import get_period_index
from ..targeting.context import (
    apply_zone_config_overrides,
    format_selected_period_suffix,
    record_selected_period,
    target_matches_requested_period,
)
from ..thermodynamics.water import psat_T
from .steam_turbine import MultiStageSteamTurbine

if TYPE_CHECKING:
    import numpy as np

    from ...domain.stream import Stream

__all__ = [
    "get_power_cogeneration_above_pinch",
    "get_power_cogeneration_below_pinch",
    "run_power_cogeneration_service",
]

_COGENERATION_TARGET_ORDER = (
    TargetType.II.value,
    TargetType.IHP.value,
    TargetType.IR.value,
    TargetType.DHP.value,
    TargetType.DR.value,
    TargetType.DI.value,
)


def get_power_cogeneration_above_pinch(
    target: CogenerationTarget,
    args: dict | None = None,
) -> CogenerationTarget:
    """Calculate above-Pinch cogeneration for a compatible thermal target object."""
    turbine_params = _prepare_turbine_parameters(target.config)
    period_ids = getattr(target, "period_ids", None)
    period_args = dict(args or {})
    if not period_ids and "period_idx" in period_args:
        period_args.pop("period_id", None)
    idx, sid = get_period_index(
        period_ids=period_ids,
        args=period_args,
    )
    utility_data = _preprocess_utilities(target, turbine_params, idx=idx)
    if utility_data is None:
        return target

    turbine = MultiStageSteamTurbine()
    total_work, details = turbine.solve(
        utility_data["stage_temperatures"],
        utility_data["stage_heat_flows"],
        mode="above_pinch",
        T_in=turbine_params["T_in"],
        P_in=turbine_params["P_in"],
        model=turbine_params["model"],
        min_eff=turbine_params["min_eff"],
        load_frac=turbine_params["load_frac"],
        mech_eff=turbine_params["mech_eff"],
        is_high_p_cond_flash=turbine_params["is_high_p_cond_flash"],
    )

    target.work_target = total_work
    target.turbine_efficiency_target = details["overall_efficiency"]
    return target


def get_power_cogeneration_below_pinch(
    temperatures: np.ndarray,
    heat_flows: np.ndarray,
    *,
    config: Configuration | None = None,
    T_sink: float | None = None,
) -> tuple[float, dict]:
    """Solve a below Pinch turbine target against an environmental sink."""
    config = config or Configuration()
    turbine_params = _prepare_turbine_parameters(config)
    sink_temperature = (
        config.environment.temperature if T_sink is None else float(T_sink)
    )

    turbine = MultiStageSteamTurbine()
    return turbine.solve(
        temperatures,
        heat_flows,
        mode="below_pinch",
        T_sink=sink_temperature,
        model=turbine_params["model"],
        min_eff=turbine_params["min_eff"],
        load_frac=turbine_params["load_frac"],
        mech_eff=turbine_params["mech_eff"],
        is_high_p_cond_flash=turbine_params["is_high_p_cond_flash"],
    )


def run_power_cogeneration_service(
    zone,
    args: dict | None = None,
    *,
    refresh_services: dict[str, object],
    cogeneration_func=get_power_cogeneration_above_pinch,
):
    """Post-process one compatible target in service preference order."""
    apply_zone_config_overrides(zone, args)
    runtime_args = dict(args or {})
    explicit_target_type = _normalize_cogeneration_base_target_type(
        runtime_args.get("base_target_type")
    )
    idx, sid = record_selected_period(zone, runtime_args)
    runtime_args["period_idx"] = idx
    if sid is not None:
        runtime_args["period_id"] = sid
    compare_args = dict(args or {}) if isinstance(args, dict) else {}
    zone._selected_cogeneration_target_type = None

    for target_type in _get_cogeneration_candidate_order(explicit_target_type):
        target = _ensure_cogeneration_target(
            zone,
            target_type=target_type,
            refresh_args=runtime_args,
            compare_args=compare_args,
            refresh_services=refresh_services,
        )
        if target is None:
            if explicit_target_type is not None:
                raise RuntimeError(
                    "Cogeneration could not produce target "
                    f"{target_type!r} for zone {zone.name!r}"
                    f"{format_selected_period_suffix(runtime_args)}."
                )
            continue

        cogeneration_func(target, args=runtime_args)
        zone._selected_cogeneration_target_type = target_type
        return zone

    raise RuntimeError(
        "Cogeneration could not find a compatible target for zone "
        f"{zone.name!r}{format_selected_period_suffix(runtime_args)} "
        f"using implicit order {' -> '.join(_COGENERATION_TARGET_ORDER)}."
    )


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


def _ensure_cogeneration_target(
    zone,
    *,
    target_type: str,
    refresh_args: dict | None,
    compare_args: dict | None,
    refresh_services: dict[str, object],
):
    """Ensure one compatible target family exists for the requested state."""
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

    refresh_service(zone, refresh_args)
    refreshed_target = zone.targets.get(target_type)
    if target_matches_requested_period(
        refreshed_target,
        args=compare_args,
        period_ids=getattr(zone, "period_ids", None),
    ):
        return refreshed_target
    return None


def _prepare_turbine_parameters(config: Configuration) -> dict:
    """Load and sanitize turbine parameters from ``config``."""
    power = config.power
    return {
        "P_in": float(power.turb_p_in),
        "T_in": float(power.turb_t_in),
        "min_eff": float(power.min_eff),
        "model": power.turb_model,
        "load_frac": min(max(float(power.load_fraction), 0.0), 1.0),
        "mech_eff": min(max(float(power.eta_mech), 0.0), 1.0),
        "is_high_p_cond_flash": bool(power.high_p_cond_flash_enabled),
    }


def _preprocess_utilities(
    target: CogenerationTarget,
    turbine_params: dict,
    *,
    idx: int | None = None,
) -> dict | None:
    """Translate target hot-utility demands into turbine stage temperatures."""
    stage_temperatures: list[float] = []
    stage_heat_flows: list[float] = []
    source_indices: list[int] = []

    u: Stream
    for i, u in enumerate(target.hot_utilities):
        t_supply = float(u.supply_temperature[idx])
        t_target = float(u.target_temperature[idx])
        heat_flow = float(u.heat_flow[idx])
        dt_cont_act = float(u.effective_delta_t_contribution[idx])
        if t_supply >= T_CRIT or heat_flow <= tol:
            continue

        T_stage = (
            t_target
            if abs(t_supply - t_target) < 1.0 + tol
            else t_target + dt_cont_act * 2
        )
        if turbine_params["P_in"] + tol < psat_T(T_stage):
            continue

        stage_temperatures.append(float(T_stage))
        stage_heat_flows.append(float(heat_flow))
        source_indices.append(i)

    if not stage_temperatures:
        return None

    return {
        "stage_temperatures": np.asarray(stage_temperatures, dtype=float),
        "stage_heat_flows": np.asarray(stage_heat_flows, dtype=float),
        "source_indices": np.asarray(source_indices, dtype=int),
    }
