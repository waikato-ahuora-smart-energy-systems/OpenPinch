"""Utility routines for estimating turbine cogeneration targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...lib.config import T_CRIT, Configuration, tol
from ...lib.enums import CogenerationTarget
from ...utils.water_properties import psat_T
from ..common.miscellaneous import get_state_index
from ..power_cogeneration.unit_models.multi_stage_steam_turbine import (
    MultiStageSteamTurbine,
)

if TYPE_CHECKING:
    import numpy as np

    from ...classes.stream import Stream

__all__ = [
    "get_power_cogeneration_above_pinch",
    "get_power_cogeneration_below_pinch",
]


def get_power_cogeneration_above_pinch(
    target: CogenerationTarget,
    args: dict | None = None,
) -> CogenerationTarget:
    """Calculate above-Pinch cogeneration for a compatible thermal target object."""
    turbine_params = _prepare_turbine_parameters(target.config)
    idx, sid = get_state_index(
        state_ids=getattr(target, "state_ids", None),
        args=args,
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
    zone_config: Configuration | None = None,
    T_sink: float | None = None,
) -> tuple[float, dict]:
    """Solve a below Pinch turbine target against an environmental sink."""
    zone_config = zone_config or Configuration()
    turbine_params = _prepare_turbine_parameters(zone_config)
    sink_temperature = zone_config.T_ENV if T_sink is None else float(T_sink)

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


def _prepare_turbine_parameters(zone_config: Configuration) -> dict:
    """Load and sanitize turbine parameters from ``zone_config``."""
    return {
        "P_in": float(zone_config.TURB_P_IN),
        "T_in": float(zone_config.TURB_T_IN),
        "min_eff": float(zone_config.MIN_EFF),
        "model": zone_config.TURB_MODEL,
        "load_frac": min(max(float(zone_config.LOAD_FRACTION), 0.0), 1.0),
        "mech_eff": min(max(float(zone_config.ETA_MECH), 0.0), 1.0),
        "is_high_p_cond_flash": bool(zone_config.IS_HIGH_P_COND_FLASH),
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
        t_supply = float(u.t_supply[idx])
        t_target = float(u.t_target[idx])
        heat_flow = float(u.heat_flow[idx])
        dt_cont_act = float(u.dt_cont_act[idx])
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
