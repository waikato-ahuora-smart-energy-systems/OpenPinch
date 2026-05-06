"""Utility routines for estimating turbine cogeneration targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...classes.multi_stage_steam_turbine import MultiStageSteamTurbine
from ...lib.config import T_CRIT, Configuration, tol
from ...utils.water_properties import psat_T

if TYPE_CHECKING:
    import numpy as np

    from ...classes.stream import Stream
    from ...classes.zone import Zone

__all__ = [
    "get_power_cogeneration_above_pinch",
    "get_power_cogeneration_below_pinch",
]


def get_power_cogeneration_above_pinch(z: Zone):
    """Calculate the power cogeneration potential above pinch for a given zone."""
    turbine_params = _prepare_turbine_parameters(z.config)
    utility_data = _preprocess_utilities(z, turbine_params)
    if utility_data is None:
        return z

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
        flash_correction=turbine_params["flash_correction"],
    )

    z.work_target = total_work
    z.turbine_efficiency_target = details["overall_efficiency"]
    return z


def get_power_cogeneration_below_pinch(
    temperatures: np.ndarray,
    heat_flows: np.ndarray,
    *,
    zone_config: Configuration | None = None,
    T_sink: float | None = None,
) -> tuple[float, dict]:
    """Solve a below-pinch turbine target against an environmental sink."""
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
        flash_correction=turbine_params["flash_correction"],
    )


def _prepare_turbine_parameters(zone_config: Configuration) -> dict:
    """Load and sanitize turbine parameters from ``zone_config``."""
    return {
        "P_in": float(zone_config.P_TURBINE_BOX),
        "T_in": float(zone_config.T_TURBINE_BOX),
        "min_eff": float(zone_config.MIN_EFF),
        "model": zone_config.COMBOBOX,
        "load_frac": min(max(float(zone_config.LOAD), 0.0), 1.0),
        "mech_eff": min(max(float(zone_config.MECH_EFF), 0.0), 1.0),
        "flash_correction": bool(
            getattr(zone_config, "CONDESATE_FLASH_CORRECTION", False)
        ),
    }


def _preprocess_utilities(z: Zone, turbine_params: dict) -> dict | None:
    """Translate hot-utility demands into turbine stage temperatures and duties."""
    stage_temperatures: list[float] = []
    stage_heat_flows: list[float] = []
    source_indices: list[int] = []

    u: Stream
    for idx, u in enumerate(z.hot_utilities):
        if u.t_supply >= T_CRIT or u.heat_flow <= tol:
            continue

        T_stage = (
            u.t_target
            if abs(u.t_supply - u.t_target) < 1.0 + tol
            else u.t_target + u.dt_cont * 2
        )
        if turbine_params["P_in"] + tol < psat_T(T_stage):
            continue

        stage_temperatures.append(float(T_stage))
        stage_heat_flows.append(float(u.heat_flow))
        source_indices.append(idx)

    if not stage_temperatures:
        return None

    return {
        "stage_temperatures": np.asarray(stage_temperatures, dtype=float),
        "stage_heat_flows": np.asarray(stage_heat_flows, dtype=float),
        "source_indices": np.asarray(source_indices, dtype=int),
    }
