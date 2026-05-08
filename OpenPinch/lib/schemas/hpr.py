"""Schemas used by heat-pump and refrigeration solvers."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from ...classes.stream_collection import StreamCollection


class HeatPumpTargetInputs(BaseModel):
    """Parameter bundle for heat pump and refrigeration targeting routines."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hpr_type: str
    Q_hpr_target: float
    Q_heat_max: float
    Q_cool_max: float
    z_amb_hot: np.ndarray
    z_amb_cold: np.ndarray
    dt_range_max: float
    T_hot: np.ndarray | list
    H_hot: np.ndarray | list
    T_cold: np.ndarray | list
    H_cold: np.ndarray | list
    n_cond: int
    n_evap: int
    eta_comp: float
    eta_exp: float
    dtcont_hp: float
    dt_hp_ihx: float
    dt_cascade_hx: float
    dt_phase_change: float
    heat_to_power_ratio: float
    cold_to_power_ratio: float
    is_heat_pumping: bool
    max_multi_start: int
    T_env: float
    dt_env_cont: float
    eta_ii_hpr_carnot: float
    eta_ii_he_carnot: float
    refrigerant_ls: List[str]
    do_refrigerant_sort: bool
    initialise_simulated_cycle: bool
    allow_integrated_expander: bool
    dT_subcool: Optional[np.ndarray] = None
    dT_superheat: Optional[np.ndarray] = None
    bckgrd_hot_streams: Optional[StreamCollection] = None
    bckgrd_cold_streams: Optional[StreamCollection] = None
    bb_minimiser: Optional[str] = None
    eta_penalty: Optional[float] = 0.01
    rho_penalty: Optional[float] = 10
    debug: bool


class HeatPumpTargetOutputs(BaseModel):
    """Normalized output requirement for heat pump and refrigeration targeting routines."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    utility_tot: float
    w_net: float | list | np.ndarray
    w_hpr: Optional[float | list | np.ndarray] = None
    w_he: Optional[float | list | np.ndarray] = None
    heat_recovery: Optional[float | list | np.ndarray] = None
    Q_ext: float
    Q_amb_hot: float
    Q_amb_cold: float
    cop_h: Optional[float | list | np.ndarray] = None
    eta_he: Optional[float | list | np.ndarray] = None
    obj: float
    success: bool
    hpr_hot_streams: StreamCollection
    hpr_cold_streams: StreamCollection
    amb_streams: StreamCollection
    T_cond: Optional[np.ndarray] = None
    T_evap: Optional[np.ndarray] = None
    Q_cond: Optional[np.ndarray] = None
    Q_evap: Optional[np.ndarray] = None
    Q_cond_he: Optional[np.ndarray] = None
    Q_evap_he: Optional[np.ndarray] = None
    dT_subcool: Optional[np.ndarray] = None
    dT_superheat: Optional[np.ndarray] = None
    T_comp_out: Optional[np.ndarray] = None
    dT_gc: Optional[np.ndarray] = None
    dT_comp: Optional[np.ndarray] = None
    Q_heat: Optional[np.ndarray] = None
    Q_cool: Optional[np.ndarray] = None
    model: Optional[Any] = None


__all__ = ["HeatPumpTargetInputs", "HeatPumpTargetOutputs"]


HeatPumpTargetInputs.model_rebuild()
HeatPumpTargetOutputs.model_rebuild()
