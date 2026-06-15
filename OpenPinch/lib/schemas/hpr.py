"""Schemas used by heat pump and refrigeration solvers."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
    n_mvr: int = 1
    eta_comp: float
    eta_mvr_comp: float = 0.7
    eta_motor: float = 0.95
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
    mvr_fluid_ls: List[str] = Field(default_factory=lambda: ["Water"])
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
    idx: int = 0
    debug: bool


class HeatPumpTargetOutputs(BaseModel):
    """Normalized output contract for HPR targeting routines."""

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


class HPRParsedState(BaseModel):
    """Internal parsed optimisation-state payload across HPR backends."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    Q_amb_hot: float = 0.0
    Q_amb_cold: float = 0.0
    T_cond: np.ndarray | None = None
    T_evap: np.ndarray | None = None
    dT_subcool: np.ndarray | None = None
    dT_superheat: np.ndarray | None = None
    dT_ihx_gas_side: np.ndarray | None = None
    T_comp_out: np.ndarray | None = None
    dT_gc: np.ndarray | None = None
    dT_comp: np.ndarray | None = None
    Q_heat: np.ndarray | None = None
    Q_cool: np.ndarray | None = None
    Q_heat_base: float | None = None
    Q_cool_base: float | None = None
    x_heat_split: np.ndarray | None = None
    x_cool_split: np.ndarray | None = None
    Q_heat_available: np.ndarray | None = None
    Q_cool_available: np.ndarray | None = None
    x_mvr_source_split: float | None = None
    x_mvr_process_split: np.ndarray | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default


class HPRThermoArtifacts(BaseModel):
    """Optional solved thermodynamic artefacts attached to a backend result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hpr_streams: StreamCollection | None = None
    model: Any = None
    debug_figure: Any = None


class HPRBackendResult(BaseModel):
    """Internal backend result before public schema validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    obj: float
    utility_tot: float
    w_net: float | list | np.ndarray
    Q_ext_heat: float
    Q_ext_cold: float
    Q_amb_hot: float
    Q_amb_cold: float
    success: bool = True
    w_hpr: float | list | np.ndarray | None = None
    w_he: float | list | np.ndarray | None = None
    heat_recovery: float | list | np.ndarray | None = None
    cop_h: float | list | np.ndarray | None = None
    eta_he: float | list | np.ndarray | None = None
    amb_streams: StreamCollection | None = None
    T_cond: np.ndarray | None = None
    T_evap: np.ndarray | None = None
    Q_cond: np.ndarray | None = None
    Q_evap: np.ndarray | None = None
    Q_cond_he: np.ndarray | None = None
    Q_evap_he: np.ndarray | None = None
    dT_subcool: np.ndarray | None = None
    dT_superheat: np.ndarray | None = None
    T_comp_out: np.ndarray | None = None
    dT_gc: np.ndarray | None = None
    dT_comp: np.ndarray | None = None
    Q_heat: np.ndarray | None = None
    Q_cool: np.ndarray | None = None
    failure_reason: str | None = None
    artifacts: HPRThermoArtifacts | None = None

    @property
    def Q_ext(self) -> float:
        return float(self.Q_ext_heat + self.Q_ext_cold)

    @property
    def hpr_streams(self) -> StreamCollection | None:
        return None if self.artifacts is None else self.artifacts.hpr_streams

    @property
    def hpr_hot_streams(self) -> StreamCollection | None:
        if self.hpr_streams is None:
            return None
        return self.hpr_streams.get_hot_utility_streams()

    @property
    def hpr_cold_streams(self) -> StreamCollection | None:
        return (
            None
            if self.hpr_streams is None
            else self.hpr_streams.get_cold_utility_streams()
        )

    @property
    def model(self) -> Any:
        return None if self.artifacts is None else self.artifacts.model

    def with_updates(self, **kwargs) -> "HPRBackendResult":
        return self.model_copy(update=kwargs)

    def __getitem__(self, key: str) -> Any:
        if key == "Q_ext":
            return self.Q_ext
        if key == "hpr_streams":
            return self.hpr_streams
        if key == "hpr_hot_streams":
            return self.hpr_hot_streams
        if key == "hpr_cold_streams":
            return self.hpr_cold_streams
        if key == "model":
            return self.model
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    def __contains__(self, key: str) -> bool:
        if not isinstance(key, str):
            return False
        if key in {
            "Q_ext",
            "hpr_streams",
            "hpr_hot_streams",
            "hpr_cold_streams",
            "model",
        }:
            return True
        return hasattr(self, key)

    def to_output_payload(self) -> dict[str, Any]:
        payload = {
            "utility_tot": self.utility_tot,
            "w_net": self.w_net,
            "w_hpr": self.w_hpr,
            "w_he": self.w_he,
            "heat_recovery": self.heat_recovery,
            "Q_ext": self.Q_ext,
            "Q_amb_hot": self.Q_amb_hot,
            "Q_amb_cold": self.Q_amb_cold,
            "cop_h": self.cop_h,
            "eta_he": self.eta_he,
            "obj": self.obj,
            "success": self.success,
            "hpr_hot_streams": self.hpr_hot_streams,
            "hpr_cold_streams": self.hpr_cold_streams,
            "amb_streams": self.amb_streams,
            "T_cond": self.T_cond,
            "T_evap": self.T_evap,
            "Q_cond": self.Q_cond,
            "Q_evap": self.Q_evap,
            "Q_cond_he": self.Q_cond_he,
            "Q_evap_he": self.Q_evap_he,
            "dT_subcool": self.dT_subcool,
            "dT_superheat": self.dT_superheat,
            "T_comp_out": self.T_comp_out,
            "dT_gc": self.dT_gc,
            "dT_comp": self.dT_comp,
            "Q_heat": self.Q_heat,
            "Q_cool": self.Q_cool,
            "model": self.model,
        }
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def failure(
        cls,
        *,
        reason: str | None = None,
        Q_amb_hot: float = 0.0,
        Q_amb_cold: float = 0.0,
    ) -> "HPRBackendResult":
        return cls(
            obj=float(np.inf),
            utility_tot=float(np.inf),
            w_net=0.0,
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=Q_amb_hot,
            Q_amb_cold=Q_amb_cold,
            success=False,
            failure_reason=reason,
        )


__all__ = [
    "HeatPumpTargetInputs",
    "HeatPumpTargetOutputs",
    "HPRParsedState",
    "HPRThermoArtifacts",
    "HPRBackendResult",
]


HeatPumpTargetInputs.model_rebuild()
HeatPumpTargetOutputs.model_rebuild()
HPRParsedState.model_rebuild()
HPRThermoArtifacts.model_rebuild()
HPRBackendResult.model_rebuild()
