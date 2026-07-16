"""Public data models returned by direct gas MVR solves."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .....classes.stream_collection import StreamCollection

DEFAULT_MVR_COMP_EFFICIENCY = 0.7
DEFAULT_MOTOR_EFFICIENCY = 0.95
DEFAULT_DIRECT_MVR_STAGES = 1
DEFAULT_TEMPERATURE_UNIT = "degC"
DEFAULT_PRESSURE_UNIT = "kPa"
DEFAULT_ENTHALPY_UNIT = "kJ/kg"
DEFAULT_HEAT_FLOW_UNIT = "kW"


@dataclass(frozen=True)
class DirectGasMVROutputUnits:
    """Units used for public direct-MVR outputs."""

    temperature: str = DEFAULT_TEMPERATURE_UNIT
    pressure: str = DEFAULT_PRESSURE_UNIT
    enthalpy: str = DEFAULT_ENTHALPY_UNIT
    heat_flow: str = DEFAULT_HEAT_FLOW_UNIT


@dataclass
class DirectGasMVRSettings:
    """User-facing settings for one direct gas MVR solve."""

    n_stages: int = DEFAULT_DIRECT_MVR_STAGES
    mvr_stage_t_lift: float | None = None
    mvr_stage_pressure_ratio: float | None = None
    liquid_injection: bool = False
    eta_mvr_comp: float = DEFAULT_MVR_COMP_EFFICIENCY
    eta_motor: float = DEFAULT_MOTOR_EFFICIENCY
    dt_diff_max: float = 0.1


@dataclass
class DirectGasMVRStageResult:
    """Solved accounting for one direct gas MVR stage."""

    source_stream: str
    stage_index: int
    p_in: float
    p_out: float
    t_in: float
    t_discharge: float
    t_hot_supply: float
    t_target: float
    heat_flow: float
    work: float
    h_hot_supply: float
    h_target: float
    th_curve: np.ndarray = field(repr=False)
    linearised_profile: np.ndarray = field(repr=False)
    q_liquid_injection: float = 0.0
    liquid_injection_applied: bool = False
    temperature_unit: str = DEFAULT_TEMPERATURE_UNIT
    pressure_unit: str = DEFAULT_PRESSURE_UNIT
    enthalpy_unit: str = DEFAULT_ENTHALPY_UNIT
    heat_flow_unit: str = DEFAULT_HEAT_FLOW_UNIT
    source_mass_flow: float = 0.0
    hot_mass_flow: float = 0.0
    liquid_injection_ratio: float = 0.0


@dataclass
class DirectGasMVRStreamSolveResult:
    """Solved direct gas MVR streams for one source stream at one period index."""

    replacement_streams: StreamCollection
    stage_results: list[DirectGasMVRStageResult] = field(default_factory=list)
