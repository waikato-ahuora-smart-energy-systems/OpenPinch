"""Pydantic schemas for OpenPinch inputs, outputs, and helper data contracts.

These models define the validated wire format used by the high-level service,
including stream/utility input payloads, zone hierarchies, target summaries,
graph structures, and specialist analysis helper payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .enums import StreamType, TurbineOptionsPropKeys
from ..classes.stream_collection import StreamCollection


# ---- Common type aliases -----------------------------------------------------
ScalarOrVU = Union[float, "ValueWithUnit"]
MaybeVU = Union[float, "ValueWithUnit", None]


# ---- Core value types --------------------------------------------------------
class ValueWithUnit(BaseModel):
    """Container storing a magnitude and its associated unit string."""

    value: Optional[float] = Field(
        default=None, description="Numeric value (magnitude)."
    )
    units: str = Field(..., description="Unit string, e.g. 'kW', '°C', 'kJ/s'.")


# ---- Utilities & Pinch temps -------------------------------------------------
class HeatUtility(BaseModel):
    """Report-friendly representation of a utility contribution."""

    name: str
    heat_flow: ScalarOrVU


class TempPinch(BaseModel):
    """Hot and cold pinch temperatures attached to a targeting record."""

    cold_temp: MaybeVU = None
    hot_temp: MaybeVU = None


class HPRTargetInputs(BaseModel):
    """Parameter bundle for heat pump and refrigeration targeting routines."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Calculated based on the case
    hpr_type: str
    Q_hpr_target: float
    Q_heat_max: float
    Q_cool_max: float
    z_amb_hot: np.ndarray
    z_amb_cold: np.ndarray
    dt_range_max: float

    # Background process net hot and cold load curves
    T_hot: np.ndarray | list
    H_hot: np.ndarray | list
    T_cold: np.ndarray | list
    H_cold: np.ndarray | list

    # From overall analysis configuration
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

    # Optional arguments
    dT_subcool: Optional[np.ndarray] = None
    dT_superheat: Optional[np.ndarray] = None
    bckgrd_hot_streams: Optional[StreamCollection] = None
    bckgrd_cold_streams: Optional[StreamCollection] = None
    bb_minimiser: Optional[str] = None
    eta_penalty: Optional[float] = 0.01
    rho_penalty: Optional[float] = 10

    # Debug mode toogle
    debug: bool


class HPRTargetOutputs(BaseModel):
    """Normalized output requirement for heat pump and refrigeration targeting routines."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # --- Common objective / result fields -------------------------
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

    # --- Flattened state fields (union of all children) -----------
    # Carnot & Simple Vapour Compression
    T_cond: Optional[np.ndarray] = None
    T_evap: Optional[np.ndarray] = None
    Q_cond: Optional[np.ndarray] = None
    Q_evap: Optional[np.ndarray] = None

    # Simple Vapour Compression only
    dT_subcool: Optional[np.ndarray] = None
    dT_superheat: Optional[np.ndarray] = None

    # Brayton only
    T_comp_out: Optional[np.ndarray] = None
    dT_gc: Optional[np.ndarray] = None
    dT_comp: Optional[np.ndarray] = None
    Q_heat: Optional[np.ndarray] = None
    Q_cool: Optional[np.ndarray] = None

    model: Optional[Any] = None


# ---- Targeting results -------------------------------------------------------
class TargetResults(BaseModel):
    """Summary metrics for a single zone/target returned by the analysis."""

    name: str

    degree_of_integration: MaybeVU = None

    Qh: ScalarOrVU
    Qc: ScalarOrVU
    Qr: ScalarOrVU

    utility_cost: MaybeVU = None
    row_type: Optional[str] = None

    hot_utilities: List[HeatUtility] = Field(default_factory=list)
    cold_utilities: List[HeatUtility] = Field(default_factory=list)

    temp_pinch: TempPinch

    work_target: MaybeVU = None
    turbine_efficiency_target: MaybeVU = None
    area: MaybeVU = None

    num_units: Optional[float] = None
    capital_cost: Optional[float] = None
    total_cost: Optional[float] = None

    exergy_sources: MaybeVU = None
    exergy_sinks: MaybeVU = None
    ETE: Optional[float] = None
    exergy_req_min: MaybeVU = None
    exergy_des_min: MaybeVU = None


class TurbineStageResult(BaseModel):
    """Detailed result for one solved turbine stage."""

    stage: int
    source_index: int
    stage_type: str
    temperature: float
    process_duty: float
    pressure_in: float
    pressure_out: float
    mass_flow_in: float
    mass_flow_extracted: float
    mass_flow_out: float
    enthalpy_in: float
    enthalpy_out: float
    condensate_enthalpy: float
    saturation_enthalpy: float
    dh_isentropic: float
    work_actual: float
    work_isentropic: float
    isentropic_efficiency: float
    turbine_model: str


class TurbineSolveResult(BaseModel):
    """Validated output for a multi-stage turbine targeting solve."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    turbine_model: str
    load_frac: float
    mech_eff: float
    min_eff: float
    flash_correction: bool
    total_work: float
    total_isentropic_work: float
    overall_efficiency: float
    total_process_duty: float
    steam_mass_flow_in: Optional[float] = None
    inlet_pressure: Optional[float] = None
    inlet_temperature: Optional[float] = None
    sink_pressure: Optional[float] = None
    sink_temperature: Optional[float] = None
    stage_temperatures: List[float] = Field(default_factory=list)
    stage_heat_flows: List[float] = Field(default_factory=list)
    stages: List[TurbineStageResult] = Field(default_factory=list)


# ---- Graphing primitives -----------------------------------------------------
class DataPoint(BaseModel):
    """Coordinate used to construct composite curves and other plots."""

    x: float
    y: float


class Segment(BaseModel):
    """Continuous plot segment optionally annotated with colour/arrows."""

    title: Optional[str] = None
    colour: Optional[int] = Field(
        default=None,
        description="Optional integer colour (e.g., RGB packed int or palette index).",
    )
    arrow: Optional[str] = Field(
        default=None,
        description="Optional arrow style; consider making this an Enum.",
    )
    data_points: List[DataPoint] = Field(default_factory=list)


class Graph(BaseModel):
    """Collection of segments representing a single graph (e.g., GCC)."""

    # Consider making 'type' an Enum (e.g., GCC, CCC, PT, etc.) for safety.
    type: str
    segments: List[Segment] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)


class GraphSet(BaseModel):
    """Named group of graphs emitted for a particular zone or context."""

    name: str = "GraphSet"
    graphs: List[Graph] = Field(default_factory=list)


# ---- Aggregate response ------------------------------------------------------
class TargetOutput(BaseModel):
    """Top-level payload returned by :func:`OpenPinch.pinch_analysis_service`."""

    name: str = "Site"
    targets: List[TargetResults]
    graphs: Optional[Dict[str, GraphSet]] = None


# ---- Heat-pump integration helpers ------------------------------------------
class HeatPumpIntegrationScenario(BaseModel):
    """User-facing definition of a candidate integrated heat-pump scenario."""

    zone: str = "Plant"
    condenser_temperature: float
    condenser_duty: float
    evaporator_temperature: float
    evaporator_duty: float
    dt_phase_change: float = 0.1
    dt_cont: float = 0.0
    htc: float = 1.0
    condenser_name: str = "HP Condenser"
    evaporator_name: str = "HP Evaporator"


class HeatPumpIntegrationComparison(BaseModel):
    """Compact before/after comparison for a heat-pump integration scenario."""

    target: str
    base_case_name: str
    scenario_case_name: str
    hot_utility_target_delta: float
    cold_utility_target_delta: float
    heat_recovery_delta: float
    hot_pinch_delta: Optional[float] = None
    cold_pinch_delta: Optional[float] = None
    approximate_power_input: float


# ---- Stream & Utility definitions -------------------------------------------
class StreamSchema(BaseModel):
    """Process stream definition supplied to the targeting service."""

    zone: str
    name: str

    t_supply: ScalarOrVU
    t_target: ScalarOrVU
    heat_flow: ScalarOrVU

    dt_cont: ScalarOrVU
    htc: ScalarOrVU

    active: bool = True


class UtilitySchema(BaseModel):
    """Utility definition including thermal and optional economic attributes."""

    name: str
    type: StreamType

    t_supply: ScalarOrVU
    t_target: ScalarOrVU
    heat_flow: Optional[ScalarOrVU] = None

    dt_cont: ScalarOrVU
    htc: ScalarOrVU
    price: ScalarOrVU

    active: bool = True

    model_config = ConfigDict(use_enum_values=True)


# ---- Zone tree ---------------------------------------------------------------
class ZoneTreeSchema(BaseModel):
    """Recursive description of the zone hierarchy for the analysis."""

    name: str
    type: str
    children: Optional[List["ZoneTreeSchema"]] = None


# ---- Options -----------------------------------------------------------------
class TurbineOption(BaseModel):
    """Configure individual turbine properties referenced by key."""

    key: TurbineOptionsPropKeys
    value: Any

    model_config = ConfigDict(use_enum_values=True)


# class Options(BaseModel):
#     """Primary checkbox-style options plus turbine configuration."""

#     main: List[MainOptionsPropKeys] = Field(default_factory=list)
#     # graphs: List[GraphOptionsPropKeys]
#     turbine: List[TurbineOption] = Field(default_factory=list)

#     model_config = ConfigDict(use_enum_values=True)


# ---- Complete request --------------------------------------------------------
class TargetInput(BaseModel):
    """Validated top-level input payload for :func:`OpenPinch.pinch_analysis_service`."""

    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = []
    options: Optional[dict] = None
    zone_tree: Optional[ZoneTreeSchema] = None


# ---- Problem table / TH data (for tests & I/O) -------------------------------
class THSchema(BaseModel):
    """Temperature-enthalpy series payload used for problem-table exchange."""

    T: List[float]
    H_hot: Optional[List[float]] = None
    H_cold: Optional[List[float]] = None
    H_net: Optional[List[float]] = None
    H_hot_net: Optional[List[float]] = None
    H_cold_net: Optional[List[float]] = None


class ProblemTableDataSchema(BaseModel):
    """Named container for a single temperature-enthalpy profile."""

    name: str
    data: THSchema


class GetInputOutputData(BaseModel):
    """Aggregate structure used by legacy I/O helpers and tests."""

    plant_profile_data: List[ProblemTableDataSchema]
    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = Field(default_factory=list)
    options: Optional[dict] = {}


# ---- Linearisation schema ---------------------------------------------------


class NonLinearStream(BaseModel):
    """Nonlinear stream definition used by piecewise linearisation utilities."""

    t_supply: float
    t_target: float
    p_supply: float
    p_target: float
    h_supply: float
    h_target: float
    composition: list[tuple[str, float]]


class LineariseInput(BaseModel):
    """Input bundle for stream linearisation workflows."""

    t_h_data: List
    num_intervals: Optional[int] = 100
    t_min: Optional[float] = 1
    streams: List[NonLinearStream]
    ppKey: str = ""
    mole_flow: float = 1.0


class LineariseOutput(BaseModel):
    """Output payload containing generated linearised stream segments."""

    streams: List[Optional[list]]


# ---- Visualisation schema ---------------------------------------------------


class VisualiseInput(BaseModel):
    """Input payload for graph visualisation conversion routines."""

    zones: list


class VisualiseOutput(BaseModel):
    """Graph payload returned by visualisation conversion routines."""

    graphs: List[GraphSet]
