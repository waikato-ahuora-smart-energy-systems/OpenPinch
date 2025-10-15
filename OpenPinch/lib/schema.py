from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .enums import MainOptionsPropKeys, StreamType, TurbineOptionsPropKeys

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


class Options(BaseModel):
    """Primary checkbox-style options plus turbine configuration."""

    main: List[MainOptionsPropKeys] = Field(default_factory=list)
    # graphs: List[GraphOptionsPropKeys]
    turbine: List[TurbineOption] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)


# ---- Complete request --------------------------------------------------------
class TargetInput(BaseModel):
    """Validated top-level input payload for :func:`OpenPinch.pinch_analysis_service`."""

    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = Field(default_factory=list)
    options: Optional[Options] = None
    zone_tree: Optional[ZoneTreeSchema] = None


# ---- Problem table / TH data (for tests & I/O) -------------------------------
class THSchema(BaseModel):
    T: List[float]
    H_hot: Optional[List[float]] = None
    H_cold: Optional[List[float]] = None
    H_net: Optional[List[float]] = None
    H_hot_net: Optional[List[float]] = None
    H_cold_net: Optional[List[float]] = None


class ProblemTableDataSchema(BaseModel):
    name: str
    data: THSchema


class GetInputOutputData(BaseModel):
    plant_profile_data: List[ProblemTableDataSchema]
    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = Field(default_factory=list)
    options: Optional[Options] = None


# ---- Linearisation schema ---------------------------------------------------


class NonLinearStream(BaseModel):
    t_supply: float
    t_target: float
    p_supply: float
    p_target: float
    h_supply: float
    h_target: float
    composition: list[tuple[str, float]]


class LineariseInput(BaseModel):
    t_h_data: List
    num_intervals: Optional[int] = 100
    t_min: Optional[float] = 1
    streams: List[NonLinearStream]
    ppKey: str = ""
    mole_flow: float = 1.0


class LineariseOutput(BaseModel):
    streams: List[Optional[list]]


# ---- Visualisation schema ---------------------------------------------------


class VisualiseInput(BaseModel):
    zones: list


class VisualiseOutput(BaseModel):
    graphs: List[GraphSet]
