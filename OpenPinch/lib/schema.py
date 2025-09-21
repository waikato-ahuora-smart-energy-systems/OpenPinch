from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from .enums import MainOptionsPropKeys, TurbineOptionsPropKeys, StreamType

# ---- Common type aliases -----------------------------------------------------
ScalarOrVU = Union[float, "ValueWithUnit"]
MaybeVU = Union[float, "ValueWithUnit", None]

# ---- Core value types --------------------------------------------------------
class ValueWithUnit(BaseModel):
    value: Optional[float] = Field(default=None, description="Numeric value (magnitude).")
    units: str = Field(..., description="Unit string, e.g. 'kW', '°C', 'kJ/s'.")

# ---- Utilities & Pinch temps -------------------------------------------------
class HeatUtility(BaseModel):
    name: str
    heat_flow: ScalarOrVU

class TempPinch(BaseModel):
    cold_temp: MaybeVU = None
    hot_temp: MaybeVU = None

# ---- Targeting results -------------------------------------------------------
class TargetResults(BaseModel):
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
    x: float
    y: float

class Segment(BaseModel):
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
    # Consider making 'type' an Enum (e.g., GCC, CCC, PT, etc.) for safety.
    type: str
    segments: List[Segment] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)

class GraphSet(BaseModel):
    name: str = "GraphSet"
    graphs: List[Graph] = Field(default_factory=list)

# ---- Aggregate response ------------------------------------------------------
class TargetOutput(BaseModel):
    name: str = "Site"
    targets: List[TargetResults]
    graphs: Optional[Dict[str, GraphSet]] = None

# ---- Stream & Utility definitions -------------------------------------------
class StreamSchema(BaseModel):
    zone: str
    name: str

    t_supply: ScalarOrVU
    t_target: ScalarOrVU
    heat_flow: ScalarOrVU

    dt_cont: ScalarOrVU
    htc: ScalarOrVU

    active: bool = True

class UtilitySchema(BaseModel):
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
    name: str
    type: str
    children: Optional[List["ZoneTreeSchema"]] = None

# ---- Options -----------------------------------------------------------------
class TurbineOption(BaseModel):
    key: TurbineOptionsPropKeys
    value: Any

    model_config = ConfigDict(use_enum_values=True)

class Options(BaseModel):
    main: List[MainOptionsPropKeys] = Field(default_factory=list)
    # graphs: List[GraphOptionsPropKeys]
    turbine: List[TurbineOption] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)

# ---- Complete request --------------------------------------------------------
class TargetInput(BaseModel):
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
