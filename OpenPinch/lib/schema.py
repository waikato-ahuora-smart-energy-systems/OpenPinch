from pydantic import BaseModel
from typing import Any, List, Optional, Dict
from .enums import MainOptionsPropKeys, TurbineOptionsPropKeys, StreamType

class ValueWithUnit(BaseModel):
    value: float | None
    units: str

# Hot or Cold Utility
class HeatUtility(BaseModel):
    name: str
    heat_flow: float | ValueWithUnit

# Pinch Temperature/s
class TempPinch(BaseModel):
    cold_temp: Optional[float] | Optional[ValueWithUnit] = None
    hot_temp: Optional[float] | Optional[ValueWithUnit] = None

# Individual Target Summary
class TargetResults(BaseModel):
    name: str
    degree_of_integration: Optional[float] | Optional[ValueWithUnit] = None
    Qh: float | ValueWithUnit
    Qc: float | ValueWithUnit
    Qr: float | ValueWithUnit
    utility_cost: float | ValueWithUnit = None
    row_type: str = None
    hot_utilities: List[HeatUtility]
    cold_utilities: List[HeatUtility]
    temp_pinch: TempPinch
    work_target: float | ValueWithUnit = None
    turbine_efficiency_target: float | ValueWithUnit = None
    area: float | ValueWithUnit = None
    num_units: float = None
    capital_cost: float = None
    total_cost: float = None
    exergy_sources: float | ValueWithUnit = None
    exergy_sinks: float | ValueWithUnit = None
    ETE: float = None
    exergy_req_min: float | ValueWithUnit = None
    exergy_des_min: float | ValueWithUnit = None

# Coordinate
class DataPoint(BaseModel):
    x: float
    y: float

# Line Segment
class Segment(BaseModel):
    title: Optional[str] = None
    colour: Optional[int] = None
    arrow: Optional[str] = None
    data_points: Optional[List[DataPoint]] = None

# Individual Graph
class Graph(BaseModel):
    segments: Optional[List[Segment]] = []
    type: str

    class Config:  
        use_enum_values = True

class GraphSet(BaseModel):
    name: str = "GraphSet"
    graphs: List[Graph]

# Aggregate request type
class TargetResponse(BaseModel):
    name: str = 'Site'
    targets: List[TargetResults]
    graphs: Dict[str, GraphSet] = None

# Single Stream Instance
class StreamSchema(BaseModel):
    zone: str
    name: str
    t_supply: float | ValueWithUnit
    t_target: float | ValueWithUnit
    heat_flow: float | ValueWithUnit
    dt_cont: float | ValueWithUnit
    htc: float | ValueWithUnit
    active: Optional[bool] = True

# Single Utility Instance
class UtilitySchema(BaseModel):
    name: str
    type: StreamType
    t_supply: float | ValueWithUnit
    t_target: float | ValueWithUnit
    heat_flow: Optional[float] | Optional[ValueWithUnit]
    dt_cont: float | ValueWithUnit
    htc: float | ValueWithUnit
    price: float | ValueWithUnit
    active: Optional[bool] = True

    class Config:  
        use_enum_values = True

# Zone relationships and nested tree structure
class ZoneTreeSchema(BaseModel):
    name: str
    type: str
    children: Optional[List["ZoneTreeSchema"]] = []

# Turbine Option Dict
class TurbineOption(BaseModel):
    key: TurbineOptionsPropKeys
    value: Any

    class Config:  
        use_enum_values = True

# Primary Options
class Options(BaseModel):
    main: List[MainOptionsPropKeys]
    # graphs: List[GraphOptionsPropKeys]
    turbine: List[TurbineOption]

    class Config:  
        use_enum_values = True

# Complete Request
class TargetRequest(BaseModel):
    streams: List[StreamSchema]
    utilities: Optional[List[UtilitySchema]] = []  
    options: Optional[Options] = None
    zone_tree: Optional[ZoneTreeSchema] = None





# Test required schema
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
    utilities: Optional[List[UtilitySchema]] = []  
    options: Optional[Options] = None