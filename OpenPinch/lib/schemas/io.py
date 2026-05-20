"""Schemas for external inputs, outputs, and user-facing I/O payloads."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..enums import ST
from .common import ScalarOrVU
from .graphs import GraphSet
from .reporting import TargetResults


class StreamSchema(BaseModel):
    """Process stream definition supplied to the targeting service."""

    zone: str
    name: str
    t_supply: ScalarOrVU
    t_target: ScalarOrVU
    heat_flow: ScalarOrVU
    dt_cont: Optional[ScalarOrVU] = 0.0
    htc: Optional[ScalarOrVU] = 1.0
    active: bool = True


class UtilitySchema(BaseModel):
    """Utility definition including thermal and optional economic attributes."""

    name: str
    type: ST
    t_supply: ScalarOrVU
    t_target: Optional[ScalarOrVU] = None
    heat_flow: Optional[ScalarOrVU] = None
    dt_cont: Optional[ScalarOrVU] = 0.0
    htc: Optional[ScalarOrVU] = 1.0
    price: Optional[ScalarOrVU] = 1.0
    active: bool = True

    model_config = ConfigDict(use_enum_values=True)


class ZoneTreeSchema(BaseModel):
    """Recursive description of the zone hierarchy for the analysis."""

    name: str
    type: str
    dt_cont_multiplier: Optional[float] = None
    children: Optional[List["ZoneTreeSchema"]] = None

    @field_validator("dt_cont_multiplier")
    @classmethod
    def _validate_dt_cont_multiplier(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("dt_cont_multiplier must be a finite non-negative value.")
        return float(value)


class TargetInput(BaseModel):
    """Validated top-level input payload for ``pinch_analysis_service``."""

    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = Field(default_factory=list)
    options: Optional[dict] = None
    zone_tree: Optional[ZoneTreeSchema] = None


class TargetOutput(BaseModel):
    """Top-level payload returned by :func:`OpenPinch.pinch_analysis_service`."""

    name: str = "Site"
    targets: List[TargetResults]
    graphs: Optional[Dict[str, GraphSet]] = None


class THSchema(BaseModel):
    """Temperature-enthalpy series payload used for Problem Table exchange."""

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
    options: Optional[dict] = Field(default_factory=dict)


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


class VisualiseInput(BaseModel):
    """Input payload for graph visualisation conversion routines."""

    zones: list


class VisualiseOutput(BaseModel):
    """Graph payload returned by visualisation conversion routines."""

    graphs: List[GraphSet]


__all__ = [
    "GetInputOutputData",
    "LineariseInput",
    "LineariseOutput",
    "NonLinearStream",
    "ProblemTableDataSchema",
    "StreamSchema",
    "THSchema",
    "TargetInput",
    "TargetOutput",
    "UtilitySchema",
    "VisualiseInput",
    "VisualiseOutput",
    "ZoneTreeSchema",
]


ZoneTreeSchema.model_rebuild()
TargetInput.model_rebuild()
TargetOutput.model_rebuild()
GetInputOutputData.model_rebuild()
LineariseInput.model_rebuild()
LineariseOutput.model_rebuild()
VisualiseInput.model_rebuild()
VisualiseOutput.model_rebuild()
