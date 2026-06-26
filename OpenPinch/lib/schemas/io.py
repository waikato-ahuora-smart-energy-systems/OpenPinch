"""Schemas for external inputs, outputs, and user-facing I/O data."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ..config_metadata import validate_configuration_options
from ..enums import ST, FluidPhase
from .common import ScalarOrVU
from .graphs import GraphSet
from .reporting import TargetResults
from .synthesis import HeatExchangerNetworkSynthesisResult


class StreamSchema(BaseModel):
    """Process stream definition supplied to the targeting service."""

    zone: str
    name: str = Field(
        validation_alias=AliasChoices("name", "stream_name"),
        serialization_alias="name",
    )
    t_supply: ScalarOrVU
    t_target: ScalarOrVU
    p_supply: Optional[ScalarOrVU] = None
    p_target: Optional[ScalarOrVU] = None
    h_supply: Optional[ScalarOrVU] = None
    h_target: Optional[ScalarOrVU] = None
    heat_flow: ScalarOrVU
    heat_capacity_flowrate: Optional[ScalarOrVU] = Field(
        default=None,
        validation_alias=AliasChoices(
            "heat_capacity_flowrate",
            "heat_capacity_flow_rate",
            "flow_heat_capacity",
        ),
        serialization_alias="heat_capacity_flowrate",
    )
    dt_cont: Optional[ScalarOrVU] = 0.0
    htc: Optional[ScalarOrVU] = 1.0
    fluid_name: Optional[str] = None
    fluid_phase: Optional[FluidPhase] = None
    active: bool = True

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    @property
    def stream_name(self) -> str:
        """Alias for the canonical stream identifier."""
        return self.name

    @stream_name.setter
    def stream_name(self, value: str) -> None:
        self.name = value

    @field_validator("fluid_name")
    @classmethod
    def _validate_fluid_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("fluid_phase", mode="before")
    @classmethod
    def _normalise_fluid_phase(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return FluidPhase.from_code_or_description(value) if text else None


class UtilitySchema(BaseModel):
    """Utility definition including thermal and optional economic attributes."""

    name: str
    type: ST
    t_supply: ScalarOrVU
    t_target: Optional[ScalarOrVU] = None
    p_supply: Optional[ScalarOrVU] = None
    p_target: Optional[ScalarOrVU] = None
    h_supply: Optional[ScalarOrVU] = None
    h_target: Optional[ScalarOrVU] = None
    heat_flow: Optional[ScalarOrVU] = None
    dt_cont: Optional[ScalarOrVU] = 0.0
    htc: Optional[ScalarOrVU] = 1.0
    price: Optional[ScalarOrVU] = 1.0
    fluid_name: Optional[str] = None
    fluid_phase: Optional[FluidPhase] = None
    active: bool = True

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("fluid_name")
    @classmethod
    def _validate_fluid_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("fluid_phase", mode="before")
    @classmethod
    def _normalise_fluid_phase(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return FluidPhase.from_code_or_description(value) if text else None


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
    """Validated top-level input data for ``pinch_analysis_service``."""

    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = Field(default_factory=list)
    options: Optional[dict] = None
    zone_tree: Optional[ZoneTreeSchema] = None

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: Optional[dict]) -> Optional[dict]:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError("TargetInput options must be provided as a dict.")
        return validate_configuration_options(value)


class TargetOutput(BaseModel):
    """Top-level response data returned by :func:`OpenPinch.pinch_analysis_service`."""

    name: str = "Site"
    period_id: Optional[str] = None
    targets: List[TargetResults]
    graphs: Optional[Dict[str, GraphSet]] = None
    design: Optional[HeatExchangerNetworkSynthesisResult] = None


class THSchema(BaseModel):
    """Temperature-enthalpy series data used for Problem Table exchange."""

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
    """Output data containing generated linearised stream segments."""

    streams: List[Optional[list]]


class VisualiseInput(BaseModel):
    """Input data for graph visualisation conversion routines."""

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
