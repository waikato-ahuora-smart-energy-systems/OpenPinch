"""Schemas accepted by the external OpenPinch analysis contract."""

from __future__ import annotations

import math
from typing import List, Optional, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..domain.configuration_fields import validate_configuration_options
from ..domain.enums import ST, FluidPhase
from .common import ScalarOrVU


class StreamSegmentSchema(BaseModel):
    """One ordered linear interval in a variable-CP stream profile."""

    name: Optional[str] = None
    t_supply: ScalarOrVU
    t_target: ScalarOrVU
    heat_flow: ScalarOrVU
    p_supply: Optional[ScalarOrVU] = None
    p_target: Optional[ScalarOrVU] = None
    h_supply: Optional[ScalarOrVU] = None
    h_target: Optional[ScalarOrVU] = None
    dt_cont: Optional[ScalarOrVU] = None
    htc: Optional[ScalarOrVU] = None
    price: Optional[ScalarOrVU] = None

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
    )


class TemperatureHeatPointSchema(BaseModel):
    """One temperature and cumulative-heat coordinate in an ordered profile."""

    cumulative_heat: ScalarOrVU
    temperature: ScalarOrVU

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TemperatureHeatProfileSchema(BaseModel):
    """Ordered temperature-cumulative-heat data for one physical stream."""

    points: List[TemperatureHeatPointSchema]
    linearisation_tolerance: float = 0.1

    model_config = ConfigDict(extra="forbid")

    @field_validator("points")
    @classmethod
    def _require_two_points(
        cls,
        value: List[TemperatureHeatPointSchema],
    ) -> List[TemperatureHeatPointSchema]:
        if len(value) < 2:
            raise ValueError("A temperature-heat profile requires at least two points.")
        return value

    @field_validator("linearisation_tolerance")
    @classmethod
    def _positive_tolerance(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("linearisation_tolerance must be finite and positive.")
        return float(value)


class StreamSchema(BaseModel):
    """Process stream definition supplied to the targeting service."""

    zone: str
    name: str
    segments: Optional[List[StreamSegmentSchema]] = None
    profile: Optional[TemperatureHeatProfileSchema] = None
    t_supply: Optional[ScalarOrVU] = None
    t_target: Optional[ScalarOrVU] = None
    p_supply: Optional[ScalarOrVU] = None
    p_target: Optional[ScalarOrVU] = None
    h_supply: Optional[ScalarOrVU] = None
    h_target: Optional[ScalarOrVU] = None
    heat_flow: Optional[ScalarOrVU] = None
    heat_capacity_flowrate: Optional[ScalarOrVU] = None
    dt_cont: Optional[ScalarOrVU] = 0.0
    htc: Optional[ScalarOrVU] = 1.0
    fluid_name: Optional[str] = None
    fluid_phase: Optional[FluidPhase] = None
    active: bool = True

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        validate_default=True,
        extra="forbid",
    )

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

    @field_validator("t_supply", "t_target", "heat_flow")
    @classmethod
    def _require_ordinary_thermal_fields(
        cls,
        value: Optional[ScalarOrVU],
        info: ValidationInfo,
    ) -> Optional[ScalarOrVU]:
        has_nested = (
            info.data.get("segments") is not None
            or info.data.get("profile") is not None
        )
        if value is None and not has_nested:
            raise ValueError(f"Ordinary streams require {info.field_name}.")
        return value

    @model_validator(mode="after")
    def _validate_thermal_definition(self) -> Self:
        if self.segments is not None and self.profile is not None:
            raise ValueError("Provide either segments or profile, not both.")
        has_nested = self.segments is not None or self.profile is not None
        if self.segments is not None and len(self.segments) == 0:
            raise ValueError("segments must contain at least one segment.")
        if has_nested and self.heat_capacity_flowrate is not None:
            raise ValueError(
                "heat_capacity_flowrate cannot be supplied with segments or profile."
            )
        return self


class UtilitySchema(BaseModel):
    """Utility definition including thermal and optional economic attributes."""

    name: str
    type: ST
    segments: Optional[List[StreamSegmentSchema]] = None
    profile: Optional[TemperatureHeatProfileSchema] = None
    t_supply: Optional[ScalarOrVU] = None
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

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        validate_default=True,
    )

    @field_validator("t_supply")
    @classmethod
    def _require_ordinary_supply_temperature(
        cls,
        value: Optional[ScalarOrVU],
        info: ValidationInfo,
    ) -> Optional[ScalarOrVU]:
        has_nested = (
            info.data.get("segments") is not None
            or info.data.get("profile") is not None
        )
        if value is None and not has_nested:
            raise ValueError("Ordinary utilities require t_supply.")
        return value

    @model_validator(mode="after")
    def _validate_thermal_definition(self) -> Self:
        if self.segments is not None and self.profile is not None:
            raise ValueError("Provide either segments or profile, not both.")
        if self.segments is not None and len(self.segments) == 0:
            raise ValueError("segments must contain at least one segment.")
        return self

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


class NonLinearStream(BaseModel):
    """Nonlinear stream definition used by piecewise linearisation utilities."""

    t_supply: float
    t_target: float
    p_supply: float
    p_target: float
    h_supply: float
    h_target: float
    composition: list[tuple[str, float]]


__all__ = [
    "NonLinearStream",
    "StreamSchema",
    "StreamSegmentSchema",
    "TargetInput",
    "TemperatureHeatPointSchema",
    "TemperatureHeatProfileSchema",
    "UtilitySchema",
    "ZoneTreeSchema",
]


ZoneTreeSchema.model_rebuild()
TargetInput.model_rebuild()
