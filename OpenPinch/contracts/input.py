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
from ..domain.enums import ST, FluidPhase, HeatExchangerKind, StreamID
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


class HeatExchangerAreaSliceSchema(BaseModel):
    """JSON-visible area contribution for one exchanger segment pair."""

    model_config = ConfigDict(extra="forbid")

    period: str
    hot_segment_identity: str
    cold_segment_identity: str
    duty: float
    hot_inlet_temperature: float
    hot_outlet_temperature: float
    cold_inlet_temperature: float
    cold_outlet_temperature: float
    hot_htc: float
    cold_htc: float
    overall_htc: float
    lmtd: float
    area: float

    @field_validator("period", "hot_segment_identity", "cold_segment_identity")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("segment area identities must not be empty")
        return text

    @field_validator("duty", "hot_htc", "cold_htc", "overall_htc", "lmtd", "area")
    @classmethod
    def _validate_positive_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("segment area values must be finite and positive")
        return float(value)

    @field_validator(
        "hot_inlet_temperature",
        "hot_outlet_temperature",
        "cold_inlet_temperature",
        "cold_outlet_temperature",
    )
    @classmethod
    def _validate_slice_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("segment area temperatures must be finite")
        return float(value)


class HeatExchangerPeriodStateSchema(BaseModel):
    """JSON-visible operating state for one exchanger and period."""

    model_config = ConfigDict(extra="forbid")

    period_id: str
    period_idx: int
    duty: float
    active: bool = True
    approach_temperatures: list[float] = Field(default_factory=list)
    source_split_fraction: float | None = None
    sink_split_fraction: float | None = None
    source_inlet_temperature: float | None = None
    source_outlet_temperature: float | None = None
    sink_inlet_temperature: float | None = None
    sink_outlet_temperature: float | None = None

    @field_validator("period_id")
    @classmethod
    def _validate_period_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("period_id must be a non-empty string")
        return value.strip()

    @field_validator("period_idx")
    @classmethod
    def _validate_period_idx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("period_idx must be non-negative")
        return int(value)

    @field_validator("duty")
    @classmethod
    def _validate_duty(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("period duty must be finite and non-negative")
        return float(value)

    @field_validator("source_split_fraction", "sink_split_fraction")
    @classmethod
    def _validate_split_fraction(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("split fractions must be finite values from zero to one")
        return float(value)

    @field_validator(
        "source_inlet_temperature",
        "source_outlet_temperature",
        "sink_inlet_temperature",
        "sink_outlet_temperature",
    )
    @classmethod
    def _validate_finite_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value):
            raise ValueError("temperatures must be finite values")
        return float(value)

    @field_validator("approach_temperatures")
    @classmethod
    def _validate_approach_temperatures(cls, value: list[float]) -> list[float]:
        for approach_temperature in value:
            if not math.isfinite(approach_temperature) or approach_temperature < 0.0:
                raise ValueError(
                    "approach temperatures must be finite non-negative values"
                )
        return [float(approach_temperature) for approach_temperature in value]


class HeatExchangerSchema(BaseModel):
    """JSON transport contract for one runtime heat exchanger dump."""

    model_config = ConfigDict(extra="forbid")

    exchanger_id: str | None = None
    kind: HeatExchangerKind
    source_stream: str
    sink_stream: str
    source_stream_role: StreamID
    sink_stream_role: StreamID
    stage: int | None = None
    period_states: list[HeatExchangerPeriodStateSchema] = Field(min_length=1)
    area: float | None = None
    match_allowed: bool = True
    capital_cost: float | None = None
    segment_area_contributions: list[HeatExchangerAreaSliceSchema] = Field(
        default_factory=list
    )

    @field_validator("exchanger_id", "source_stream", "sink_stream")
    @classmethod
    def _validate_identity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "stream and exchanger identities must be non-empty strings"
            )
        return value.strip()

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage must be a positive integer when supplied")
        return value

    @field_validator("area", "capital_cost")
    @classmethod
    def _validate_non_negative_finite(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("numeric exchanger values must be finite and non-negative")
        return float(value)

    @model_validator(mode="after")
    def _validate_period_states(self) -> Self:
        expected_indices = tuple(range(len(self.period_states)))
        actual_indices = tuple(state.period_idx for state in self.period_states)
        if actual_indices != expected_indices:
            raise ValueError("period_states must be ordered by contiguous period_idx")
        period_ids = tuple(state.period_id for state in self.period_states)
        if len(set(period_ids)) != len(period_ids):
            raise ValueError("period_states must use unique period_id values")
        return self

    @model_validator(mode="after")
    def _validate_direction_semantics(self) -> Self:
        if self.source_stream == self.sink_stream:
            raise ValueError("source_stream and sink_stream must be distinct")

        expected_roles = {
            HeatExchangerKind.RECOVERY: (StreamID.Process, StreamID.Process),
            HeatExchangerKind.HOT_UTILITY: (StreamID.Utility, StreamID.Process),
            HeatExchangerKind.COLD_UTILITY: (StreamID.Process, StreamID.Utility),
        }
        expected_source_role, expected_sink_role = expected_roles[self.kind]
        if (
            self.source_stream_role != expected_source_role
            or self.sink_stream_role != expected_sink_role
        ):
            raise ValueError(
                f"{self.kind.value} exchangers must link "
                f"{expected_source_role.value} -> {expected_sink_role.value}"
            )
        if self.kind is HeatExchangerKind.RECOVERY and self.stage is None:
            raise ValueError("recovery exchangers must include a synthesis stage")
        return self

    @model_validator(mode="after")
    def _validate_area_contributions(self) -> Self:
        totals: dict[str, float] = {}
        for contribution in self.segment_area_contributions:
            totals[contribution.period] = (
                totals.get(contribution.period, 0.0) + contribution.area
            )
        design_area = max(totals.values()) if totals else None
        if design_area is None:
            return self
        if self.area is not None and not math.isclose(
            self.area,
            design_area,
            rel_tol=1e-4,
            abs_tol=1e-3,
        ):
            raise ValueError(
                "area must match the maximum period-total segment area "
                f"{design_area:.12g}"
            )
        object.__setattr__(self, "area", design_area)
        return self


class HeatExchangerNetworkSchema(BaseModel):
    """JSON transport contract for a runtime heat exchanger network dump."""

    model_config = ConfigDict(extra="forbid")

    exchangers: list[HeatExchangerSchema] = Field(default_factory=list)
    run_id: str | None = None
    task_id: str | None = None
    period_id: str | None = None
    method: str | None = None
    stage_count: int | None = None
    objective_value: float | None = None
    total_annual_cost: float | None = None
    utility_cost: float | None = None
    capital_cost: float | None = None
    summary_metrics: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict
    )

    @field_validator("run_id", "task_id", "period_id", "method")
    @classmethod
    def _validate_optional_identity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("network metadata identities must be non-empty strings")
        return value.strip()

    @field_validator("stage_count")
    @classmethod
    def _validate_stage_count(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage_count must be a positive integer when supplied")
        return value

    @field_validator(
        "objective_value",
        "total_annual_cost",
        "utility_cost",
        "capital_cost",
    )
    @classmethod
    def _validate_optional_non_negative_finite(
        cls,
        value: float | None,
    ) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("network numeric values must be finite and non-negative")
        return float(value)

    @field_validator("summary_metrics")
    @classmethod
    def _validate_summary_metrics(
        cls,
        value: dict[str, float | int | str | bool | None],
    ) -> dict[str, float | int | str | bool | None]:
        for metric_name, metric_value in value.items():
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ValueError("summary metric names must be non-empty strings")
            if isinstance(metric_value, float) and not math.isfinite(metric_value):
                raise ValueError("summary metric values must be finite")
        return value

    @model_validator(mode="after")
    def _validate_period_state_alignment(self) -> Self:
        if not self.exchangers:
            return self
        ordered = tuple(state.period_id for state in self.exchangers[0].period_states)
        for exchanger in self.exchangers[1:]:
            current = tuple(state.period_id for state in exchanger.period_states)
            if current != ordered:
                raise ValueError(
                    "all exchangers in a network must use the same ordered period_ids"
                )
        return self


class TargetInput(BaseModel):
    """Validated top-level input data for :class:`PinchProblem`."""

    streams: List[StreamSchema]
    utilities: List[UtilitySchema] = Field(default_factory=list)
    options: Optional[dict] = None
    zone_tree: Optional[ZoneTreeSchema] = None
    network: HeatExchangerNetworkSchema | None = None

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
    "HeatExchangerAreaSliceSchema",
    "HeatExchangerNetworkSchema",
    "HeatExchangerPeriodStateSchema",
    "HeatExchangerSchema",
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
