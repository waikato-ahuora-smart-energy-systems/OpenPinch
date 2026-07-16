"""OpenPinch-native heat exchanger design records."""

from __future__ import annotations

import math
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..lib.enums import HeatExchangerKind, HeatExchangerStreamRole
from . import _heat_exchanger_area

HeatExchangerAreaSlice = _heat_exchanger_area.HeatExchangerAreaSlice


class HeatExchangerPeriodState(BaseModel):
    """One exchanger's operational state for one ordered operating period."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    period_id: str
    period_idx: int
    duty: float
    active: bool = True
    approach_temperatures: tuple[float, ...] = Field(default_factory=tuple)
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

    @field_validator(
        "source_split_fraction",
        "sink_split_fraction",
    )
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
    def _validate_approach_temperatures(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        for approach_temperature in value:
            if not math.isfinite(approach_temperature) or approach_temperature < 0.0:
                raise ValueError(
                    "approach temperatures must be finite non-negative values"
                )
        return tuple(float(approach_temperature) for approach_temperature in value)

    @property
    def source_mid_temperature(self) -> float | None:
        """Return the source midpoint temperature when both endpoints exist."""

        return _midpoint_temperature(
            self.source_inlet_temperature,
            self.source_outlet_temperature,
        )

    @property
    def sink_mid_temperature(self) -> float | None:
        """Return the sink midpoint temperature when both endpoints exist."""

        return _midpoint_temperature(
            self.sink_inlet_temperature,
            self.sink_outlet_temperature,
        )


class HeatExchanger(BaseModel):
    """One labelled heat-transfer link in a heat exchanger network."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    exchanger_id: str | None = None
    kind: HeatExchangerKind
    source_stream: str
    sink_stream: str
    source_stream_role: HeatExchangerStreamRole
    sink_stream_role: HeatExchangerStreamRole
    stage: int | None = None
    period_states: tuple[HeatExchangerPeriodState, ...] = Field(min_length=1)
    area: float | None = None
    match_allowed: bool = True
    capital_cost: float | None = None
    segment_area_contributions: tuple[HeatExchangerAreaSlice, ...] = Field(
        default_factory=tuple
    )
    solver_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
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

    @field_validator(
        "area",
        "capital_cost",
    )
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
            HeatExchangerKind.RECOVERY: (
                HeatExchangerStreamRole.PROCESS,
                HeatExchangerStreamRole.PROCESS,
            ),
            HeatExchangerKind.HOT_UTILITY: (
                HeatExchangerStreamRole.UTILITY,
                HeatExchangerStreamRole.PROCESS,
            ),
            HeatExchangerKind.COLD_UTILITY: (
                HeatExchangerStreamRole.PROCESS,
                HeatExchangerStreamRole.UTILITY,
            ),
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
    def _validate_segment_design_area(self) -> Self:
        design_area = _heat_exchanger_area.validate_segment_design_area(
            self.area,
            self.segment_area_contributions,
        )
        if self.area is None and design_area is not None:
            object.__setattr__(self, "area", design_area)
        return self

    @property
    def has_segment_area_contributions(self) -> bool:
        """Return whether exact local segment-area slices are available."""

        return bool(self.segment_area_contributions)

    @property
    def segment_duty_by_period(self) -> dict[str, float]:
        """Return local slice duty totals grouped by operating period."""
        return _heat_exchanger_area.segment_duty_by_period(
            self.segment_area_contributions
        )

    @property
    def segment_area_by_period(self) -> dict[str, float]:
        """Return local slice area totals grouped by operating period."""
        return _heat_exchanger_area.segment_area_by_period(
            self.segment_area_contributions
        )

    @property
    def segment_design_area(self) -> float | None:
        """Return the maximum period-total slice area when slices are available."""
        return _heat_exchanger_area.segment_design_area(self.segment_area_contributions)

    @property
    def period_ids(self) -> tuple[str, ...]:
        """Return ordered operating-period identities for this exchanger."""

        return tuple(state.period_id for state in self.period_states)

    def state(self, period_id: str | None = None) -> HeatExchangerPeriodState:
        """Return one period state, requiring identity for multiperiod results."""

        if period_id is None:
            if len(self.period_states) != 1:
                raise ValueError(
                    "period_id is required when an exchanger has multiple period states"
                )
            return self.period_states[0]
        for state in self.period_states:
            if state.period_id == period_id:
                return state
        raise ValueError(
            f"unknown period_id {period_id!r}; expected one of {self.period_ids!r}"
        )

    def involves_stream(self, stream_id: str) -> bool:
        """Return whether this exchanger uses ``stream_id`` as source or sink."""
        return self.source_stream == stream_id or self.sink_stream == stream_id

    def matches(
        self,
        *,
        source_stream: str,
        sink_stream: str,
        stage: int | None = None,
    ) -> bool:
        """Return whether this exchanger matches a labelled stream-stage link."""
        if self.source_stream != source_stream or self.sink_stream != sink_stream:
            return False
        return stage is None or self.stage == stage


def _midpoint_temperature(
    inlet_temperature: float | None,
    outlet_temperature: float | None,
) -> float | None:
    if inlet_temperature is None or outlet_temperature is None:
        return None
    return (inlet_temperature + outlet_temperature) / 2


__all__ = [
    "HeatExchanger",
    "HeatExchangerPeriodState",
    "HeatExchangerKind",
    "HeatExchangerStreamRole",
]
