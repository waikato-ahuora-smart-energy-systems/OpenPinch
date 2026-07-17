"""Internal operating-period state for a parent heat exchanger."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


def _midpoint_temperature(
    inlet_temperature: float | None,
    outlet_temperature: float | None,
) -> float | None:
    if inlet_temperature is None or outlet_temperature is None:
        return None
    return (inlet_temperature + outlet_temperature) / 2
