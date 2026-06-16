"""OpenPinch-native heat exchanger design records."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HeatExchangerKind(str, Enum):
    """Supported heat-transfer link families in a HEN design."""

    RECOVERY = "recovery"
    HOT_UTILITY = "hot_utility"
    COLD_UTILITY = "cold_utility"


class HeatExchangerStreamRole(str, Enum):
    """Identity class for the source and sink streams on an exchanger link."""

    PROCESS = "process"
    UTILITY = "utility"


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
    duty: float
    area: float | None = None
    active: bool = True
    match_allowed: bool = True
    approach_temperatures: tuple[float, ...] = Field(default_factory=tuple)
    source_inlet_temperature: float | None = None
    source_outlet_temperature: float | None = None
    sink_inlet_temperature: float | None = None
    sink_outlet_temperature: float | None = None
    capital_cost: float | None = None
    operating_cost: float | None = None
    total_annual_cost: float | None = None
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
        "duty",
        "area",
        "capital_cost",
        "operating_cost",
        "total_annual_cost",
    )
    @classmethod
    def _validate_non_negative_finite(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("numeric exchanger values must be finite and non-negative")
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
            if (
                not math.isfinite(approach_temperature)
                or approach_temperature < 0.0
            ):
                raise ValueError(
                    "approach temperatures must be finite non-negative values"
                )
        return tuple(float(approach_temperature) for approach_temperature in value)

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


__all__ = [
    "HeatExchanger",
    "HeatExchangerKind",
    "HeatExchangerStreamRole",
]
