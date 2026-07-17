"""Result models for heat exchanger network controllability analysis."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..domain.enums import HeatExchangerKind


class HeatExchangerNetworkControllabilityEndpoint(BaseModel):
    """One process-stream outlet temperature treated as a controlled output."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_id: str
    stream_id: str
    side: Literal["source", "sink"]
    exchanger_count: int
    total_duty: float

    @field_validator("output_id", "stream_id")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("endpoint identities must be non-empty strings")
        return value.strip()

    @field_validator("exchanger_count")
    @classmethod
    def _validate_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("exchanger_count must be non-negative")
        return value

    @field_validator("total_duty")
    @classmethod
    def _validate_total_duty(cls, value: float) -> float:
        return _validate_non_negative_finite(value, "total_duty")


class HeatExchangerNetworkControllabilityActuator(BaseModel):
    """One manipulated variable available to control the HEN."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    actuator_id: str
    exchanger_id: str | None = None
    kind: HeatExchangerKind
    source_stream: str
    sink_stream: str
    stage: int | None = None
    manipulated_variable: Literal[
        "recovery_bypass_fraction",
        "hot_utility_flow",
        "cold_utility_flow",
    ]
    duty: float

    @field_validator("actuator_id", "source_stream", "sink_stream")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("actuator identities must be non-empty strings")
        return value.strip()

    @field_validator("exchanger_id")
    @classmethod
    def _validate_optional_identity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("exchanger_id must be a non-empty string when supplied")
        return value.strip()

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage must be positive when supplied")
        return value

    @field_validator("duty")
    @classmethod
    def _validate_duty(cls, value: float) -> float:
        return _validate_non_negative_finite(value, "duty")


class HeatExchangerNetworkControllabilityPairing(BaseModel):
    """Best steady-state output/actuator pairing entry."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_id: str
    actuator_id: str
    interaction: float

    @field_validator("output_id", "actuator_id")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("pairing identities must be non-empty strings")
        return value.strip()

    @field_validator("interaction")
    @classmethod
    def _validate_interaction(cls, value: float) -> float:
        return _validate_score(value, "interaction")


class HeatExchangerNetworkControllabilityComponents(BaseModel):
    """Component scores contributing to the composite controllability score."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rank: float
    pairing: float
    authority: float
    conditioning: float
    redundancy: float
    thermal_margin: float | None = None

    @field_validator(
        "rank",
        "pairing",
        "authority",
        "conditioning",
        "redundancy",
        "thermal_margin",
    )
    @classmethod
    def _validate_component(cls, value: float | None) -> float | None:
        if value is None:
            return value
        return _validate_score(value, "component score")


class HeatExchangerNetworkControllabilityResult(BaseModel):
    """Quantified controllability assessment for a heat exchanger network."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    score: float
    rating: Literal["strong", "moderate", "weak", "poor"]
    components: HeatExchangerNetworkControllabilityComponents
    outputs: tuple[HeatExchangerNetworkControllabilityEndpoint, ...] = Field(
        default_factory=tuple,
    )
    actuators: tuple[HeatExchangerNetworkControllabilityActuator, ...] = Field(
        default_factory=tuple,
    )
    interaction_matrix: tuple[tuple[float, ...], ...] = Field(default_factory=tuple)
    pairings: tuple[HeatExchangerNetworkControllabilityPairing, ...] = Field(
        default_factory=tuple,
    )
    matrix_rank: int = 0
    condition_number: float | None = None
    singular_values: tuple[float, ...] = Field(default_factory=tuple)
    minimum_approach_temperature: float | None = None
    diagnostics: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("score")
    @classmethod
    def _validate_score(cls, value: float) -> float:
        return _validate_score(value, "score")

    @field_validator("interaction_matrix")
    @classmethod
    def _validate_interaction_matrix(
        cls,
        value: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        rows = []
        width = None
        for row in value:
            clean_row = tuple(_validate_score(item, "interaction") for item in row)
            if width is None:
                width = len(clean_row)
            elif len(clean_row) != width:
                raise ValueError("interaction_matrix rows must have equal width")
            rows.append(clean_row)
        return tuple(rows)

    @field_validator("matrix_rank")
    @classmethod
    def _validate_matrix_rank(cls, value: int) -> int:
        if value < 0:
            raise ValueError("matrix_rank must be non-negative")
        return value

    @field_validator("condition_number")
    @classmethod
    def _validate_condition_number(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("condition_number must be finite and non-negative")
        return float(value)

    @field_validator("singular_values")
    @classmethod
    def _validate_singular_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            _validate_non_negative_finite(item, "singular value") for item in value
        )

    @field_validator("minimum_approach_temperature")
    @classmethod
    def _validate_minimum_approach(
        cls,
        value: float | None,
    ) -> float | None:
        if value is None:
            return value
        return _validate_non_negative_finite(value, "minimum_approach_temperature")


def _validate_score(value: float, field_name: str) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return float(value)


def _validate_non_negative_finite(value: float, field_name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return float(value)


__all__ = [
    "HeatExchangerNetworkControllabilityActuator",
    "HeatExchangerNetworkControllabilityComponents",
    "HeatExchangerNetworkControllabilityEndpoint",
    "HeatExchangerNetworkControllabilityPairing",
    "HeatExchangerNetworkControllabilityResult",
]
