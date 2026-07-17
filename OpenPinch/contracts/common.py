"""Common schema primitives and shared type aliases."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ValueWithUnit(BaseModel):
    """Container storing a magnitude and its associated unit string."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    value: Optional[float] = Field(
        default=None, description="Numeric value (magnitude)."
    )
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )


class PeriodValueWithUnit(BaseModel):
    """Container storing multi-period magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    values: list[float] = Field(..., description="Per-period magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )


class PeriodValueWithUnitAndWeights(BaseModel):
    """Container storing multi-period magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    values: list[float] = Field(..., description="Per-period magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )
    weights: Optional[list[float]] = Field(
        default=None, description="Optional ordered period weights."
    )


ScalarOrVU = Union[
    float, ValueWithUnit, PeriodValueWithUnit, PeriodValueWithUnitAndWeights
]
MaybeVU = Union[
    float, ValueWithUnit, PeriodValueWithUnit, PeriodValueWithUnitAndWeights, None
]
HPRMetric = Union[ValueWithUnit, None]


__all__ = [
    "HPRMetric",
    "MaybeVU",
    "ScalarOrVU",
    "PeriodValueWithUnit",
    "PeriodValueWithUnitAndWeights",
    "ValueWithUnit",
]
