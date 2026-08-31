"""Common schema primitives and shared type aliases."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ValueWithUnit(BaseModel):
    """Container storing a magnitude and its associated unit string."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = Field(
        default=None, description="Numeric value (magnitude)."
    )
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )


class PeriodValueWithUnit(BaseModel):
    """Container storing multi-period magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(..., description="Per-period magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )


class PeriodValueWithUnitAndWeights(BaseModel):
    """Container storing multi-period magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(..., description="Per-period magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )
    weights: Optional[list[float]] = Field(
        default=None, description="Optional ordered period weights."
    )


class PeriodValueWithUnitAndIds(BaseModel):
    """Container storing magnitudes keyed to explicit operating-period identities."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(..., description="Per-period magnitudes.")
    period_ids: list[str] = Field(..., description="Ordered operating-period IDs.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )

    @model_validator(mode="after")
    def _validate_period_identity(self):
        if len(self.values) != len(self.period_ids):
            raise ValueError("period_ids must align with values")
        if not self.period_ids or any(
            not period_id.strip() for period_id in self.period_ids
        ):
            raise ValueError("period_ids must be non-empty")
        if len(set(self.period_ids)) != len(self.period_ids):
            raise ValueError("period_ids must be unique")
        return self


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
    "PeriodValueWithUnitAndIds",
    "PeriodValueWithUnitAndWeights",
    "ValueWithUnit",
]
