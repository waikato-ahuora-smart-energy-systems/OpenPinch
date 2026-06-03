"""Common schema primitives and shared type aliases."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
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


class StatefulValueWithUnit(BaseModel):
    """Container storing multi-state magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    values: list[float] = Field(..., description="Per-state magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )


class StatefulValueWithUnitAndWeights(BaseModel):
    """Container storing multi-state magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    values: list[float] = Field(..., description="Per-state magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )
    weights: Optional[list[float]] = Field(
        default=None, description="Optional ordered state weights."
    )


ScalarOrVU = Union[
    float, ValueWithUnit, StatefulValueWithUnit, StatefulValueWithUnitAndWeights
]
MaybeVU = Union[
    float, ValueWithUnit, StatefulValueWithUnit, StatefulValueWithUnitAndWeights, None
]
HPRMetric = Union[float, list[float], np.ndarray, None]


__all__ = [
    "HPRMetric",
    "MaybeVU",
    "ScalarOrVU",
    "StatefulValueWithUnit",
    "StatefulValueWithUnitAndWeights",
    "ValueWithUnit",
]
