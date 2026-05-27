"""Common schema primitives and shared type aliases."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ValueWithUnit(BaseModel):
    """Container storing a magnitude and its associated unit string."""

    model_config = ConfigDict(populate_by_name=True)

    value: Optional[float] = Field(
        default=None, description="Numeric value (magnitude)."
    )
    units: str = Field(..., description="Unit string, e.g. 'kW', '°C', 'kJ/s'.")

    @model_validator(mode="before")
    @classmethod
    def _accept_unit_alias(cls, data):
        if isinstance(data, dict) and "unit" in data and "units" not in data:
            data = dict(data)
            data["units"] = data.pop("unit")
        return data


class StatefulValueWithUnit(BaseModel):
    """Container storing multi-state magnitudes, weights, and a shared unit."""

    model_config = ConfigDict(populate_by_name=True)

    values: list[float] = Field(..., description="Per-state magnitudes.")
    unit: Optional[str] = Field(
        default=None, description="Shared unit string, e.g. 'degC' or 'kW'."
    )
    state_ids: Optional[list[str]] = Field(
        default=None, description="Optional state identifiers."
    )
    weights: Optional[list[float]] = Field(
        default=None, description="Optional state weights."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_units_alias(cls, data):
        if isinstance(data, dict) and "units" in data and "unit" not in data:
            data = dict(data)
            data["unit"] = data.pop("units")
        return data

    @property
    def units(self) -> Optional[str]:
        """Compatibility alias used by generic unit-extraction helpers."""
        return self.unit


ScalarOrVU = Union[float, ValueWithUnit, StatefulValueWithUnit]
MaybeVU = Union[float, ValueWithUnit, StatefulValueWithUnit, None]
HPRMetric = Union[float, list[float], np.ndarray, None]


__all__ = [
    "HPRMetric",
    "MaybeVU",
    "ScalarOrVU",
    "StatefulValueWithUnit",
    "ValueWithUnit",
]
