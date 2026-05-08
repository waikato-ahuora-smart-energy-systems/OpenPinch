"""Common schema primitives and shared type aliases."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field

class ValueWithUnit(BaseModel):
    """Container storing a magnitude and its associated unit string."""

    value: Optional[float] = Field(
        default=None, description="Numeric value (magnitude)."
    )
    units: str = Field(..., description="Unit string, e.g. 'kW', '°C', 'kJ/s'.")


ScalarOrVU = Union[float, ValueWithUnit]
MaybeVU = Union[float, ValueWithUnit, None]
HPRMetric = Union[float, list[float], np.ndarray, None]


__all__ = ["HPRMetric", "MaybeVU", "ScalarOrVU", "ValueWithUnit"]
