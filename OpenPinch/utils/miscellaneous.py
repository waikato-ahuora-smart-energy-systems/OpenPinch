"""Shared numerical helpers."""

from typing import Union
import numpy as np

from ..lib import *


def key_name(zone_name: str, target_type: str = TargetType.DI.value):
    """Compose the canonical dictionary key for storing zone targets."""
    return f"{zone_name}/{target_type}"


def get_value(val: Union[float, dict, ValueWithUnit]) -> float:
    """Extract a numeric value from raw floats, dict payloads, or :class:`ValueWithUnit`."""
    if isinstance(val, float):
        return val
    elif isinstance(val, dict):
        return val["value"]
    elif isinstance(val, ValueWithUnit):
        return val.value
    else:
        raise TypeError(
            f"Unsupported type: {type(val)}. Expected float, dict, or ValueWithUnit."
        )

def linear_interpolation(xi: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Performs linear interpolation to estimate y at a given x, using two known points (x1, y1) and (x2, y2)."""
    if x1 == x2:
        raise ValueError(
            "Cannot perform interpolation when x1 == x2 (undefined slope)."
        )
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    yi = m * xi + c
    return yi


def delta_with_zero_at_start(x: np.ndarray) -> np.ndarray:
    """Compute difference between successive entries in a column and include a zero in the first entry."""  
    return np.insert(
        delta_vals(x),
        0,
        0.0
    ) 


def delta_vals(x: np.ndarray, descending_vals: bool = True) -> np.ndarray:
    """Compute difference between successive entries in a column."""        
    return (
        x[:-1] - x[1:] 
        if descending_vals else
        x[1:] - x[:-1]
    )
