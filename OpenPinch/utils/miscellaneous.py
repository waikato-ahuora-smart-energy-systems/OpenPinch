"""Shared numerical helpers."""

from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt

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


def clean_composite_curve_ends(
    y_vals: np.ndarray | list, x_vals: np.ndarray | list
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove redundant points in composite curves."""
    mask_0 = np.isclose(x_vals, x_vals[0] * np.ones(len(x_vals)), atol=tol)
    mask_1 = np.isclose(x_vals, x_vals[-1] * np.ones(len(x_vals)), atol=tol)
    if mask_1.sum() > 1:
        x_clean = x_vals[mask_0.sum()-1 : -mask_1.sum()+1]
        y_clean = y_vals[mask_0.sum()-1 : -mask_1.sum()+1]
    else:
        x_clean = x_vals[mask_0.sum()-1 : ]
        y_clean = y_vals[mask_0.sum()-1 : ]        
    return y_clean, x_clean


def clean_composite_curve(
    y_array: np.ndarray, x_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove redundant points in composite curves."""

    # Round to avoid tiny numerical errors
    y_vals, x_vals = clean_composite_curve_ends(y_array, x_array)

    if len(x_vals) <= 2:
        return y_vals, x_vals

    x_clean, y_clean = [x_vals[0]], [y_vals[0]]

    for i in range(1, len(x_vals) - 1):
        x1, x2, x3 = x_vals[i - 1], x_vals[i], x_vals[i + 1]
        y1, y2, y3 = y_vals[i - 1], y_vals[i], y_vals[i + 1]

        if x1 == x3:
            # All three x are the same; keep x2 only if y2 is different
            if x1 != x2:
                x_clean.append(x2)
                y_clean.append(y2)
        else:
            # Linear interpolation check
            y_interp = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
            if abs(y2 - y_interp) > tol:
                x_clean.append(x2)
                y_clean.append(y2)

    x_clean.append(x_vals[-1])
    y_clean.append(y_vals[-1])

    if abs(x_clean[0] - x_clean[1]) < tol:
        x_clean.pop(0)
        y_clean.pop(0)

    i = len(x_clean) - 1
    if abs(x_clean[i] - x_clean[i - 1]) < tol:
        x_clean.pop(i)
        y_clean.pop(i)

    return y_clean, x_clean

def graph_simple_cc_plot(Tc, Hc, Th, Hh):
    fig, ax = plt.subplots()
    ax.plot(Hc, Tc, label="Cold composite")
    ax.plot(Hh, Th, label="Hot composite")
    ax.set_ylabel("Temperature")
    ax.set_xlabel("Enthalpy")
    ax.set_title("Balanced Composite Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()