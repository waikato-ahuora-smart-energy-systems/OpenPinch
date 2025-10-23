"""Temperature driving force method."""
import numpy as np 

from ..classes import *
from ..lib import *
from ..utils import *

__all__ = ["get_temperature_driving_forces"]

#######################################################################################################
# Public API
#######################################################################################################

def get_temperature_driving_forces(
        T_hot: np.array, 
        H_hot: np.array, 
        T_cold: np.array, 
        H_cold: np.array,
        min_dT: float = 0,
) -> float:
    """Determines the temeprature driving force plot using vectorized numpy operations."""
    T_hot = np.asarray(T_hot, dtype=float)
    H_hot = np.asarray(H_hot, dtype=float)
    T_cold = np.asarray(T_cold, dtype=float)
    H_cold = np.asarray(H_cold, dtype=float)

    if abs((H_hot[0] - H_hot[-1]) - (H_cold[0] - H_cold[-1])) > tol:
        # Raise an error due to heat flow imbalance, which is a requirement for this analysis. 
        raise ValueError("The temperature driving force plot requires the inputted composite curves to be balanced.")

    if abs(H_hot[0] - H_cold[0]) > tol:
        # Shift the hot and cold cascades to start from zero at the lowest temperature. 
        H_hot = H_hot - H_hot[-1]
        H_cold = H_cold - H_cold[-1]

    # Collect a unified heat-load grid across both curves
    h_vals = _build_h_grid(H_hot, H_cold)
    h_start = h_vals[:-1]
    h_end = h_vals[1:]
    dh_vals = h_end - h_start

    # Interpolate temperatures for each H at both ends
    t_h1 = _interp_with_plateaus(H_hot, T_hot, h_start, side="right")
    t_h2 = _interp_with_plateaus(H_hot, T_hot, h_end, side="left")
    t_c1 = _interp_with_plateaus(H_cold, T_cold, h_start, side="right")
    t_c2 = _interp_with_plateaus(H_cold, T_cold, h_end, side="left")

    delta_T1_raw = t_h1 - t_c1
    delta_T2_raw = t_h2 - t_c2

    discontinuities = _collect_discontinuities(H_hot, H_cold)
    if discontinuities:
        for idx in range(len(delta_T2_raw) - 2, -1, -1):
            if _is_discontinuity(h_end[idx], discontinuities):
                delta_T2_raw[idx] = max(delta_T2_raw[idx], delta_T2_raw[idx + 1])

    delta_T1 = delta_T1_raw - min_dT
    delta_T2 = delta_T2_raw - min_dT

    return {
        "h_vals": h_vals, 
        "delta_T1": delta_T1, 
        "delta_T2": delta_T2, 
        "dh_vals": dh_vals,
        "t_h1": t_h1,
        "t_h2": t_h2,
        "t_c1": t_c1,
        "t_c2": t_c2, 
    }


#######################################################################################################
# Helper functions
#######################################################################################################

def _build_h_grid(h_hot: np.ndarray, h_cold: np.ndarray) -> np.ndarray:
    """Create a unified, sorted heat-flow grid across both curves."""
    return np.union1d(h_hot, h_cold).astype(float, copy=False)


def _interp_with_plateaus(
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    targets: np.ndarray,
    side: str,
) -> np.ndarray:
    """Interpolate temperatures while respecting vertical segments in the composite curves."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    h_vals = np.asarray(h_vals, dtype=float)
    t_vals = np.asarray(t_vals, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if h_vals.size == 1:
        return np.full_like(targets, t_vals[0], dtype=float)

    h_monotonic = _make_monotonic(h_vals, side)
    return np.interp(targets, h_monotonic, t_vals)


def _make_monotonic(h_vals: np.ndarray, side: str) -> np.ndarray:
    """Adjust an array so repeated values become strictly increasing for interpolation."""
    adjusted = np.array(h_vals, dtype=float, copy=True)
    if adjusted.size <= 1:
        return adjusted

    eps = tol * 0.5 if tol > 0 else 1e-9

    idx = 0
    n = adjusted.size
    while idx < n - 1:
        j = idx + 1
        while j < n and abs(adjusted[j] - adjusted[idx]) <= tol:
            j += 1
        if j - idx > 1:
            length = j - idx
            if side == "right":
                offsets = np.arange(length - 1, -1, -1, dtype=float) * eps
                adjusted[idx:j] = adjusted[idx] - offsets
            else:  # side == "left"
                offsets = np.arange(length, dtype=float) * eps
                adjusted[idx:j] = adjusted[idx] + offsets
        idx = j

    return adjusted


def _collect_discontinuities(h_hot: np.ndarray, h_cold: np.ndarray) -> set[float]:
    """Identify heat loads where either curve has a vertical segment."""
    return set(_discontinuity_values(h_hot)) | set(_discontinuity_values(h_cold))


def _discontinuity_values(h_vals: np.ndarray) -> np.ndarray:
    """Return the heat loads associated with zero-width segments."""
    if h_vals.size < 2:
        return np.empty(0, dtype=float)
    mask = np.isclose(np.diff(h_vals), 0.0, atol=tol)
    if not mask.any():
        return np.empty(0, dtype=float)
    return h_vals[1:][mask]


def _is_discontinuity(value: float, discontinuities: set[float]) -> bool:
    """Check if a heat load corresponds to a discontinuity."""
    for disc in discontinuities:
        if abs(value - disc) <= tol:
            return True
    return False
