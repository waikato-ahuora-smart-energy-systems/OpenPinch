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
    dp = int(-math.log10(tol))
    T_hot = np.asarray(T_hot, dtype=float).round(dp)
    H_hot = np.asarray(H_hot, dtype=float).round(dp)
    T_cold = np.asarray(T_cold, dtype=float).round(dp)
    H_cold = np.asarray(H_cold, dtype=float).round(dp)

    if T_hot.size != H_hot.size or T_cold.size != H_cold.size:
        raise ValueError("Composite curve temperature and heat arrays must be the same length.")
    if T_hot.size == 0 or T_cold.size == 0:
        raise ValueError("Composite curve arrays cannot be empty.")
    if abs((np.max(H_hot) - np.min(H_hot)) - (np.max(H_cold) - np.min(H_cold))) > tol:
        raise ValueError("The temperature driving force plot requires the inputted composite curves to be balanced.")

    H_hot, T_hot = _normalise_curve(H_hot, T_hot)
    H_cold, T_cold = _normalise_curve(H_cold, T_cold)

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
                delta_T2_raw[idx] = min(delta_T2_raw[idx], delta_T2_raw[idx + 1])

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


def _normalise_curve(h_vals: np.ndarray, t_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure a composite curve has ascending heat flow and starts at zero."""
    if h_vals.size != t_vals.size:
        raise ValueError("Composite curve temperature and heat arrays must be the same length.")

    # Orient so heat flow increases from lower to higher values
    if h_vals[0] > h_vals[-1]:
        h_vals = h_vals[::-1]
        t_vals = t_vals[::-1]

    # Shift the curve so it begins at zero heat load
    offset = h_vals[0]
    if abs(offset) > tol:
        h_vals = h_vals - offset

    return h_vals, t_vals


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
