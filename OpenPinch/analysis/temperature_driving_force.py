"""Temperature driving force method."""
import pandas as pd
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

    # Collect H_val intervals and sort
    h_vals = _build_h_grid(H_hot, H_cold)
    h_start = h_vals.iloc[:-1].to_numpy()
    h_end = h_vals.iloc[1:].to_numpy()
    dh_vals = h_start - h_end

    # Interpolate temperatures for each H at both ends
    t_h1 = _interp_with_plateaus(H_hot, T_hot, h_start, keep="first")
    t_h2 = _interp_with_plateaus(H_hot, T_hot, h_end, keep="last")
    t_c1 = _interp_with_plateaus(H_cold, T_cold, h_start, keep="first")
    t_c2 = _interp_with_plateaus(H_cold, T_cold, h_end, keep="last")

    delta_T1 = (t_h1 - t_c1) - min_dT
    delta_T2 = (t_h2 - t_c2) - min_dT

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

def _build_h_grid(h_hot: np.ndarray, h_cold: np.ndarray) -> pd.Series:
    """Create a heat flow grid that preserves discontinuities."""
    return np.array(
        set(h_hot[:].tolist() + h_cold[:].tolist())
    ).sort() 


def _interp_with_plateaus(h_vals: np.ndarray, t_vals: np.ndarray, targets: np.ndarray, keep: str) -> np.ndarray:
    """Interpolate temperatures while respecting vertical segments in the composite curves."""
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")

    h_vals = np.asarray(h_vals, dtype=float)
    t_vals = np.asarray(t_vals, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if h_vals.size == 1:
        return np.full_like(targets, t_vals[0], dtype=float)

    deltas = np.abs(np.diff(h_vals))
    if keep == "first":
        mask = np.empty_like(h_vals, dtype=bool)
        mask[0] = True
        mask[1:] = deltas > tol
    else:
        mask = np.empty_like(h_vals, dtype=bool)
        mask[-1] = True
        mask[:-1] = deltas > tol

    h_monotonic = h_vals[mask]
    t_monotonic = t_vals[mask]

    if h_monotonic.size == 1:
        return np.full_like(targets, t_monotonic[0], dtype=float)

    return np.interp(targets, h_monotonic, t_monotonic)

