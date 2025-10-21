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
    if abs((H_hot[0] - H_hot[-1]) - (H_cold[0] - H_cold[-1])) > tol:
        # Raise an error due to heat flow imbalance, which is a requirement for this analysis. 
        raise ValueError("The temperature driving force plot requires the inputted composite curves to be balanced.")

    if abs(H_hot[0] - H_cold[0]) > tol:
        # Shift the hot and cold cascades to start from zero at the lowest temperature. 
        H_hot = H_hot - H_hot[-1]
        H_cold = H_cold - H_cold[-1]

    # Collect H_val intervals and sort
    h_vals = pd.Series(H_hot[:-1].tolist() + H_cold[:-1].tolist()).sort_values().reset_index(drop=True) # TODO: check this code
    # h_vals = sorted(set(H_hot[:-1].tolist() + H_cold[:-1].tolist()), reverse=False)
    h_start = h_vals[:-1].values
    h_end = h_vals[1:].values
    dh_vals = h_start - h_end

    # Interpolate temperatures for each H at both ends
    t_h1 = np.interp(h_start, H_hot, T_hot)
    t_h2 = np.interp(h_end, H_hot, T_hot)
    t_c1 = np.interp(h_start, H_cold, T_cold)
    t_c2 = np.interp(h_end, H_cold, T_cold)

    delta_T1 = (t_h1 - t_c1) - min_dT
    delta_T2 = (t_h2 - t_c2) - min_dT

    return {
        "h_vals": h_vals, 
        "delta_T1": delta_T1, 
        "delta_T2": delta_T2, 
        "dh_vals": dh_vals,
    }


#######################################################################################################
# Helper functions
#######################################################################################################


