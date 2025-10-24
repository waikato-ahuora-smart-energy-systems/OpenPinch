"""Area targeting methods."""
import pandas as pd
import numpy as np 

from ..classes import *
from ..lib import *
from ..utils import *
from ..analysis.temperature_driving_force import get_temperature_driving_forces

__all__ = ["get_balanced_CC", "get_area_targets"]

#######################################################################################################
# Public API
#######################################################################################################


def get_balanced_CC(
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    H_hot_ut: np.ndarray,
    H_cold_ut: np.ndarray,
    dT_vals: np.ndarray = None,
    RCP_hot: np.ndarray = None,
    RCP_cold: np.ndarray = None,
    RCP_hot_ut: np.ndarray = None,
    RCP_cold_ut: np.ndarray = None,    
) -> ProblemTable:
    """Creates the balanced Composite Curve (CC) using both process and utility streams."""
    H_hot_bal = H_hot + H_hot_ut
    H_cold_bal = H_cold + H_cold_ut
    res = {
        PT.H_HOT_BAL.value: H_hot_bal,
        PT.H_COLD_BAL.value: H_cold_bal,
    }
    
    if (
        RCP_hot is not None and
        RCP_cold is not None and
        RCP_hot_ut is not None and
        RCP_cold_ut is not None
    ):
        dH_hot_bal = np.insert(
            H_hot_bal[:-1] - H_hot_bal[1:],
            0,
            0,
        )
        dH_cold_bal = np.insert(
            H_cold_bal[:-1] - H_cold_bal[1:],
            0,
            0,
        )
        R_hot_bal = np.where(
            dH_hot_bal > 0,
            (RCP_hot + RCP_hot_ut) * dT_vals / dH_hot_bal,
            0.0,
        )
        R_cold_bal = np.where(
            dH_cold_bal > 0,
            (RCP_cold + RCP_cold_ut) * dT_vals / dH_cold_bal,
            0.0,
        )
        res.update(
            {
                PT.H_HOT_BAL.value: H_hot + H_hot_ut,
                PT.H_COLD_BAL.value: H_cold + H_cold_ut,
                PT.RCP_HOT_BAL.value: RCP_hot + RCP_hot_ut,
                PT.RCP_COLD_BAL.value: RCP_cold + RCP_cold_ut,
                PT.R_HOT_BAL.value: R_hot_bal,
                PT.R_COLD_BAL.value: R_cold_bal,
            }
        )
    return res


def get_area_targets(
    T_vals: np.ndarray,
    H_hot_bal: np.ndarray,
    H_cold_bal: np.ndarray,
    R_HOT_BAL: np.ndarray,
    R_COLD_BAL: np.ndarray,   
) -> dict:
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized numpy operations."""
    if abs((H_hot_bal[0] - H_hot_bal[-1]) - (H_cold_bal[0] - H_cold_bal[-1])) > tol:
        # Raise an error due to heat flow imbalance, which is a requirement for this analysis. 
        raise ValueError("The temperature driving force plot requires the inputted composite curves to be balanced.")

    # Shift the hot and cold cascades to start from zero at the lowest temperature. 
    if abs(H_hot_bal[0]) > tol: #if not np.isclose(H_cold[0], 0, tol):
        H_hot_bal = H_hot_bal - H_hot_bal[-1]
    if abs(H_cold_bal[0]) > tol:
        H_cold_bal = H_cold_bal - H_cold_bal[-1]    

    # Find HTC value for each temperature interval
    tdf = get_temperature_driving_forces(
        T_vals, H_hot_bal, T_vals, H_cold_bal, 
    )
    dt_lm_i = compute_LMTD_from_dts(
        tdf["delta_T1"],
        tdf["delta_T2"],
    )
    Q_i = tdf["dh_vals"]
    U_i = 1
    area_i = Q_i / (U_i * dt_lm_i)

    return area_i.sum()



