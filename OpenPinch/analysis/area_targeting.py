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
) -> ProblemTable:
    """Creates the balanced Composite Curve (CC) using both process and utility streams."""
    return {
        PT.H_HOT_BAL.value: H_hot + H_hot_ut,
        PT.H_COLD_BAL.value: H_cold + H_cold_ut,
    }


def get_area_targets(
    T_vals: np.ndarray,
    H_hot_bal: np.ndarray,
    H_cold_bal: np.ndarray,
    RCP_hot: np.ndarray,
    RCP_cold: np.ndarray,
    RCP_hot_ut: np.ndarray,
    RCP_cold_ut: np.ndarray,    
) -> dict:
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized numpy operations."""
    if H_hot_bal - H_cold_bal > tol:
        raise ValueError("Balanced Composite Curves are imbalanced.")
    
    # Find HTC value for each temperature interval
    RCP_hot_bal = RCP_hot + RCP_hot_ut
    RCP_cold_bal = RCP_cold + RCP_cold_ut

    dH_bal_pt = H_hot_bal[1:] - H_cold_bal[:-1]

    R_hot_bal = RCP_hot_bal / dH_bal_pt if dH_bal_pt > 0 else 0
    R_cold_bal = RCP_cold_bal / dH_bal_pt if dH_bal_pt > 0 else 0

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

    return {
        PT.RCP_HOT_BAL.value: RCP_hot_bal,
        PT.RCP_COLD_BAL.value: RCP_cold_bal,
        PT.R_HOT_BAL.value: R_hot_bal,
        PT.R_COLD_BAL.value: R_cold_bal,
        "area": area_i.sum(),
    }