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


def get_area_targets() -> float: # pt: ProblemTable
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized numpy operations."""
    if pt.col[PT.H_HOT_BAL.value] - pt.col[PT.H_COLD_BAL.value] > tol:
        raise ValueError("Balanced Composite Curves are imbalanced.")
    
    # Find HTC value for each temperature interval
    pt.col[PT.RCP_HOT_BAL.value] = pt.col[PT.RCP_HOT.value] + pt.col[PT.RCP_HOT_UT.value]
    pt.col[PT.RCP_COLD_BAL.value] = pt.col[PT.RCP_COLD.value] + pt.col[PT.RCP_COLD_UT.value]

    dH_bal_pt = pt.col[PT.H_HOT_BAL.value][1:] - pt.col[PT.H_COLD_BAL.value][:-1]

    pt.col[PT.R_HOT_BAL.value] = pt.col[PT.RCP_HOT_BAL.value] / dH_bal_pt if dH_bal_pt > 0 else 0
    pt.col[PT.R_COLD_BAL.value] = pt.col[PT.RCP_COLD_BAL.value] / dH_bal_pt if dH_bal_pt > 0 else 0

    tdf = get_temperature_driving_forces(
        pt.col[PT.T.value], pt.col[PT.H_HOT_BAL.value], pt.col[PT.T.value], pt.col[PT.H_COLD_BAL.value], 
    )
    dt_lm_i = compute_LMTD_from_dts(
        tdf["delta_T1"],
        tdf["delta_T2"],
    )
    Q_i = tdf["dh_vals"]
    U_i = 1
    area_i = Q_i / (U_i * dt_lm_i)

    return {
        "area": area_i.sum(),
        "area_i": area_i,
        "Q_i": Q_i,
        "U_i": U_i,
        "dt_lm_i": dt_lm_i,
    }

    # dh_vals = tdf["dh_vals"]

    

    # r_hot = np.interp(h_end, pt['HCC'], pt['RH'])
    # r_cold = np.interp(h_end, pt['CCC'], pt['RC'])
    # u_o = 1 / (r_hot + r_cold)

    # area_segments = ntu * cp_min / u_o
    # total_area = np.sum(area_segments)

    # return float(total_area)
    return 0
