import pandas as pd
import numpy as np 
from ..utils import *
from ..lib import *
from ..classes import *

__all__ = ["get_area_targets"]

#######################################################################################################
# Public API --- TODO: check accuracy
#######################################################################################################

def get_area_targets(pt: ProblemTable, config: Configuration) -> float:
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized numpy operations."""
    # if abs(pt['HCC'].iloc[0] - pt['CCC'].iloc[0]) > tol:
    #     raise ValueError("Balanced Composite Curves are imbalanced.")

    # # Collect H_val intervals and sort
    # h_vals = pd.Series(pt['HCC'].iloc[:-1].tolist() + pt['CCC'].iloc[:-1].tolist()).sort_values().reset_index(drop=True)
    # h_start = h_vals[:-1].values
    # h_end = h_vals[1:].values
    # dh = h_start - h_end

    # # Interpolate temperatures for each H at both ends
    # t_h1 = np.interp(h_start, pt['HCC'], pt['T'])
    # t_h2 = np.interp(h_end, pt['HCC'], pt['T'])
    # t_c1 = np.interp(h_start, pt['CCC'], pt['T'])
    # t_c2 = np.interp(h_end, pt['CCC'], pt['T'])

    # delta_T1 = t_h1 - t_c1
    # delta_T2 = t_h2 - t_c2

    # t_lmtd = np.where(
    #     abs(delta_T1 - delta_T2) < 1e-6,
    #     (delta_T1 + delta_T2) / 2,
    #     (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
    # )

    # cp_hot = dh / (t_h1 - t_h2)
    # cp_cold = dh / (t_c1 - t_c2)
    # cp_min = np.minimum(cp_hot, cp_cold)
    # cp_max = np.maximum(cp_hot, cp_cold)

    # eff = dh / (cp_min * (t_h1 - t_c2))
    # cp_star = cp_min / cp_max

    # if config.CF_SELECTED:
    #     arrangement = HX.CF.value
    # elif config.PF_SELECTED:
    #     arrangement = HX.PF.value
    # else:
    #     arrangement = HX.ShellTube.value

    # ntu = np.vectorize(HX_NTU)(arrangement, eff, cp_star)

    # r_hot = np.interp(h_end, pt['HCC'], pt['RH'])
    # r_cold = np.interp(h_end, pt['CCC'], pt['RC'])
    # u_o = 1 / (r_hot + r_cold)

    # area_segments = ntu * cp_min / u_o
    # total_area = np.sum(area_segments)

    # return float(total_area)
    return 0