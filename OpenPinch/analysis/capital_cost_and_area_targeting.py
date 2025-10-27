"""Area targeting methods."""
import pandas as pd
import numpy as np 

from ..classes import *
from ..lib import *
from ..utils import *
from .temperature_driving_force import get_temperature_driving_forces

__all__ = ["get_balanced_CC", "get_capital_cost_and_area_targets", "get_area_targets", "get_min_number_hx"]

# TODO: Need to review get_min_number_hx which sets the target for the bumber of heat exchangers. 
#.      Need to compute captial cost estimates with a new method. 

#######################################################################################################
# Public API
#######################################################################################################


def get_capital_cost_and_area_targets(
    T_vals: np.ndarray,
    H_hot_bal: np.ndarray,
    H_cold_bal: np.ndarray,
    R_HOT_BAL: np.ndarray,
    R_COLD_BAL: np.ndarray,   
) -> dict:
    
    # get area
    # num_units = get_min_number_hx(pt)
    # capital_cost = compute_capital_cost(area, num_units, zone_config)
    # annual_capital_cost = compute_annual_capital_cost(area, num_units, zone_config)    

    return get_area_targets(
        T_vals,
        H_hot_bal,
        H_cold_bal,
        R_HOT_BAL,
        R_COLD_BAL,  
    )


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
        R_hot_bal = np.zeros_like(dH_hot_bal, dtype=float)
        mask_hot = dH_hot_bal > tol
        np.divide(
            (RCP_hot + RCP_hot_ut) * dT_vals,
            dH_hot_bal,
            out=R_hot_bal,
            where=mask_hot,
        )
        R_cold_bal = np.zeros_like(dH_cold_bal, dtype=float)
        mask_cold = dH_cold_bal > tol
        np.divide(
            (RCP_cold + RCP_cold_ut) * dT_vals,
            dH_cold_bal,
            out=R_cold_bal,
            where=mask_cold,
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
    R_hot_bal: np.ndarray,
    R_cold_bal: np.ndarray,   
) -> dict:
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized numpy operations."""
    if abs((H_hot_bal[0] - H_hot_bal[-1]) - (H_cold_bal[0] - H_cold_bal[-1])) > tol:
        # Raise an error due to heat flow imbalance, which is a requirement for this analysis. 
        raise ValueError("The temperature driving force plot requires the inputted composite curves to be balanced.")

    # Shift the hot and cold cascades to start from zero at the lowest temperature. 
    if abs(H_hot_bal[0]) > tol:
        H_hot_bal = H_hot_bal - H_hot_bal[-1]
    if abs(H_cold_bal[0]) > tol:
        H_cold_bal = H_cold_bal - H_cold_bal[-1]    

    Th, Hh = clean_composite_curve_ends(T_vals, H_hot_bal)
    Tc, Hc = clean_composite_curve_ends(T_vals, H_cold_bal)

    tdf = get_temperature_driving_forces(Th, Hh, Tc, Hc)
    dt_lm_i = compute_LMTD_from_dts(
        tdf["delta_T1"],
        tdf["delta_T2"],
    )
    Q_i = tdf["dh_vals"]

    # Find HTC value for each temperature interval
    U_i = 1
    
    area_i = Q_i / (U_i * dt_lm_i)

    return {
        "Total heat exchanger area target": area_i.sum()
    }


def get_min_number_hx(z, pt_df: ProblemTable, bcc_star_df: ProblemTable) -> int:
    """
    Estimates the minimum number of heat exchangers required for the pinch problem using vectorized interval logic.

    Args:
        z: Zone with hot/cold streams and utilities.
        pt_df (ProblemTable): Problem table DataFrame with temperature column.
        bcc_star_df (ProblemTable): Balanced Composite Curve data with 'CCC' and 'HCC'.

    Returns:
        int: Minimum number of exchangers.
    """
    T_vals = pt_df.iloc[:, 0].values
    CCC = bcc_star_df["CCC"].values
    HCC = bcc_star_df["HCC"].values

    num_hx = 0
    i = 0
    while i < len(T_vals) - 1:
        if abs(CCC[i + 1] - HCC[i + 1]) > tol:
            break
        i += 1

    i_1 = i
    i += 1

    while i < len(T_vals):
        i_0 = i_1
        if abs(CCC[i] - HCC[i]) < tol or i == len(T_vals) - 1:
            i_1 = i
            T_high, T_low = T_vals[i_0], T_vals[i_1]

            def count_crossing(streams):
                """Count process streams whose adjusted temperatures intersect interval [T_low, T_high]."""
                t_max = np.array([s.t_max_star for s in streams])
                t_min = np.array([s.t_min_star for s in streams])
                return np.sum(
                    ((t_max > T_low + tol) & (t_max <= T_high + tol))
                    | ((t_min >= T_low - tol) & (t_min < T_high - tol))
                    | ((t_min < T_low - tol) & (t_max > T_high + tol))
                )

            num_hx += count_crossing(z.hot_streams)
            num_hx += count_crossing(z.cold_streams)

            def count_utility_crossing(utilities):
                """Count utility streams whose adjusted temperatures intersect interval [T_low, T_high]."""
                t_max = np.array([u.t_max_star for u in utilities])
                t_min = np.array([u.t_min_star for u in utilities])
                return np.sum(
                    (t_max > T_low + tol) & (t_max <= T_high + tol)
                    | (t_min >= T_low - tol) & (t_min < T_high - tol)
                )

            num_hx += count_utility_crossing(z.hot_utilities)
            num_hx += count_utility_crossing(z.cold_utilities)
            num_hx -= 1

            j = i_1
            while j < len(T_vals) - 1:
                if abs(CCC[j + 1] - HCC[j + 1]) > tol:
                    break
                j += 1

            i = j
            i_1 = j

        i += 1

    return int(num_hx)
