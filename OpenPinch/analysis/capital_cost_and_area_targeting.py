"""Area targeting methods."""
import pandas as pd
import numpy as np 

from ..classes import *
from ..lib import *
from ..utils import *
from .temperature_driving_force import *

__all__ = ["get_balanced_CC", "get_capital_cost_targets", "get_area_targets", "get_min_number_hx"]


#######################################################################################################
# Public API
#######################################################################################################


def get_capital_cost_targets(
    area: float,
    num_units: int,
    zone_config: Configuration,
) -> dict:
    capital_cost = compute_capital_cost(
        area, 
        num_units, 
        zone_config.FIXED_COST, 
        zone_config.VARIABLE_COST,
        zone_config.COST_EXP,
    )
    annual_capital_cost = compute_annual_capital_cost(
        capital_cost,
        zone_config.DISCOUNT_RATE,
        zone_config.SERV_LIFE,
    )    
    return capital_cost, annual_capital_cost


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
    Q_i: np.ndarray = tdf["dh_vals"]
    R_i = _map_interval_resistances_to_tdf(
        T_vals,
        R_hot_bal,
        R_cold_bal,
        tdf["t_h1"],
        tdf["t_h2"],
        tdf["t_c1"],
        tdf["t_c2"],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        U_i = np.where(R_i > tol, 1.0 / R_i, 1.0)
    if not(Q_i.shape == U_i.shape == dt_lm_i.shape):
        raise ValueError("Shape of heat exchanger area calculation arrays are unequal.")
    area_i = Q_i / (U_i * dt_lm_i)
    return area_i.sum()


def get_min_number_hx(
    T_vals: np.ndarray,
    H_hot_bal: np.ndarray,
    H_cold_bal: np.ndarray,
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
) -> int:
    """Estimates the minimum number of heat exchangers required for the pinch problem using vectorized interval logic.
    """
    num_hx: int = 0
    H_net_bal = H_cold_bal - H_hot_bal
    mask = np.isclose(H_net_bal, 0.0, atol=tol)
    mask_true_positions = np.flatnonzero(mask).tolist()
    idx_pairs = []
    for i in range(len(mask_true_positions) - 1):
        if mask_true_positions[i] + 1 < mask_true_positions[i+1]:
            idx_pairs.append(
                (mask_true_positions[i], mask_true_positions[i+1])
            )

    for i0, i1 in idx_pairs:
        T_high, T_low = T_vals[i0], T_vals[i1]
        num_hx += _count_crossing(T_low, T_high, hot_streams)
        num_hx += _count_crossing(T_low, T_high, cold_streams)        
        num_hx += _count_utility_range_container(T_low, T_high, hot_utilities)
        num_hx += _count_utility_range_container(T_low, T_high, cold_utilities)

    return int(num_hx- len(idx_pairs))


#######################################################################################################
# Helper functions
#######################################################################################################


def _map_interval_resistances_to_tdf(
    T_vals: np.ndarray,
    R_hot_bal: np.ndarray,
    R_cold_bal: np.ndarray,
    t_h1: np.ndarray,
    t_h2: np.ndarray,
    t_c1: np.ndarray,
    t_c2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align total hot/cold resistances with temperature-driving-force intervals using a vectorised mask.

    Returns:
        tuple[np.ndarray, np.ndarray]: (total_resistance, mask) where total_resistance has the same
        length as the temperature driving force intervals and mask is a boolean matrix that maps
        each interval to the corresponding temperature band in ``T_vals``.
    """
    interval_lower = T_vals[1:]
    interval_upper = T_vals[:-1]

    active_hot = (
        (t_h1[np.newaxis, :] >= interval_lower[:, np.newaxis] - tol)
        & 
        (t_h2[np.newaxis, :] <= interval_upper[:, np.newaxis] + tol)
    )
    active_cold = (
        (t_c1[np.newaxis, :] >= interval_lower[:, np.newaxis] - tol)
        & 
        (t_c2[np.newaxis, :] <= interval_upper[:, np.newaxis] + tol)
    )
    R_hot_mat = np.ones(shape=active_hot.shape) * R_hot_bal[1:, np.newaxis]
    R_cold_mat = np.ones(shape=active_hot.shape) * R_cold_bal[1:, np.newaxis]
    Rh = (active_hot * R_hot_mat).sum(axis=0)
    Rc = (active_cold * R_cold_mat).sum(axis=0)
    return Rh + Rc


def _count_crossing(T_low: float, T_high: float, streams: StreamCollection):
    """Count process streams whose adjusted temperatures intersect interval [T_low, T_high]."""
    t_max = np.array([s.t_max_star for s in streams])
    t_min = np.array([s.t_min_star for s in streams])
    return np.sum(
        ((t_max > T_low + tol) & (t_max <= T_high + tol))
        | ((t_min >= T_low - tol) & (t_min < T_high - tol))
        | ((t_min < T_low - tol) & (t_max > T_high + tol))
    )


def _count_utility_range_container(T_low: float, T_high: float, utilities: StreamCollection):
    """Count utility streams whose adjusted temperatures intersect interval [T_low, T_high]."""
    t_max = np.array([u.t_max_star for u in utilities])
    t_min = np.array([u.t_min_star for u in utilities])
    active = np.array([1 if u.heat_flow > tol else 0 for u in utilities])
    return np.sum(
        (t_min >= T_low - tol) & (t_max <= T_high + tol) & (active)
    )
