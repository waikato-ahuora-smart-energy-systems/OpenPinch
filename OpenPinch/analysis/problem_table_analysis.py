import numpy as np
from typing import Tuple
from ..utils.water_properties import Tsat_p
from ..lib import *
from ..classes import StreamCollection, ProblemTable, Stream
from .support_methods import insert_temperature_interval_into_pt, linear_interpolation

__all__ = ["problem_table_algorithm, calc_problem_table"]


#######################################################################################################
# Public API
#######################################################################################################

def problem_table_algorithm(hot_streams: StreamCollection, cold_streams: StreamCollection, all_streams: StreamCollection, config: Configuration) -> Tuple[ProblemTable, ProblemTable, dict]:
    """Perform the problem table algorithm for a given set of hot and cold streams."""
    # Get all possible temperature intervals, remove duplicates and order from high to low
    pt, pt_real = _get_temperature_intervals(streams=all_streams, config=config)
    
    # Perform the heat cascade of the problem table
    pt, pt_real = calc_problem_table(pt, hot_streams, cold_streams), calc_problem_table(pt_real, hot_streams, cold_streams, False)
    target_values = _set_zonal_targets(pt, pt_real)
    
    # Correct the location of the cold composite curve and limiting GCC (real temperatures)
    pt_real = _correct_pt_composite_curves(pt_real, target_values["heat_recovery_target"], target_values["heat_recovery_limit"])
    
    # Add additional temperature intervals for ease in targeting utility etc
    pt, pt_real = _add_temperature_intervals_at_constant_h(target_values["heat_recovery_target"], pt, pt_real)

    return pt, pt_real, target_values


def calc_problem_table(pt: ProblemTable, hot_streams: StreamCollection = None, cold_streams: StreamCollection = None, shifted: bool = True) -> ProblemTable:
    """Fast calculation of the problem table using vectorized operations."""

    # If streams are provided, calculate CP and RCP contributions per temperature interval
    if hot_streams is not None and cold_streams is not None:
        cp_hot, rcp_hot, cp_cold, rcp_cold = _sum_mcp_between_temperature_boundaries(
            pt.col[PT.T.value], hot_streams, cold_streams, shifted
        )
        pt.col[PT.CP_HOT.value] = cp_hot
        pt.col[PT.RCP_HOT.value] = rcp_hot
        pt.col[PT.CP_COLD.value] = cp_cold
        pt.col[PT.RCP_COLD.value] = rcp_cold

    # Extract numeric arrays for fast computation
    T_col = pt.col[PT.T.value]
    cp_hot = pt.col[PT.CP_HOT.value]
    cp_cold = pt.col[PT.CP_COLD.value]

    # ΔT = T_i - T_i+1, with first value set to 0.0
    delta_T = np.empty_like(T_col)
    delta_T[0] = 0.0
    delta_T[1:] = T_col[:-1] - T_col[1:]
    pt.col[PT.DELTA_T.value] = delta_T

    # ΔH_HOT = ΔT * CP_HOT
    delta_H_hot = delta_T * cp_hot
    H_hot = np.cumsum(delta_H_hot)
    pt.col[PT.DELTA_H_HOT.value] = delta_H_hot
    pt.col[PT.H_HOT.value] = H_hot

    # ΔH_COLD = ΔT * CP_COLD
    delta_H_cold = delta_T * cp_cold
    H_cold = np.cumsum(delta_H_cold)
    pt.col[PT.DELTA_H_COLD.value] = delta_H_cold
    pt.col[PT.H_COLD.value] = H_cold

    # MCP_NET = CP_COLD - CP_HOT
    mcp_net = cp_cold - cp_hot
    pt.col[PT.MCP_NET.value] = mcp_net

    # ΔH_NET = ΔT * MCP_NET
    delta_H_net = delta_T * mcp_net
    H_net = -np.cumsum(delta_H_net)
    pt.col[PT.DELTA_H_NET.value] = delta_H_net
    pt.col[PT.H_NET.value] = H_net

    # Find minimum H_NET (for cascade shifting)
    min_H = H_net.min()

    # Shift the composite curves for alignment
    pt.col[PT.H_HOT.value] = H_hot[-1] - H_hot
    pt.col[PT.H_COLD.value] = H_cold[-1] + (H_net[-1] - min_H) - H_cold
    pt.col[PT.H_NET.value] = H_net - min_H

    return pt


#######################################################################################################
# Helper functions
#######################################################################################################

def _get_temperature_intervals(streams: List[Stream] = [], config: Configuration = None) -> Tuple[ProblemTable, ProblemTable]:
    """Returns unordered T and T* intervals for given streams and utilities."""

    T_star = [t for s in streams for t in (s.t_min_star, s.t_max_star)]
    T = [t for s in streams for t in (s.t_min, s.t_max)]

    if isinstance(config, Configuration):
        if config.TURBINE_WORK_BUTTON:
            T_val = config.T_TURBINE_BOX
            Tsat_val = Tsat_p(config.P_TURBINE_BOX)
            T_star.extend([T_val, Tsat_val])
            T.extend([T_val, Tsat_val])

        T_star.append(config.TEMP_REF)
        T.append(config.TEMP_REF)

    pt = ProblemTable({PT.T.value: sorted(set(T_star), reverse=True)})
    pt_real = ProblemTable({PT.T.value: sorted(set(T), reverse=True)})

    return pt, pt_real


def _sum_mcp_between_temperature_boundaries(
    temperatures: List[float],
    hot_streams: List[Stream],
    cold_streams: List[Stream],
    shifted: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Vectorized CP and rCP summation across temperature intervals."""

    def calc_active_matrix(streams: List[Stream], use_shifted: bool) -> np.ndarray:
        t_min = np.array([s.t_min_star if use_shifted else s.t_min for s in streams])
        t_max = np.array([s.t_max_star if use_shifted else s.t_max for s in streams])

        lower_bounds = np.array(temperatures[1:])
        upper_bounds = np.array(temperatures[:-1])

        # Shape: (intervals, streams)
        active = (t_max[np.newaxis, :] > lower_bounds[:, np.newaxis] + tol) & \
                 (t_min[np.newaxis, :] < upper_bounds[:, np.newaxis] - tol)

        return active

    def sum_cp_rcp(streams: List[Stream], active: np.ndarray):
        cp = np.array([s.CP for s in streams])
        rcp = np.array([s.rCP for s in streams])
        cp_sum = active @ cp
        rcp_sum = active @ rcp
        return np.insert(cp_sum, 0, 0.0), np.insert(rcp_sum, 0, 0.0)

    hot_active = calc_active_matrix(hot_streams, shifted)
    cold_active = calc_active_matrix(cold_streams, shifted)

    cp_hot, rcp_hot = sum_cp_rcp(hot_streams, hot_active)
    cp_cold, rcp_cold = sum_cp_rcp(cold_streams, cold_active)

    return cp_hot.tolist(), rcp_hot.tolist(), cp_cold.tolist(), rcp_cold.tolist()


def _correct_pt_composite_curves(pt: ProblemTable, heat_recovery_target: float, heat_recovery_limit: float) -> ProblemTable:
    """Shift H_COLD and H_NET columns to match targeted heat recovery levels."""
    delta_shift = heat_recovery_limit - heat_recovery_target
    pt.col[PT.H_COLD.value] += delta_shift
    pt.col[PT.H_NET.value] += delta_shift
    return pt


def _add_temperature_intervals_at_constant_h(heat_recovery_target: float, pt: ProblemTable, pt_real: ProblemTable):
    if heat_recovery_target > tol:
        pt = _insert_temperature_interval_into_pt_at_constant_h(pt, PT.T.value, PT.H_HOT.value, PT.H_COLD.value)
        pt_real = _insert_temperature_interval_into_pt_at_constant_h(pt_real, PT.T.value, PT.H_HOT.value, PT.H_COLD.value)
    return pt, pt_real


def _insert_temperature_interval_into_pt_at_constant_h(pt: ProblemTable, col_T: str =PT.T.value, col_HCC: str =PT.H_HOT.value, col_CCC: str =PT.H_COLD.value) -> ProblemTable:
    """Insert temperature intervals into the process table where HCC and CCC intersect at constant enthalpy."""
    # --- HCC to CCC projection ---
    i = _get_composite_curve_starting_points_loc(pt, col_HCC, hcc=True)
    pt = _get_T_value_at_cc_starts(pt, i, col_T, col_HCC, col_CCC, hcc=True)
    # --- CCC to HCC projection ---
    i = _get_composite_curve_starting_points_loc(pt, col_CCC, hcc=False)
    pt = _get_T_value_at_cc_starts(pt, i, col_T, col_HCC, col_CCC, hcc=False)
    return pt


def _get_composite_curve_starting_points_loc(pt: ProblemTable, col: str = PT.H_HOT.value, hcc: bool = True) -> int:
    """Find the starting index of the composite curve to be projected."""
    i_range = range(1, len(pt)) if hcc else range(len(pt) - 1, 0, -1)
    for i in i_range:
        if pt.loc[i - 1, col] > pt.loc[i, col] + tol:
            break
    return i - 1 if hcc else i


def _get_T_value_at_cc_starts(pt: ProblemTable, start_index: int, col_T: str =PT.T.value, col_HCC: str =PT.H_HOT.value, col_CCC: str =PT.H_COLD.value, hcc: bool =True) -> ProblemTable:
    """Find and insert the temperature value where composite curves intersect at constant enthalpy."""
    col0, col1 = (col_HCC, col_CCC) if hcc else (col_CCC, col_HCC)

    # T_insert_vals = []
    i = start_index
    k = 0
    while 0 < i < len(pt) and k < 2:
        h_0 = pt.loc[i, col0]
        j_range = range(len(pt) - 1, 0, -1) if hcc else range(1, len(pt) - 1)

        for j in j_range:
            if abs(pt.loc[j, col1] - h_0) < tol:
                k += 1
                break

            z = 0 if hcc else 1
            h_j = pt.loc[j + z, col1]
            h_jm1 = pt.loc[j - 1 + z, col1]

            if h_0 < h_jm1 - tol and h_0 > h_j + tol:
                t_j = pt.loc[j + z, col_T]
                t_jm1 = pt.loc[j - 1 + z, col_T]
                T_C = linear_interpolation(h_0, h_j, h_jm1, t_j, t_jm1)
                # T_insert_vals.append(T_C)
                pt, n_int_added = insert_temperature_interval_into_pt(pt, [T_C])
                k += n_int_added
                break
        i = j
    
    # for T_C in T_insert_vals:
    #     pt = insert_temperature_interval_into_pt(pt, T_C)

    return pt


def _set_zonal_targets(pt: ProblemTable, pt_real: ProblemTable) -> dict:
    """Assign thermal targets and integration degree to the zone based on process table data."""
    return {
        "hot_utility_target": pt.loc[0, PT.H_NET.value], 
        "cold_utility_target": pt.loc[-1, PT.H_NET.value],
        "heat_recovery_target": pt.loc[0, PT.H_HOT.value] - pt.loc[-1, PT.H_NET.value],
        "heat_recovery_limit": pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value],
        "degree_of_int": (
            (pt.loc[0, PT.H_HOT.value] - pt.loc[-1, PT.H_NET.value]) / (pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value])
            if (pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value]) > 0 else 1.0
        )
    }
