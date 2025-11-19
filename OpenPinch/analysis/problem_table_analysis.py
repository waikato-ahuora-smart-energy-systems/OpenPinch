"""Problem-table generation and cascade utilities for pinch analysis.

This module collects the vectorised implementations of the temperature
interval problem table, cascade shifting logic, and helper routines used when
deriving zonal targets and plotting data for composite curves.
"""

from typing import Dict, Tuple

import numpy as np

from ..classes import *
from ..lib import *
from ..utils import *


__all__ = ["get_process_heat_cascade", "get_utility_heat_cascade", "create_problem_table_with_t_int"]


#######################################################################################################
# Public API
#######################################################################################################


def get_process_heat_cascade(
    hot_streams: StreamCollection = StreamCollection(),
    cold_streams: StreamCollection = StreamCollection(),
    all_streams: StreamCollection = StreamCollection(),
    zone_config: Configuration = None,
    include_real_pt: bool = True,
) -> Tuple[ProblemTable, ProblemTable, dict]:
    """Prepare, calculate and analyse the problem table for a given set of hot and cold streams."""
    # Get all possible temperature intervals, remove duplicates and order from high to low
    pt = create_problem_table_with_t_int(
        all_streams,
        True,
        zone_config,
    )
    # Perform the heat cascade of the problem table
    _problem_table_algorithm(pt, hot_streams, cold_streams)

    if not include_real_pt or zone_config == None:
        return pt, None, None
    
    pt_real = create_problem_table_with_t_int(
        all_streams, 
        False,
        zone_config,
    )

    _problem_table_algorithm(pt_real, hot_streams, cold_streams, False)
    target_values = _set_zonal_targets(pt, pt_real)

    # Correct the location of the cold composite curve and limiting GCC (real temperatures)
    _shift_pt_real_composite_curves(
        pt_real,
        target_values["heat_recovery_target"],
        target_values["heat_recovery_limit"],
    )

    # Add additional temperature intervals for ease in targeting utility etc
    _add_temperature_intervals_at_constant_h(pt, target_values["heat_recovery_target"])
    _add_temperature_intervals_at_constant_h(pt_real, target_values["heat_recovery_target"])

    return pt, pt_real, target_values


def get_utility_heat_cascade(
    T_int_vals: np.ndarray,
    hot_utilities: List[Stream],
    cold_utilities: List[Stream],
    is_shifted: bool = True,
) -> Dict[str, np.ndarray]:
    """Prepare and calculate the utility heat cascade a given set of hot and cold utilities."""
    pt_ut = ProblemTable({PT.T.value: T_int_vals})
    _problem_table_algorithm(pt_ut, hot_utilities, cold_utilities, is_shifted)

    h_net_values = pt_ut.col[PT.H_NET.value]
    H_NET_UT = h_net_values.max() - h_net_values
    
    h_ut_cc =  pt_ut.col[PT.H_HOT.value]
    c_ut_cc =  pt_ut.col[PT.H_COLD.value] - pt_ut.col[PT.H_COLD.value].max()

    return {
        PT.H_NET_UT.value: H_NET_UT,
        PT.H_HOT_UT.value: h_ut_cc,
        PT.H_COLD_UT.value: c_ut_cc,
        PT.RCP_HOT_UT.value: pt_ut.col[PT.RCP_HOT.value],
        PT.RCP_COLD_UT.value: pt_ut.col[PT.RCP_COLD.value],
        PT.RCP_UT_NET.value: pt_ut.col[PT.RCP_HOT.value] + pt_ut.col[PT.RCP_COLD.value],
    }


def create_problem_table_with_t_int(
    streams: List[Stream] | StreamCollection = [], 
    is_shifted: bool = True,
    zone_config: Configuration = None,
) -> Tuple[ProblemTable, ProblemTable]:
    """Returns ordered T and T* intervals for given streams and utilities."""

    T_vals = [
        t 
        for s in streams 
        for t in ((s.t_min_star, s.t_max_star) if is_shifted else (s.t_min, s.t_max))
    ]

    if isinstance(zone_config, Configuration):
        if zone_config.DO_TURBINE_WORK:
            T_vals.extend([
                zone_config.T_TURBINE_BOX, 
                Tsat_p(zone_config.P_TURBINE_BOX),
            ])

        if zone_config.DO_EXERGY_TARGETING:
            T_vals.append(zone_config.T_ENV)

        if zone_config.DO_PROCESS_HP_TARGETING or zone_config.DO_UTILITY_HP_TARGETING:
            T_vals.append(zone_config.T_ENV - zone_config.DT_ENV_CONT)
            T_vals.append(zone_config.T_ENV - zone_config.DT_ENV_CONT - zone_config.DT_PHASE_CHANGE)
            T_vals.append(zone_config.T_ENV + zone_config.DT_ENV_CONT)
            T_vals.append(zone_config.T_ENV + zone_config.DT_ENV_CONT + zone_config.DT_PHASE_CHANGE)

    dp = int(-math.log10(tol))
    T_vals = np.array(T_vals).round(dp)
    return ProblemTable(
        {
            PT.T.value: sorted(set(T_vals), reverse=True)
        }
    )


#######################################################################################################
# Helper functions
#######################################################################################################


def _problem_table_algorithm(
    pt: ProblemTable,
    hot_streams: StreamCollection = None,
    cold_streams: StreamCollection = None,
    is_shifted: bool = True,
) -> ProblemTable:
    """Fast calculation of the problem table using vectorized cascade formulas."""

    # Sum m_dot*Cp contributions from hot streams per interval (sets CP_HOT and rCP_HOT)
    if hot_streams is not None:
        sum_cp_hot, sum_rcp_hot = _sum_mcp_between_temperature_boundaries(
            pt.col[PT.T.value], hot_streams, is_shifted
        )
        pt.col[PT.CP_HOT.value] = sum_cp_hot
        pt.col[PT.RCP_HOT.value] = sum_rcp_hot

    # Sum m_dot*Cp contributions from cold streams per interval (sets CP_COLD and rCP_COLD)
    if cold_streams is not None:
        sum_cp_cold, sum_rcp_cold = _sum_mcp_between_temperature_boundaries(
            pt.col[PT.T.value], cold_streams, is_shifted
        )    
        pt.col[PT.CP_COLD.value] = sum_cp_cold
        pt.col[PT.RCP_COLD.value] = sum_rcp_cold

    # ΔT_i = T_{i-1} - T_i
    pt.col[PT.DELTA_T.value] = delta_with_zero_at_start(pt.col[PT.T.value])
    
    # ΔH_hot = ΔT * CP_hot
    pt.col[PT.DELTA_H_HOT.value] = pt.col[PT.DELTA_T.value] * pt.col[PT.CP_HOT.value]

    # H_hot = Σ ΔH_hot
    pt.col[PT.H_HOT.value] = np.cumsum(pt.col[PT.DELTA_H_HOT.value])

    # ΔH_cold = ΔT * CP_cold
    pt.col[PT.DELTA_H_COLD.value] = pt.col[PT.DELTA_T.value] * pt.col[PT.CP_COLD.value]

    # H_cold = Σ ΔH_cold
    pt.col[PT.H_COLD.value] = np.cumsum(pt.col[PT.DELTA_H_COLD.value])

    # CP_NET = CP_cold - CP_hot
    pt.col[PT.CP_NET.value] = pt.col[PT.CP_COLD.value] - pt.col[PT.CP_HOT.value]

    # ΔH_net = ΔT * CP_NET
    pt.col[PT.DELTA_H_NET.value] = pt.col[PT.DELTA_T.value] * pt.col[PT.CP_NET.value]

    # H_net = -Σ ΔH_net
    pt.col[PT.H_NET.value] = -np.cumsum(pt.col[PT.DELTA_H_NET.value])

    # Shift cascades so the minimum H_net equals zero, aligning hot/cold composites at the pinch
    min_H = pt.col[PT.H_NET.value].min()
    shift = pt.col[PT.H_NET.value][-1] - min_H

    pt.col[PT.H_HOT.value] = pt.col[PT.H_HOT.value][-1] - pt.col[PT.H_HOT.value]
    pt.col[PT.H_COLD.value] = pt.col[PT.H_COLD.value][-1] + shift - pt.col[PT.H_COLD.value]
    pt.col[PT.H_NET.value] = pt.col[PT.H_NET.value] - min_H

    return pt


def _sum_mcp_between_temperature_boundaries(
    temperatures: List[float],
    streams: List[Stream],
    is_shifted: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Vectorized CP and rCP summation across temperature intervals."""

    def calc_active_matrix(streams: List[Stream], use_shifted: bool) -> np.ndarray:
        t_min = np.array([s.t_min_star if use_shifted else s.t_min for s in streams])
        t_max = np.array([s.t_max_star if use_shifted else s.t_max for s in streams])

        lower_bounds = np.array(temperatures[1:])
        upper_bounds = np.array(temperatures[:-1])

        # Shape: (intervals, streams)
        active = (t_max[np.newaxis, :] > lower_bounds[:, np.newaxis] + tol) & (
            t_min[np.newaxis, :] < upper_bounds[:, np.newaxis] - tol
        )

        return active

    def sum_cp_rcp(streams: List[Stream], active: np.ndarray):
        cp = np.array([s.CP for s in streams])
        rcp = np.array([s.rCP for s in streams])
        cp_sum = active @ cp
        rcp_sum = active @ rcp
        return np.insert(cp_sum, 0, 0.0), np.insert(rcp_sum, 0, 0.0)
    
    is_active = calc_active_matrix(streams, is_shifted)
    cp_array, rcp_array = sum_cp_rcp(streams, is_active)
    return cp_array.tolist(), rcp_array.tolist()


def _shift_pt_real_composite_curves(
    pt: ProblemTable, heat_recovery_target: float, heat_recovery_limit: float
) -> ProblemTable:
    """Shift H_COLD and H_NET columns to match targeted heat recovery levels."""
    delta_shift = heat_recovery_limit - heat_recovery_target
    pt.col[PT.H_COLD.value] += delta_shift
    pt.col[PT.H_NET.value] += delta_shift
    return pt


def _add_temperature_intervals_at_constant_h(
    pt: ProblemTable,
    heat_recovery_target: float,
) -> ProblemTable:
    """Insert breakpoints where composite curves intersect at constant enthalpy."""
    if heat_recovery_target > tol:
        _insert_temperature_interval_into_pt_at_constant_h(pt)


def _insert_temperature_interval_into_pt_at_constant_h(
        pt: ProblemTable
) -> ProblemTable:
    """Insert temperature intervals into the process table where HCC and CCC intersect at constant enthalpy."""
    T_new = []
    # --- HCC to CCC projection ---
    if pt.col[PT.H_HOT.value][0] > 0.0:
        T_new.append(
            _get_T_start_on_opposite_cc(pt, pt.col[PT.H_HOT.value][0], PT.H_COLD.value)
        )
    # --- CCC to HCC projection ---
    if pt.col[PT.H_COLD.value][-1] > 0.0:
        T_new.append(
            _get_T_start_on_opposite_cc(pt, pt.col[PT.H_COLD.value][-1], PT.H_HOT.value)
        )
    T_new = [t for t in T_new if t is not None]
    if len(T_new) > 0:
        pt.insert_temperature_interval(T_new)
    return pt


def _get_T_start_on_opposite_cc(
    pt: ProblemTable,
    h0: float,
    col_CC: str,
    col_T: str = PT.T.value,
) -> float:
    """Find and insert the temperature value where composite curves intersect at constant enthalpy."""

    temp_cc = pt.col[col_CC] - h0

    if temp_cc.size < 2:
        return None

    zero_hits = np.flatnonzero(np.abs(temp_cc) < tol)
    if zero_hits.size > 0:
        return None

    transitions = np.flatnonzero((temp_cc[:-1] >= tol) & (temp_cc[1:] <= -tol))

    if transitions.size != 1:
        return None

    idx = int(transitions[0])

    cc_vals = pt.col[col_CC]
    T_vals = pt.col[col_T]

    T_new = linear_interpolation( 
        h0,
        cc_vals[idx], cc_vals[idx + 1],
        T_vals[idx], T_vals[idx + 1],
    )

    return T_new


def _set_zonal_targets(pt: ProblemTable, pt_real: ProblemTable) -> dict:
    """Assign thermal targets and integration degree to the zone based on process table data."""
    return {
        "hot_utility_target": pt.loc[0, PT.H_NET.value],
        "cold_utility_target": pt.loc[-1, PT.H_NET.value],
        "heat_recovery_target": pt.loc[0, PT.H_HOT.value] - pt.loc[-1, PT.H_NET.value],
        "heat_recovery_limit": pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value],
        "degree_of_int": (
            (pt.loc[0, PT.H_HOT.value] - pt.loc[-1, PT.H_NET.value]) / (pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value])
            if (pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value]) > 0
            else 1.0
        ),
    }
