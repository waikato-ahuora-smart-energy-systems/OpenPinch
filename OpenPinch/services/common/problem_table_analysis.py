"""Problem-table generation and cascade utilities for pinch analysis.

This module collects the vectorised implementations of the temperature
interval problem table, cascade shifting logic, and helper routines used when
deriving zonal targets and plotting data for composite curves.
"""

import math
from typing import List, Tuple

import numpy as np

from ...classes.stream_collection import StreamCollection
from ...classes.problem_table import ProblemTable
from ...lib.config import tol
from ...lib.enums import PT
from ...lib.problem_table_types import ProblemTableUpdateKwargs
from ...services.common.gcc_manipulation import get_additional_GCCs
from ...utils.miscellaneous import delta_with_zero_at_start, linear_interpolation


__all__ = [
    "get_process_heat_cascade",
    "get_utility_heat_cascade",
    "create_problem_table_with_t_int",
    "problem_table_algorithm",
    "get_heat_recovery_target_from_pt",
    "set_zonal_targets",
]


#######################################################################################################
# Public API
#######################################################################################################


def get_process_heat_cascade(
    hot_streams: StreamCollection = None,
    cold_streams: StreamCollection = None,
    all_streams: StreamCollection = None,
    is_shifted: bool = True,
    known_heat_recovery: float = None,
    extra_T_intervals: list = None,
    is_full_analysis: bool = False,
) -> ProblemTable:
    """Prepare, calculate and analyse the problem table for a given set of hot and cold streams."""
    if hot_streams is None:
        hot_streams = StreamCollection()
    if cold_streams is None:
        cold_streams = StreamCollection()
    if extra_T_intervals is None:
        extra_T_intervals = []

    # Get all possible temperature intervals, remove duplicates and order from high to low
    if all_streams is None:
        all_streams = hot_streams + cold_streams

    pt = create_problem_table_with_t_int(
        streams=all_streams,
        is_shifted=is_shifted,
        extra_T_intervals=extra_T_intervals,
    )
    # Perform the heat cascade of the problem table
    problem_table_algorithm(
        pt=pt,
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        is_shifted=is_shifted,
    )

    if isinstance(known_heat_recovery, float):
        # Correct the location of the cold composite curve and limiting GCC (real temperatures)
        _shift_pt_to_set_heat_recovery(
            pt=pt,
            heat_recovery_target=known_heat_recovery,
            current_heat_recovery=get_heat_recovery_target_from_pt(pt),
        )

    heat_recovery_target = get_heat_recovery_target_from_pt(pt)
    if heat_recovery_target > tol:
        # Add additional temperature intervals for ease in targeting utility etc
        _insert_temperature_interval_into_pt_at_constant_h(pt)

    if is_full_analysis:
        get_additional_GCCs(pt, is_process_stream=True)

    return pt


def get_utility_heat_cascade(
    T_int_vals: np.ndarray,
    hot_utilities: StreamCollection = None,
    cold_utilities: StreamCollection = None,
    is_shifted: bool = True,
) -> ProblemTableUpdateKwargs:
    """Prepare and calculate the utility heat cascade a given set of hot and cold utilities."""
    pt_ut = ProblemTable({PT.T: T_int_vals})
    problem_table_algorithm(pt_ut, hot_utilities, cold_utilities, is_shifted)

    h_net_values = pt_ut[PT.H_NET]
    H_NET_UT = h_net_values.max() - h_net_values

    h_ut_cc = pt_ut[PT.H_HOT]
    c_ut_cc = pt_ut[PT.H_COLD] - pt_ut[PT.H_COLD].max()

    return {
        "T_col": T_int_vals,
        "updates": {
            PT.H_NET_UT: H_NET_UT,
            PT.H_HOT_UT: h_ut_cc,
            PT.H_COLD_UT: c_ut_cc,
            PT.RCP_HOT_UT: pt_ut[PT.RCP_HOT],
            PT.RCP_COLD_UT: pt_ut[PT.RCP_COLD],
            PT.RCP_UT_NET: pt_ut[PT.RCP_HOT] + pt_ut[PT.RCP_COLD],
        },
    }


def create_problem_table_with_t_int(
    streams: StreamCollection = None,
    is_shifted: bool = True,
    extra_T_intervals: list | float | np.ndarray = None,
) -> ProblemTable:
    """Return a problem table populated with ordered unique temperature intervals."""
    if streams is None:
        streams = StreamCollection()
    if extra_T_intervals is None:
        extra_T_intervals = []
    if isinstance(extra_T_intervals, (int, float)):
        extra_T_intervals = [extra_T_intervals]
    elif isinstance(extra_T_intervals, np.ndarray):
        extra_T_intervals = extra_T_intervals.tolist()

    T_vals = [
        t
        for s in streams
        for t in ((s.t_min_star, s.t_max_star) if is_shifted else (s.t_min, s.t_max))
    ] + extra_T_intervals
    dp = int(-math.log10(tol))
    T_vals = np.array(T_vals).round(dp)
    return ProblemTable({PT.T: sorted(set(T_vals), reverse=True)})


def problem_table_algorithm(
    pt: ProblemTable,
    hot_streams: StreamCollection = None,
    cold_streams: StreamCollection = None,
    is_shifted: bool = True,
) -> ProblemTable:
    """Fast calculation of the problem table using vectorized cascade formulas."""

    # Sum m_dot*Cp contributions from hot streams per interval (sets CP_HOT and rCP_HOT)
    if hot_streams is not None:
        sum_cp_hot, sum_rcp_hot = _sum_mcp_between_temperature_boundaries(
            pt[PT.T], hot_streams, is_shifted
        )
        pt[PT.CP_HOT] = sum_cp_hot
        pt[PT.RCP_HOT] = sum_rcp_hot
    else:
        pt[PT.CP_HOT].fill(0.0)
        pt[PT.RCP_HOT].fill(0.0)

    # Sum m_dot*Cp contributions from cold streams per interval (sets CP_COLD and rCP_COLD)
    if cold_streams is not None:
        sum_cp_cold, sum_rcp_cold = _sum_mcp_between_temperature_boundaries(
            pt[PT.T], cold_streams, is_shifted
        )
        pt[PT.CP_COLD] = sum_cp_cold
        pt[PT.RCP_COLD] = sum_rcp_cold
    else:
        pt[PT.CP_COLD].fill(0.0)
        pt[PT.RCP_COLD].fill(0.0)

    # ΔT_i = T_{i-1} - T_i
    pt[PT.DELTA_T] = delta_with_zero_at_start(pt[PT.T])

    # ΔH_hot = ΔT * CP_hot
    pt[PT.DELTA_H_HOT] = pt[PT.DELTA_T] * pt[PT.CP_HOT]

    # H_hot = Σ ΔH_hot
    pt[PT.H_HOT] = np.cumsum(pt[PT.DELTA_H_HOT])

    # ΔH_cold = ΔT * CP_cold
    pt[PT.DELTA_H_COLD] = pt[PT.DELTA_T] * pt[PT.CP_COLD]

    # H_cold = Σ ΔH_cold
    pt[PT.H_COLD] = np.cumsum(pt[PT.DELTA_H_COLD])

    # CP_NET = CP_cold - CP_hot
    pt[PT.CP_NET] = pt[PT.CP_COLD] - pt[PT.CP_HOT]

    # ΔH_net = ΔT * CP_NET
    pt[PT.DELTA_H_NET] = pt[PT.DELTA_T] * pt[PT.CP_NET]

    # H_net = -Σ ΔH_net
    pt[PT.H_NET] = -np.cumsum(pt[PT.DELTA_H_NET])

    # Shift cascades so the minimum H_net equals zero, aligning hot/cold composites at the pinch
    min_H = pt[PT.H_NET].min()
    shift = pt[PT.H_NET][-1] - min_H

    pt[PT.H_HOT] = pt[PT.H_HOT][-1] - pt[PT.H_HOT]
    pt[PT.H_COLD] = pt[PT.H_COLD][-1] + shift - pt[PT.H_COLD]
    pt[PT.H_NET] = pt[PT.H_NET] - min_H

    return pt


def get_heat_recovery_target_from_pt(
    pt: ProblemTable,
) -> float:
    """Compute the heat-recovery target implied by a solved problem table.

    Parameters
    ----------
    pt:
        Problem table with populated ``H_hot`` and ``H_net`` columns.

    Returns
    -------
    float
        Maximum direct heat recovery for the analysed zone.
    """
    return pt.loc[0, PT.H_HOT] - pt.loc[-1, PT.H_NET]


def set_zonal_targets(
    pt: ProblemTable,
    pt_real: ProblemTable,
) -> dict:
    """Assign thermal targets and integration degree to the zone based on process table data."""
    return {
        "hot_utility_target": pt.loc[0, PT.H_NET],
        "cold_utility_target": pt.loc[-1, PT.H_NET],
        "heat_recovery_target": pt.loc[0, PT.H_HOT] - pt.loc[-1, PT.H_NET],
        "heat_recovery_limit": pt_real.loc[0, PT.H_HOT] - pt_real.loc[-1, PT.H_NET],
        "degree_of_int": (
            (pt.loc[0, PT.H_HOT] - pt.loc[-1, PT.H_NET])
            / (pt_real.loc[0, PT.H_HOT] - pt_real.loc[-1, PT.H_NET])
            if (pt_real.loc[0, PT.H_HOT] - pt_real.loc[-1, PT.H_NET]) > 0
            else 1.0
        ),
    }


#######################################################################################################
# Helper functions
#######################################################################################################


def _sum_mcp_between_temperature_boundaries(
    temperatures: List[float],
    streams: StreamCollection,
    is_shifted: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Vectorized CP and rCP summation across temperature intervals."""

    def calc_active_matrix(streams: StreamCollection, use_shifted: bool) -> np.ndarray:
        t_min = np.array([s.t_min_star if use_shifted else s.t_min for s in streams])
        t_max = np.array([s.t_max_star if use_shifted else s.t_max for s in streams])

        lower_bounds = np.array(temperatures[1:])
        upper_bounds = np.array(temperatures[:-1])

        # Shape: (intervals, streams)
        active = (t_max[np.newaxis, :] > lower_bounds[:, np.newaxis] + tol * 10) & (
            t_min[np.newaxis, :] < upper_bounds[:, np.newaxis] - tol * 10
        )

        return active

    def sum_cp_rcp(streams: StreamCollection, active: np.ndarray):
        cp = np.array([s.CP for s in streams])
        rcp = np.array([s.rCP for s in streams])
        cp_sum = active @ cp
        rcp_sum = active @ rcp
        return np.insert(cp_sum, 0, 0.0), np.insert(rcp_sum, 0, 0.0)

    is_active = calc_active_matrix(streams, is_shifted)
    cp_array, rcp_array = sum_cp_rcp(streams, is_active)
    return cp_array.tolist(), rcp_array.tolist()


def _shift_pt_to_set_heat_recovery(
    pt: ProblemTable, heat_recovery_target: float, current_heat_recovery: float
) -> ProblemTable:
    """Shift H_COLD and H_NET columns to match targeted heat recovery levels."""
    delta_shift = current_heat_recovery - heat_recovery_target
    pt[PT.H_COLD] += delta_shift
    pt[PT.H_NET] += delta_shift
    return pt


def _insert_temperature_interval_into_pt_at_constant_h(
    pt: ProblemTable,
) -> ProblemTable:
    """Insert temperature intervals into the process table where HCC and CCC intersect at constant enthalpy."""
    T_new = []
    # --- HCC to CCC projection ---
    if pt[PT.H_HOT][0] > 0.0:
        T_new.append(_get_T_start_on_opposite_cc(pt, pt[PT.H_HOT][0], PT.H_COLD))
    # --- CCC to HCC projection ---
    if pt[PT.H_COLD][-1] > 0.0:
        T_new.append(_get_T_start_on_opposite_cc(pt, pt[PT.H_COLD][-1], PT.H_HOT))
    T_new = [t for t in T_new if t is not None]
    if len(T_new) > 0:
        pt.insert_temperature_interval(T_new)
    return pt


def _get_T_start_on_opposite_cc(
    pt: ProblemTable,
    h0: float,
    col_CC: str | PT,
    col_T: str | PT = PT.T,
) -> float:
    """Find and insert the temperature value where composite curves intersect at constant enthalpy."""

    temp_cc = pt[col_CC] - h0

    if temp_cc.size < 2:
        return None

    zero_hits = np.flatnonzero(np.abs(temp_cc) < tol)
    if zero_hits.size > 0:
        return None

    transitions = np.flatnonzero((temp_cc[:-1] >= tol) & (temp_cc[1:] <= -tol))

    if transitions.size != 1:
        return None

    idx = int(transitions[0])

    cc_vals = pt[col_CC]
    T_vals = pt[col_T]

    T_new = linear_interpolation(
        h0,
        cc_vals[idx],
        cc_vals[idx + 1],
        T_vals[idx],
        T_vals[idx + 1],
    )

    return T_new
