"""Problem-table generation and cascade utilities for pinch analysis.

This module collects the vectorised implementations of the temperature
interval problem table, cascade shifting logic, and helper routines used when
deriving zonal targets and plotting data for composite curves.
"""

from typing import Tuple

import numpy as np

from ..classes import *
from ..lib import *
from ..utils import *
from .support_methods import *

__all__ = ["get_process_heat_cascade", "get_utility_heat_cascade"]


#######################################################################################################
# Public API
#######################################################################################################

@timing_decorator
def get_process_heat_cascade(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    all_streams: StreamCollection,
    config: Configuration,
) -> Tuple[ProblemTable, ProblemTable, dict]:
    """Prepare, calculate and analyse the problem table for a given set of hot and cold streams."""
    # Get all possible temperature intervals, remove duplicates and order from high to low
    pt, pt_real = _get_temperature_intervals(streams=all_streams, config=config)

    # Perform the heat cascade of the problem table
    pt = _problem_table_algorithm(pt, hot_streams, cold_streams)
    pt_real = _problem_table_algorithm(pt_real, hot_streams, cold_streams, False)
    target_values = _set_zonal_targets(pt, pt_real)

    # Correct the location of the cold composite curve and limiting GCC (real temperatures)
    pt_real = _shift_pt_real_composite_curves(
        pt_real,
        target_values["heat_recovery_target"],
        target_values["heat_recovery_limit"],
    )

    # Add additional temperature intervals for ease in targeting utility etc
    pt, pt_real = _add_temperature_intervals_at_constant_h(
        target_values["heat_recovery_target"], pt, pt_real
    )

    return pt, pt_real, target_values

@timing_decorator
def get_utility_heat_cascade(
    pt: ProblemTable,
    hot_utilities: List[Stream],
    cold_utilities: List[Stream],
    is_shifted: bool = True,
) -> ProblemTable:
    """Prepare and calculate the utility heat cascade a given set of hot and cold utilities."""
    pt_temp = _problem_table_algorithm(
        ProblemTable({PT.T.value: pt.col[PT.T.value]}),
        hot_utilities,
        cold_utilities,
        is_shifted,
    )
    pt.col[PT.H_UT_NET.value] = (
        pt_temp.col[PT.H_NET.value].max() - pt_temp.col[PT.H_NET.value]
    )
    pt.col[PT.RCP_HOT_UT.value] = pt_temp.col[PT.RCP_HOT.value]
    pt.col[PT.RCP_COLD_UT.value] = pt_temp.col[PT.RCP_COLD.value]
    pt.col[PT.RCP_UT_NET.value] = (
        pt_temp.col[PT.RCP_HOT.value] + pt_temp.col[PT.RCP_COLD.value]
    )
    return pt


#######################################################################################################
# Helper functions
#######################################################################################################


def _problem_table_algorithm(
    pt: ProblemTable,
    hot_streams: StreamCollection = None,
    cold_streams: StreamCollection = None,
    is_shifted: bool = True,
) -> ProblemTable:
    
    if hot_streams is not None:
        cp_hot, rcp_hot = _sum_mcp_between_temperature_boundaries(
            pt.col[PT.T.value], hot_streams, is_shifted
        )
        pt.col[PT.CP_HOT.value] = cp_hot
        pt.col[PT.RCP_HOT.value] = rcp_hot

    if cold_streams is not None:
        cp_cold, rcp_cold = _sum_mcp_between_temperature_boundaries(
            pt.col[PT.T.value], cold_streams, is_shifted
        )    
        pt.col[PT.CP_COLD.value] = cp_cold
        pt.col[PT.RCP_COLD.value] = rcp_cold

    pt.col[PT.DELTA_T.value] = pt.delta_col(PT.T.value)

    pt.col[PT.DELTA_H_HOT.value] = pt.col[PT.DELTA_T.value] * pt.col[PT.CP_HOT.value]

    pt.col[PT.DELTA_H_COLD.value] = pt.col[PT.DELTA_T.value] * pt.col[PT.CP_COLD.value]

    pt.col[PT.H_HOT.value] = np.cumsum(pt.col[PT.DELTA_H_HOT.value])
    pt.col[PT.H_COLD.value] = np.cumsum(pt.col[PT.DELTA_H_COLD.value])

    pt.col[PT.MCP_NET.value] = pt.col[PT.CP_COLD.value] - pt.col[PT.CP_HOT.value]
    pt.col[PT.DELTA_H_NET.value] = pt.col[PT.DELTA_T.value] * pt.col[PT.MCP_NET.value]
    pt.col[PT.H_NET.value] = -np.cumsum(pt.col[PT.DELTA_H_NET.value])

    min_H = pt.col[PT.H_NET.value].min()
    shift = pt.col[PT.H_NET.value][-1] - min_H

    pt.col[PT.H_HOT.value] = pt.col[PT.H_HOT.value][-1] - pt.col[PT.H_HOT.value]
    pt.col[PT.H_COLD.value] = pt.col[PT.H_COLD.value][-1] + shift - pt.col[PT.H_COLD.value]
    pt.col[PT.H_NET.value] = pt.col[PT.H_NET.value] - min_H

    return pt


def _get_temperature_intervals(
    streams: List[Stream] = [], config: Configuration = None
) -> Tuple[ProblemTable, ProblemTable]:
    """Returns unordered T and T* intervals for given streams and utilities."""

    T_star = [t for s in streams for t in (s.t_min_star, s.t_max_star)]
    T = [t for s in streams for t in (s.t_min, s.t_max)]

    if isinstance(config, Configuration):
        if config.DO_TURBINE_WORK:
            T_val = config.T_TURBINE_BOX
            Tsat_val = Tsat_p(config.P_TURBINE_BOX)
            T_star.extend([T_val, Tsat_val])
            T.extend([T_val, Tsat_val])

        if config.DO_EXERGY_TARGETING:
            T_star.append(config.T_ENV)
            T.append(config.T_ENV)

    pt = ProblemTable({PT.T.value: sorted(set(T_star), reverse=True)})
    pt_real = ProblemTable({PT.T.value: sorted(set(T), reverse=True)})

    return pt, pt_real


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
    heat_recovery_target: float, pt: ProblemTable, pt_real: ProblemTable
):
    """Insert breakpoints where composite curves intersect at constant enthalpy."""
    if heat_recovery_target > tol:
        _insert_temperature_interval_into_pt_at_constant_h(pt)
        _insert_temperature_interval_into_pt_at_constant_h(pt_real)
    return pt, pt_real


def _insert_temperature_interval_into_pt_at_constant_h(pt: ProblemTable) -> ProblemTable:
    """Insert temperature intervals into the process table where HCC and CCC intersect at constant enthalpy."""
    # --- HCC to CCC projection ---
    if pt.col[PT.H_HOT.value][0] > 0.0:
        _insert_cc_start_on_opposite_cc(pt, pt.col[PT.H_HOT.value][0], PT.H_COLD.value)
    # --- CCC to HCC projection ---
    if pt.col[PT.H_COLD.value][-1] > 0.0:
        _insert_cc_start_on_opposite_cc(pt, pt.col[PT.H_COLD.value][-1], PT.H_HOT.value)
    return pt


def _insert_cc_start_on_opposite_cc(
    pt: ProblemTable,
    h0: float,
    col_CC: str,
    col_T: str = PT.T.value,
) -> ProblemTable:
    """Find and insert the temperature value where composite curves intersect at constant enthalpy."""

    temp_cc = pt.col[col_CC] - h0

    if temp_cc.size < 2:
        return pt

    zero_hits = np.flatnonzero(np.abs(temp_cc) < tol)
    if zero_hits.size > 0:
        return pt

    transitions = np.flatnonzero((temp_cc[:-1] >= tol) & (temp_cc[1:] <= -tol))

    if transitions.size != 1:
        pass
        return pt

    idx = int(transitions[0])

    cc_vals = pt.col[col_CC]
    T_vals = pt.col[col_T]

    T_new = linear_interpolation( 
        h0,
        cc_vals[idx],
        cc_vals[idx + 1],
        T_vals[idx],
        T_vals[idx + 1],
    )

    pt, _ = insert_temperature_interval_into_pt(pt, [T_new])
    return pt



def _set_zonal_targets(pt: ProblemTable, pt_real: ProblemTable) -> dict:
    """Assign thermal targets and integration degree to the zone based on process table data."""
    return {
        "hot_utility_target": pt.loc[0, PT.H_NET.value],
        "cold_utility_target": pt.loc[-1, PT.H_NET.value],
        "heat_recovery_target": pt.loc[0, PT.H_HOT.value] - pt.loc[-1, PT.H_NET.value],
        "heat_recovery_limit": pt_real.loc[0, PT.H_HOT.value]
        - pt_real.loc[-1, PT.H_NET.value],
        "degree_of_int": (
            (pt.loc[0, PT.H_HOT.value] - pt.loc[-1, PT.H_NET.value])
            / (pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value])
            if (pt_real.loc[0, PT.H_HOT.value] - pt_real.loc[-1, PT.H_NET.value]) > 0
            else 1.0
        ),
    }
