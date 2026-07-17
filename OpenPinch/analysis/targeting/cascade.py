"""Problem Table generation and cascade utilities for pinch analysis.

This module collects the vectorised implementations of the temperature
interval problem table, cascade shifting logic, and helper routines used when
deriving zonal targets and plotting data for composite curves.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ...domain._problem_table.types import ProblemTableUpdateKwargs
from ...domain.configuration import tol
from ...domain.enums import ProblemTableLabel
from ...domain.problem_table import ProblemTable
from ...domain.stream_collection import StreamCollection
from ..numerics import (
    delta_with_zero_at_start,
    linear_interpolation,
)
from .grand_composite import get_additional_GCCs

__all__ = [
    "get_process_heat_cascade",
    "get_utility_heat_cascade",
    "create_problem_table_with_t_int",
    "problem_table_algorithm",
    "get_heat_recovery_target_from_pt",
    "set_zonal_targets",
]


################################################################################
# Public API
################################################################################


def get_process_heat_cascade(
    hot_streams: StreamCollection = None,
    cold_streams: StreamCollection = None,
    all_streams: StreamCollection = None,
    is_shifted: bool = True,
    known_heat_recovery: float = None,
    extra_T_intervals: list = None,
    is_full_analysis: bool = False,
    period_idx: int | None = None,
) -> ProblemTable:
    """Prepare, calculate, and analyse the problem table for given streams."""
    if hot_streams is None:
        hot_streams = StreamCollection()
    if cold_streams is None:
        cold_streams = StreamCollection()
    if extra_T_intervals is None:
        extra_T_intervals = []

    # Get all possible temperature intervals, remove duplicates, and order high to low.
    if all_streams is None:
        all_streams = hot_streams + cold_streams

    pt = create_problem_table_with_t_int(
        streams=all_streams,
        is_shifted=is_shifted,
        extra_T_intervals=extra_T_intervals,
        period_idx=period_idx,
    )
    if len(pt) == 0:
        return pt

    # Perform the heat cascade of the problem table
    problem_table_algorithm(
        pt=pt,
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        is_shifted=is_shifted,
        idx=period_idx,
    )

    if isinstance(known_heat_recovery, float):
        # Correct the cold composite curve and limiting GCC at real temperatures.
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
    period_idx: int | None = None,
) -> ProblemTableUpdateKwargs:
    """Prepare and calculate the utility heat cascade for a utility set."""
    pt_ut = ProblemTable({ProblemTableLabel.T: T_int_vals})
    problem_table_algorithm(
        pt_ut,
        hot_utilities,
        cold_utilities,
        is_shifted,
        idx=period_idx,
    )

    h_net_values = pt_ut[ProblemTableLabel.H_NET]
    H_NET_UT = h_net_values.max() - h_net_values

    h_ut_cc = pt_ut[ProblemTableLabel.H_HOT]
    c_ut_cc = pt_ut[ProblemTableLabel.H_COLD] - pt_ut[ProblemTableLabel.H_COLD].max()

    return {
        "T_col": T_int_vals,
        "updates": {
            ProblemTableLabel.H_NET_UT: H_NET_UT,
            ProblemTableLabel.H_HOT_UT: h_ut_cc,
            ProblemTableLabel.H_COLD_UT: c_ut_cc,
            ProblemTableLabel.RCP_HOT_UT: pt_ut[ProblemTableLabel.RCP_HOT],
            ProblemTableLabel.RCP_COLD_UT: pt_ut[ProblemTableLabel.RCP_COLD],
            ProblemTableLabel.RCP_UT_NET: pt_ut[ProblemTableLabel.RCP_HOT]
            + pt_ut[ProblemTableLabel.RCP_COLD],
        },
    }


def create_problem_table_with_t_int(
    streams: StreamCollection = None,
    is_shifted: bool = True,
    extra_T_intervals: list | float | np.ndarray = None,
    period_idx: int | None = None,
) -> ProblemTable:
    """Return a problem table populated with ordered unique temperature intervals."""
    if streams is None:
        streams = StreamCollection()

    if isinstance(streams, StreamCollection):
        numeric = streams.segment_numeric_view(period_idx)
        if is_shifted:
            T_vals = np.concatenate((numeric.t_min_star, numeric.t_max_star))
        else:
            T_vals = np.concatenate((numeric.t_min, numeric.t_max))
    else:
        if is_shifted:
            T_vals = [
                float(t)
                for s in streams
                for t in (
                    s.shifted_minimum_temperature[period_idx],
                    s.shifted_maximum_temperature[period_idx],
                )
            ]
        else:
            T_vals = [
                float(t)
                for s in streams
                for t in (
                    s.minimum_temperature[period_idx],
                    s.maximum_temperature[period_idx],
                )
            ]
        T_vals = np.asarray(T_vals, dtype=float)

    if extra_T_intervals is not None:
        extra_vals = np.atleast_1d(np.asarray(extra_T_intervals, dtype=float))
        if extra_vals.size:
            T_vals = np.concatenate((T_vals, extra_vals))

    if T_vals.size == 0:
        return ProblemTable({ProblemTableLabel.T: []})

    dp = int(-math.log10(tol))
    T_vals = np.round(T_vals[np.isfinite(T_vals)], dp)
    return ProblemTable({ProblemTableLabel.T: np.unique(T_vals)[::-1]})


def problem_table_algorithm(
    pt: ProblemTable,
    hot_streams: StreamCollection = None,
    cold_streams: StreamCollection = None,
    is_shifted: bool = True,
    idx: int | None = None,
) -> ProblemTable:
    """Fast calculation of the problem table using vectorized cascade formulas."""

    # Sum m_dot*Cp contributions from hot streams per interval (sets CP_HOT and rCP_HOT)
    if hot_streams is not None:
        pt[ProblemTableLabel.CP_HOT], pt[ProblemTableLabel.RCP_HOT] = (
            _sum_mcp_between_temperature_boundaries(
                pt[ProblemTableLabel.T], hot_streams, is_shifted, idx=idx
            )
        )
    else:
        pt[ProblemTableLabel.CP_HOT].fill(0.0)
        pt[ProblemTableLabel.RCP_HOT].fill(0.0)

    # Sum m_dot*Cp contributions from cold streams per interval.
    if cold_streams is not None:
        pt[ProblemTableLabel.CP_COLD], pt[ProblemTableLabel.RCP_COLD] = (
            _sum_mcp_between_temperature_boundaries(
                pt[ProblemTableLabel.T], cold_streams, is_shifted, idx=idx
            )
        )
    else:
        pt[ProblemTableLabel.CP_COLD].fill(0.0)
        pt[ProblemTableLabel.RCP_COLD].fill(0.0)

    # ΔT_i = T_{i-1} - T_i
    pt[ProblemTableLabel.DELTA_T] = delta_with_zero_at_start(pt[ProblemTableLabel.T])

    # ΔH_hot = ΔT * CP_hot
    pt[ProblemTableLabel.DELTA_H_HOT] = (
        pt[ProblemTableLabel.DELTA_T] * pt[ProblemTableLabel.CP_HOT]
    )

    # H_hot = Σ ΔH_hot
    pt[ProblemTableLabel.H_HOT] = np.cumsum(pt[ProblemTableLabel.DELTA_H_HOT])

    # ΔH_cold = ΔT * CP_cold
    pt[ProblemTableLabel.DELTA_H_COLD] = (
        pt[ProblemTableLabel.DELTA_T] * pt[ProblemTableLabel.CP_COLD]
    )

    # H_cold = Σ ΔH_cold
    pt[ProblemTableLabel.H_COLD] = np.cumsum(pt[ProblemTableLabel.DELTA_H_COLD])

    # CP_NET = CP_cold - CP_hot
    pt[ProblemTableLabel.CP_NET] = (
        pt[ProblemTableLabel.CP_COLD] - pt[ProblemTableLabel.CP_HOT]
    )

    # ΔH_net = ΔT * CP_NET
    pt[ProblemTableLabel.DELTA_H_NET] = (
        pt[ProblemTableLabel.DELTA_T] * pt[ProblemTableLabel.CP_NET]
    )

    # H_net = -Σ ΔH_net
    pt[ProblemTableLabel.H_NET] = -np.cumsum(pt[ProblemTableLabel.DELTA_H_NET])

    # Shift cascades so minimum H_net is zero and the curves align at the pinch.
    pt[ProblemTableLabel.H_HOT] = (
        pt[ProblemTableLabel.H_HOT][-1] - pt[ProblemTableLabel.H_HOT]
    )
    pt[ProblemTableLabel.H_COLD] = (
        pt[ProblemTableLabel.H_COLD][-1]
        + pt[ProblemTableLabel.H_NET][-1]
        - pt[ProblemTableLabel.H_NET].min()
    ) - pt[ProblemTableLabel.H_COLD]
    pt[ProblemTableLabel.H_NET] -= pt[ProblemTableLabel.H_NET].min()

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
    return pt.loc[0, ProblemTableLabel.H_HOT] - pt.loc[-1, ProblemTableLabel.H_NET]


def set_zonal_targets(
    pt: ProblemTable,
    pt_real: ProblemTable,
) -> dict:
    """Assign thermal targets and integration degree from process-table data."""
    return {
        "hot_utility_target": pt.loc[0, ProblemTableLabel.H_NET],
        "cold_utility_target": pt.loc[-1, ProblemTableLabel.H_NET],
        "heat_recovery_target": pt.loc[0, ProblemTableLabel.H_HOT]
        - pt.loc[-1, ProblemTableLabel.H_NET],
        "heat_recovery_limit": pt_real.loc[0, ProblemTableLabel.H_HOT]
        - pt_real.loc[-1, ProblemTableLabel.H_NET],
        "degree_of_int": (
            (pt.loc[0, ProblemTableLabel.H_HOT] - pt.loc[-1, ProblemTableLabel.H_NET])
            / (
                pt_real.loc[0, ProblemTableLabel.H_HOT]
                - pt_real.loc[-1, ProblemTableLabel.H_NET]
            )
            if (
                pt_real.loc[0, ProblemTableLabel.H_HOT]
                - pt_real.loc[-1, ProblemTableLabel.H_NET]
            )
            > 0
            else 1.0
        ),
    }


################################################################################
# Helper functions
################################################################################


def _sum_mcp_between_temperature_boundaries(
    temperatures: List[float],
    streams: StreamCollection,
    is_shifted: bool = True,
    idx: int | None = None,
) -> tuple[list[float], list[float]]:
    """Vectorized CP and rCP summation across temperature intervals."""
    return_lists = not isinstance(streams, StreamCollection)
    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.size == 0:
        return [], []
    if len(streams) == 0 or temperatures.size == 1:
        zeros = np.zeros(temperatures.size, dtype=float)
        return (zeros.tolist(), zeros.tolist()) if return_lists else (zeros, zeros)

    if return_lists:
        streams = StreamCollection(list(streams))

    numeric = streams.segment_numeric_view(idx)
    t_min = numeric.t_min_star if is_shifted else numeric.t_min
    t_max = numeric.t_max_star if is_shifted else numeric.t_max
    active_streams = numeric.active & np.isfinite(t_min) & np.isfinite(t_max)
    if not np.any(active_streams):
        zeros = np.zeros(temperatures.size, dtype=float)
        return (zeros.tolist(), zeros.tolist()) if return_lists else (zeros, zeros)

    t_min = t_min[active_streams]
    t_max = t_max[active_streams]
    cp = numeric.cp[active_streams]
    rcp = numeric.rcp[active_streams]
    lower_bounds = temperatures[1:]
    upper_bounds = temperatures[:-1]

    active = (t_max[np.newaxis, :] > lower_bounds[:, np.newaxis] + tol * 10) & (
        t_min[np.newaxis, :] < upper_bounds[:, np.newaxis] - tol * 10
    )

    cp_array = np.empty(temperatures.size, dtype=float)
    rcp_array = np.empty(temperatures.size, dtype=float)
    cp_array[0] = 0.0
    rcp_array[0] = 0.0
    cp_array[1:] = active @ cp
    rcp_array[1:] = active @ rcp
    if return_lists:
        return cp_array.tolist(), rcp_array.tolist()
    return cp_array, rcp_array


def _shift_pt_to_set_heat_recovery(
    pt: ProblemTable, heat_recovery_target: float, current_heat_recovery: float
) -> ProblemTable:
    """Shift H_COLD and H_NET columns to match targeted heat recovery levels."""
    delta_shift = current_heat_recovery - heat_recovery_target
    pt[ProblemTableLabel.H_COLD] += delta_shift
    pt[ProblemTableLabel.H_NET] += delta_shift
    return pt


def _insert_temperature_interval_into_pt_at_constant_h(
    pt: ProblemTable,
) -> ProblemTable:
    """Insert temperature intervals where HCC and CCC intersect at constant enthalpy."""
    T_new = []
    # --- HCC to CCC projection ---
    if pt[ProblemTableLabel.H_HOT][0] > 0.0:
        T_new.append(
            _get_T_start_on_opposite_cc(
                pt, pt[ProblemTableLabel.H_HOT][0], ProblemTableLabel.H_COLD
            )
        )
    # --- CCC to HCC projection ---
    if pt[ProblemTableLabel.H_COLD][-1] > 0.0:
        T_new.append(
            _get_T_start_on_opposite_cc(
                pt, pt[ProblemTableLabel.H_COLD][-1], ProblemTableLabel.H_HOT
            )
        )
    T_new = [t for t in T_new if t is not None]
    if len(T_new) > 0:
        pt.insert_temperature_interval(T_new)
    return pt


def _get_T_start_on_opposite_cc(
    pt: ProblemTable,
    h0: float,
    col_CC: str | ProblemTableLabel,
    col_T: str | ProblemTableLabel = ProblemTableLabel.T,
) -> float:
    """Find the temperature where composite curves intersect at constant enthalpy."""

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

    return linear_interpolation(
        h0,
        pt[col_CC][idx],
        pt[col_CC][idx + 1],
        pt[col_T][idx],
        pt[col_T][idx + 1],
    )
