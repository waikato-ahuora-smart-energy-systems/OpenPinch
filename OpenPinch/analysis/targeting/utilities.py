"""Target multiple utilities over a heating or cooling profile from the pinch."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...domain.configuration import tol
from ...domain.enums import ProblemTableLabel
from ...domain.problem_table import ProblemTable
from ...domain.stream_collection import StreamCollection
from .cascade import get_utility_heat_cascade
from .grand_composite import get_seperated_gcc_heat_load_profiles

__all__ = ["target_utilities_for_load_profiles", "get_utility_targets"]

################################################################################
# Public API
################################################################################


def get_utility_targets(
    pt: ProblemTable,
    pt_real: ProblemTable = None,
    hot_utilities: StreamCollection = None,
    cold_utilities: StreamCollection = None,
    is_direct_integration: bool = True,
    idx: int | None = None,
) -> Tuple[ProblemTable, ProblemTable, StreamCollection, StreamCollection]:
    """Target utility usage and compute GCC variants for a zone.

    Parameters
    ----------
    pt, pt_real:
        Shifted and real problem tables used for constructing composite curves.
    hot_utilities, cold_utilities:
        Candidate utility collections that will be targeted across temperature
        intervals.
    is_direct_integration:
        When ``True`` (default) the function assumes the zone represents a
        process area and applies additional targeting logic appropriate for that
        context.

    Returns
    -------
    tuple
        Updated ``(pt, pt_real, hot_utilities, cold_utilities)`` collections with
        derived profiles embedded.
    """

    # Target multiple utility use
    if is_direct_integration:
        hot_utilities, cold_utilities = target_utilities_for_load_profiles(
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
            T_vals=pt[ProblemTableLabel.T],
            H_net_cold=pt[ProblemTableLabel.H_NET_COLD],
            H_net_hot=pt[ProblemTableLabel.H_NET_HOT],
            pinch_idx=pt.pinch_idx(ProblemTableLabel.H_NET_A),
            is_real_temperatures=False,
            idx=idx,
        )

    pt.update(
        **get_utility_heat_cascade(
            T_int_vals=pt[ProblemTableLabel.T],
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
            is_shifted=True,
            period_idx=idx,
        )
    )
    pt.update(
        **get_seperated_gcc_heat_load_profiles(
            T_col=pt[ProblemTableLabel.T],
            H_net=pt[ProblemTableLabel.H_NET_UT],
            rcp_net=pt[ProblemTableLabel.RCP_UT_NET],
            is_process_stream=False,
        )
    )
    if isinstance(pt_real, ProblemTable):
        pt_real.update(
            **get_utility_heat_cascade(
                T_int_vals=pt_real[ProblemTableLabel.T],
                hot_utilities=hot_utilities,
                cold_utilities=cold_utilities,
                is_shifted=False,
                period_idx=idx,
            )
        )
        pt_real.update(
            **get_seperated_gcc_heat_load_profiles(
                T_col=pt_real[ProblemTableLabel.T],
                H_net=pt_real[ProblemTableLabel.H_NET_UT],
                rcp_net=pt_real[ProblemTableLabel.RCP_UT_NET],
                is_process_stream=False,
            )
        )
    return pt, pt_real, hot_utilities, cold_utilities


################################################################################
# Helper functions
################################################################################


def target_utilities_for_load_profiles(
    *,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    T_vals: np.ndarray,
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    pinch_idx: Tuple[int, int],
    is_real_temperatures: bool = False,
    idx: int | None = None,
) -> Tuple[StreamCollection, StreamCollection]:
    """Targets multiple utilities for precomputed hot- and cold-side load profiles."""
    hot_duties, cold_duties = _calculate_utility_duties_for_load_profiles(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        T_vals=T_vals,
        H_net_cold=H_net_cold,
        H_net_hot=H_net_hot,
        pinch_idx=pinch_idx,
        is_real_temperatures=is_real_temperatures,
        idx=idx,
    )
    hot_utilities = _apply_utility_duties(hot_utilities, hot_duties, idx=idx)
    cold_utilities = _apply_utility_duties(cold_utilities, cold_duties, idx=idx)
    return hot_utilities, cold_utilities


def _calculate_utility_duties_for_load_profiles(
    *,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    T_vals: np.ndarray,
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    pinch_idx: Tuple[int, int],
    is_real_temperatures: bool = False,
    idx: int | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calculate duties without mutating reusable candidate utility streams."""
    hot_duties = (0.0,) * len(hot_utilities)
    cold_duties = (0.0,) * len(cold_utilities)
    if abs(H_net_cold[0]) > tol:
        if len(hot_utilities) == 0:
            raise ValueError(
                "Hot utility targeting failed. No hot utilities provided but "
                "heat load profile indicates utility use is required."
            )
        hot_duties = _calculate_assigned_utility_duties(
            T_vals=T_vals,
            H_vals=np.abs(H_net_cold),
            u_ls=hot_utilities,
            pinch_row=pinch_idx[0],
            is_hot_ut=True,
            is_real_temperatures=is_real_temperatures,
            idx=idx,
        )
    if abs(H_net_hot[-1]) > tol:
        if len(cold_utilities) == 0:
            raise ValueError(
                "Cold utility targeting failed. No cold utilities provided but "
                "heat load profile indicates utility use is required."
            )
        cold_duties = _calculate_assigned_utility_duties(
            T_vals=T_vals,
            H_vals=np.abs(H_net_hot),
            u_ls=cold_utilities,
            pinch_row=pinch_idx[1],
            is_hot_ut=False,
            is_real_temperatures=is_real_temperatures,
            idx=idx,
        )
    return hot_duties, cold_duties


def _assign_utility(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    u_ls: StreamCollection,
    pinch_row: int,
    is_hot_ut: bool,
    is_real_temperatures: bool,
    idx: int | None,
) -> StreamCollection:
    """Assigns utility heat duties based on vertical heat transfer across a pinch."""
    duties = _calculate_assigned_utility_duties(
        T_vals=T_vals,
        H_vals=H_vals,
        u_ls=u_ls,
        pinch_row=pinch_row,
        is_hot_ut=is_hot_ut,
        is_real_temperatures=is_real_temperatures,
        idx=idx,
    )
    return _apply_utility_duties(u_ls, duties, idx=idx)


def _apply_utility_duties(
    utilities: StreamCollection,
    duties: tuple[float, ...],
    *,
    idx: int | None,
) -> StreamCollection:
    """Replace the selected-period duty of every utility."""
    for utility, duty in zip(utilities, duties, strict=True):
        utility.set_value_attr_at_idx(
            attr_name="heat_flow",
            value=float(duty) if duty > tol else 0.0,
            idx=idx,
        )
    return utilities


def _calculate_assigned_utility_duties(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    u_ls: StreamCollection,
    pinch_row: int,
    is_hot_ut: bool,
    is_real_temperatures: bool,
    idx: int | None,
) -> tuple[float, ...]:
    """Return ordered utility duties using the canonical targeting algorithm."""
    if is_hot_ut:
        T_segment = T_vals[: pinch_row + 1]
        H_segment = H_vals[: pinch_row + 1]
        segment_limit = H_segment[0]
    else:
        T_segment = T_vals[pinch_row:]
        H_segment = H_vals[pinch_row:]
        segment_limit = H_segment[-1]

    if len(np.where(H_segment < tol)) != 1:
        raise ValueError(
            "Error in utility targeting. Please report the data that produced "
            "this error."
        )

    utilities = tuple(u_ls)
    duties = [0.0] * len(utilities)
    indices = range(len(utilities) - 1, -1, -1) if is_hot_ut else range(len(utilities))
    Q_assigned = 0.0
    for utility_index in indices:
        u = utilities[utility_index]
        if is_real_temperatures:
            t_lo, t_hi = u.minimum_temperature, u.maximum_temperature
        else:
            t_lo, t_hi = u.shifted_minimum_temperature, u.shifted_maximum_temperature
        if is_hot_ut:
            Ts, Tt = float(t_hi[idx]), float(t_lo[idx])
        else:
            Ts, Tt = float(t_lo[idx]), float(t_hi[idx])

        Q_ut_max = _maximise_utility_duty(
            T_segment,
            H_segment,
            Ts,
            Tt,
            is_hot_ut,
            Q_assigned,
        )
        if u.maximum_heat_flow is not None:
            maximum_duty = StreamCollection._value_at_idx(u._maximum_heat_flow, idx)
            if np.isfinite(maximum_duty):
                Q_ut_max = min(Q_ut_max, maximum_duty)
        if Q_ut_max > tol:
            duties[utility_index] = float(Q_ut_max)
            Q_assigned += Q_ut_max

        if abs(segment_limit - Q_assigned) < tol:
            break

    return tuple(duties)


def _maximise_utility_duty(
    T_segment: np.ndarray,
    H_segment: np.ndarray,
    Ts: float,
    Tt: float,
    is_hot_ut: bool,
    Q_assigned: float,
) -> float:
    """Determine remaining heat duty within temperature and assignment limits."""
    if T_segment.size < 2:
        return 0.0

    if is_hot_ut:
        current_T = T_segment[1:]
        previous_T = T_segment[:-1]
        current_H = H_segment[1:]
        adjacent_H = H_segment[:-1]
        Q_pot = adjacent_H - Q_assigned
        dt_tar = Tt - current_T
        dt_sup = Ts - previous_T
    else:
        current_T = T_segment[:-1]
        next_T = T_segment[1:]
        current_H = H_segment[:-1]
        adjacent_H = H_segment[1:]
        Q_pot = adjacent_H - Q_assigned
        dt_tar = current_T - Tt
        dt_sup = next_T - Ts

    valid_mask = (adjacent_H != current_H) & (dt_sup >= -tol) & (Q_pot > tol)
    if not np.any(valid_mask):
        return 0.0

    dt_tar_valid = dt_tar[valid_mask]
    if dt_tar_valid.max() < 0:
        return 0.0

    def _candidate_limit(q_pot_values: np.ndarray) -> tuple[float, float, float]:
        q_pot_valid = q_pot_values[valid_mask]
        q_ts_max = q_pot_valid.max()
        q_tt = np.full_like(q_pot_valid, np.inf, dtype=float)
        slope_mask = (-dt_tar_valid) > tol
        if np.any(slope_mask):
            q_tt[slope_mask] = (
                q_pot_valid[slope_mask] / (-dt_tar_valid[slope_mask]) * abs(Tt - Ts)
            )
        q_tt_max = q_tt.min() if q_tt.size > 0 else np.inf
        return min(q_ts_max, q_tt_max), q_ts_max, q_tt_max

    q_adj, _, _ = _candidate_limit(Q_pot)
    _, _, q_tt_cur = _candidate_limit(current_H - Q_assigned)
    # When the utility target lies inside the GCC range, both ends of every
    # piecewise-linear interval constrain its profile. The former condition
    # inspected the adjacent-end limit and could discard a tighter current-end
    # limit, allowing the sensible utility profile to cross the GCC.
    if np.isfinite(q_tt_cur):
        return min(q_adj, q_tt_cur)
    return q_adj
