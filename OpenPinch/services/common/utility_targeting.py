"""Target multiple utilities over a heating or cooling profile from the pinch."""

from typing import Tuple

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.stream_collection import StreamCollection
from ...lib.config import tol
from ...lib.enums import PT
from .gcc_manipulation import get_seperated_gcc_heat_load_profiles
from .problem_table_analysis import get_utility_heat_cascade

__all__ = ["get_utility_targets"]

################################################################################
# Public API
################################################################################


def get_utility_targets(
    pt: ProblemTable,
    pt_real: ProblemTable = None,
    hot_utilities: StreamCollection = None,
    cold_utilities: StreamCollection = None,
    is_direct_integration: bool = True,
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
        hot_utilities, cold_utilities = _target_utility(
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
            T_vals=pt[PT.T],
            H_net_cold=pt[PT.H_NET_COLD],
            H_net_hot=pt[PT.H_NET_HOT],
            pinch_idx=pt.pinch_idx(PT.H_NET_A),
            is_real_temperatures=False,
        )

    pt.update(
        **get_utility_heat_cascade(
            T_int_vals=pt[PT.T],
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
            is_shifted=True,
        )
    )
    pt.update(
        **get_seperated_gcc_heat_load_profiles(
            T_col=pt[PT.T],
            H_net=pt[PT.H_NET_UT],
            rcp_net=pt[PT.RCP_UT_NET],
            is_process_stream=False,
        )
    )
    if isinstance(pt_real, ProblemTable):
        pt_real.update(
            **get_utility_heat_cascade(
                T_int_vals=pt_real[PT.T],
                hot_utilities=hot_utilities,
                cold_utilities=cold_utilities,
                is_shifted=False,
            )
        )
        pt_real.update(
            **get_seperated_gcc_heat_load_profiles(
                T_col=pt_real[PT.T],
                H_net=pt_real[PT.H_NET_UT],
                rcp_net=pt_real[PT.RCP_UT_NET],
                is_process_stream=False,
            )
        )
    return pt, pt_real, hot_utilities, cold_utilities


################################################################################
# Helper functions
################################################################################


def _target_utility(
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    T_vals: np.ndarray,
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    pinch_idx: Tuple[int, int],
    is_real_temperatures: bool = False,
) -> Tuple[StreamCollection, StreamCollection]:
    """Targets multiple utility use considering a fixed target temperature."""
    if abs(H_net_cold[0]) > tol:
        if len(hot_utilities) == 0:
            raise ValueError(
                "Hot utility targeting failed. No hot utilities provided but "
                "heat load profile indicates utility use is required."
            )
        hot_utilities = _assign_utility(
            T_vals=T_vals,
            H_vals=np.abs(H_net_cold),
            u_ls=hot_utilities,
            pinch_row=pinch_idx[0],
            is_hot_ut=True,
            is_real_temperatures=is_real_temperatures,
        )
    if abs(H_net_hot[-1]) > tol:
        if len(cold_utilities) == 0:
            raise ValueError(
                "Cold utility targeting failed. No cold utilities provided but "
                "heat load profile indicates utility use is required."
            )
        cold_utilities = _assign_utility(
            T_vals=T_vals,
            H_vals=np.abs(H_net_hot),
            u_ls=cold_utilities,
            pinch_row=pinch_idx[1],
            is_hot_ut=False,
            is_real_temperatures=is_real_temperatures,
        )
    return hot_utilities, cold_utilities


def _assign_utility(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    u_ls: StreamCollection,
    pinch_row: int,
    is_hot_ut: bool,
    is_real_temperatures: bool,
) -> StreamCollection:
    """Assigns utility heat duties based on vertical heat transfer across a pinch."""
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

    Q_assigned = 0.0
    for u in reversed(u_ls) if is_hot_ut else u_ls:
        Ts, Tt = (
            (
                (u.t_max, u.t_min)
                if is_real_temperatures
                else (u.t_max_star, u.t_min_star)
            )
            if is_hot_ut
            else (
                (u.t_min, u.t_max)
                if is_real_temperatures
                else (u.t_min_star, u.t_max_star)
            )
        )

        Q_ut_max = _maximise_utility_duty(
            T_segment,
            H_segment,
            Ts,
            Tt,
            is_hot_ut,
            Q_assigned,
        )
        if Q_ut_max > tol:
            u.heat_flow = Q_ut_max
            Q_assigned += Q_ut_max

        if abs(segment_limit - Q_assigned) < tol:
            break

    return u_ls


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
    Q_pot_valid = Q_pot[valid_mask]

    if dt_tar_valid.max() < 0:
        return 0.0

    Q_ts_max = Q_pot_valid.max()

    Q_tt = np.full_like(Q_pot_valid, np.inf, dtype=float)
    slope_mask = (-dt_tar_valid) > tol
    if np.any(slope_mask):
        Q_tt[slope_mask] = (
            Q_pot_valid[slope_mask] / (-dt_tar_valid[slope_mask]) * abs(Tt - Ts)
        )
    Q_tt_max = Q_tt.min() if Q_tt.size > 0 else np.inf

    return min(Q_ts_max, Q_tt_max) if dt_tar_valid.max() >= 0 else 0.0
