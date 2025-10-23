"""Target multiple utilities over a heating or cooler profile starting from the pinch temperature."""

import numpy as np

from ..classes import *
from ..lib import *
from ..utils import *
from .problem_table_analysis import get_utility_heat_cascade
from .gcc_manipulation import *
from .area_targeting import get_balanced_CC


__all__ = ["get_utility_targets"]

#######################################################################################################
# Public API
#######################################################################################################


def get_utility_targets(
    pt: ProblemTable,
    pt_real: ProblemTable,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    is_process_zone: bool = True,
    config: Configuration = Configuration(),
) -> Tuple[ProblemTable, ProblemTable, StreamCollection, StreamCollection]:
    """Target utility usage and compute GCC variants for a zone.

    Parameters
    ----------
    pt, pt_real:
        Shifted and real problem tables used for constructing composite curves.
    hot_utilities, cold_utilities:
        Candidate utility collections that will be targeted across temperature
        intervals.
    is_process_zone:
        When ``True`` (default) the function assumes the zone represents a
        process area and applies additional targeting logic appropriate for that
        context.

    Returns
    -------
    tuple
        Updated ``(pt, pt_real, hot_utilities, cold_utilities)`` collections with
        derived profiles embedded.
    """
    
    # Calculate various GCC profiles
    if is_process_zone:
        get_GCC_without_pockets(pt)
        
    if config.DO_VERT_GCC:
        pt.update(
            get_GCC_with_vertical_heat_transfer(
                pt.col[PT.H_COLD.value],
                pt.col[PT.H_HOT.value],
                pt.col[PT.H_NET.value],
            )
        )

    if config.DO_ASSITED_HT:
        pt.update(
            get_GGC_pockets(pt)
        )

    pt.update(
        get_GCC_needing_utility(
            pt.col[PT.H_NET_NP.value]
        )
    )    
    pt.update(
        get_seperated_gcc_heat_load_profiles(
            pt, 
            col_H_net=PT.H_NET_A.value
        )
    )

    # Target multiple utility use
    if is_process_zone:
        hot_pinch_row, cold_pinch_row, _ = pt.pinch_idx(PT.H_NET_A)
        is_real_temperatures = False
        hot_utilities = _target_utility(
            hot_utilities, 
            pt.col[PT.T.value], 
            pt.col[PT.H_COLD_NET.value],
            hot_pinch_row, 
            cold_pinch_row,
            is_real_temperatures,
        )
        cold_utilities = _target_utility(
            cold_utilities, 
            pt.col[PT.T.value], 
            pt.col[PT.H_HOT_NET.value],
            hot_pinch_row, 
            cold_pinch_row,
            is_real_temperatures,
        )

    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value],
            hot_utilities,
            cold_utilities,
            is_shifted=True,
        )
    )
    pt.update(
        get_seperated_gcc_heat_load_profiles(
            pt,
            col_H_net=PT.H_UT_NET.value,
            col_H_cold_net=PT.H_COLD_UT.value,
            col_H_hot_net=PT.H_HOT_UT.value,
            is_process_stream=False,
        )
    )
    pt_real.update(
        get_utility_heat_cascade(
            pt_real.col[PT.T.value], 
            hot_utilities, 
            cold_utilities, 
            is_shifted=False
        )
    )
    pt_real.update(
        get_seperated_gcc_heat_load_profiles(
            pt_real,
            col_H_net=PT.H_UT_NET.value,
            col_H_cold_net=PT.H_COLD_UT.value,
            col_H_hot_net=PT.H_HOT_UT.value,
            is_process_stream=False,
        )
    )
    
    pt_real = get_balanced_CC(
        pt_real
    )
    return pt, pt_real, hot_utilities, cold_utilities


#######################################################################################################
# Helper functions
#######################################################################################################


def _target_utility(
    utilities: List[Stream], 
    T_vals: np.ndarray, 
    H_vals: np.ndarray, 
    hot_pinch_row: int, 
    cold_pinch_row: int, 
    is_real_temperatures: bool = False,
) -> List[Stream]:
    """Targets multiple utility use considering a fixed target temperature."""
    if len(utilities) == 0:
        return utilities

    if H_vals.min() < -tol:
        H_vals = H_vals * -1

    if utilities[0].type == StreamType.Hot.value and abs(H_vals[0]) > tol:
        utilities = _assign_utility(
            T_vals, H_vals, utilities, hot_pinch_row, is_hot_ut=True, is_real_temperatures=is_real_temperatures
        )

    elif utilities[0].type == StreamType.Cold.value and abs(H_vals[-1]) > tol:
        utilities = _assign_utility(
            T_vals, H_vals, utilities, cold_pinch_row, is_hot_ut=False, is_real_temperatures=is_real_temperatures
        )

    return utilities


def _assign_utility(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    u_ls: List[Stream],
    pinch_row: int,
    is_hot_ut: bool,
    is_real_temperatures: bool,
) -> List[Stream]:
    """Assigns utility heat duties based on vertical heat transfer across a pinch."""
    if is_hot_ut:
        T_segment = T_vals[: pinch_row + 1]
        H_segment = H_vals[: pinch_row + 1]
        segment_limit = H_segment[0]
    else:
        T_segment = T_vals[pinch_row - 1:]
        H_segment = H_vals[pinch_row - 1:]
        segment_limit = H_segment[-1]

    Q_assigned = 0.0
    for u in reversed(u_ls) if is_hot_ut else u_ls:
        Ts, Tt = (
            ((u.t_max, u.t_min) if is_real_temperatures else (u.t_max_star, u.t_min_star))
            if is_hot_ut
            else ((u.t_min, u.t_max) if is_real_temperatures else (u.t_min_star, u.t_max_star))
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
            u.set_heat_flow(Q_ut_max)
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
    """Determine remaining heat duty a utility can serve given temperature limits and prior assignments."""
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

    valid_mask = (
        (adjacent_H != current_H)
        & (dt_sup >= -tol)
        & (Q_pot > tol)
    )
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
            Q_pot_valid[slope_mask]
            / (-dt_tar_valid[slope_mask])
            * abs(Tt - Ts)
        )
    Q_tt_max = Q_tt.min() if Q_tt.size > 0 else np.inf

    return min(Q_ts_max, Q_tt_max) if dt_tar_valid.max() >= 0 else 0.0
