"""Target multiple utilities over a heating or cooler profile starting from the pinch temperature."""

import numpy as np

from ..classes import *
from ..lib import *
from ..utils import *
from .problem_table_analysis import get_utility_heat_cascade
from .support_methods import (
    get_pinch_loc,
    insert_temperature_interval_into_pt,
    linear_interpolation,
)

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
        pt = _calc_GCC_without_pockets(pt)
    if config.DO_VERT_GCC:
        pt = _calc_GCC_with_vertical_heat_transfer(pt)
    if config.DO_ASSITED_HT:
        pt = _calc_GGC_pockets(pt)
    pt = _calc_GCC_needing_utility(pt, config.GCC_FOR_TARGETING)
    
    # Add assisted integration targeting here...
    pt = _calc_seperated_heat_load_profiles(pt, col_H_net=PT.H_NET_A.value)

    # Target multiple utility use
    if is_process_zone:
        hot_utilities = _target_utility(
            hot_utilities, pt, PT.T.value, PT.H_COLD_NET.value
        )
        cold_utilities = _target_utility(
            cold_utilities, pt, PT.T.value, PT.H_HOT_NET.value
        )

    pt = get_utility_heat_cascade(
        pt, 
        hot_utilities, 
        cold_utilities, 
        is_shifted=True
    )

    pt = _calc_seperated_heat_load_profiles(
        pt,
        col_H_net=PT.H_UT_NET.value,
        col_H_cold_net=PT.H_COLD_UT.value,
        col_H_hot_net=PT.H_HOT_UT.value,
        is_process_stream=False,
    )

    pt_real = get_utility_heat_cascade(
        pt_real, hot_utilities, cold_utilities, is_shifted=False
    )

    pt_real = _calc_seperated_heat_load_profiles(
        pt_real,
        col_H_net=PT.H_UT_NET.value,
        col_H_cold_net=PT.H_COLD_UT.value,
        col_H_hot_net=PT.H_HOT_UT.value,
        is_process_stream=False,
    )

    pt_real = _calc_balanced_CC(
        pt_real
    )

    return pt, pt_real, hot_utilities, cold_utilities


#######################################################################################################
# Helper functions: get_utility_targets
#######################################################################################################

@timing_decorator
def _target_utility(
    utilities: List[Stream], pt: ProblemTable, col_T: str, col_H: str, real_T=False
) -> List[Stream]:
    """Targets multiple utility use considering a fixed target temperature."""
    if len(utilities) == 0:
        return utilities

    pt = pt.copy
    # pt.round(6)
    hot_pinch_row, cold_pinch_row, _ = get_pinch_loc(pt, col_H)

    if pt.col[col_H].min() < -tol:
        pt.col[col_H] = pt.col[col_H] * -1

    if utilities[0].type == StreamType.Hot.value and abs(pt.loc[0, col_H]) > tol:
        utilities = _assign_utility(
            pt, col_T, col_H, utilities, hot_pinch_row, is_hot_ut=True, real_T=real_T
        )

    elif utilities[0].type == StreamType.Cold.value and abs(pt.loc[-1, col_H]) > tol:
        utilities = _assign_utility(
            pt, col_T, col_H, utilities, cold_pinch_row, is_hot_ut=False, real_T=real_T
        )

    return utilities


def _calc_GCC_without_pockets(
    pt: ProblemTable, col_H_NP: str = PT.H_NET_NP.value, col_H: str = PT.H_NET.value
) -> Tuple[ProblemTable, ProblemTable]:
    """Flatten GCC pockets by inserting breakpoints so the profile becomes monotonic."""
    pt.col[col_H_NP] = pt.col[col_H]

    hot_pinch_loc, cold_pinch_loc, valid = get_pinch_loc(pt)
    if not valid:
        return pt

    # Remove any possible pocket segments between the pinches
    if hot_pinch_loc + 1 < cold_pinch_loc:
        for j in range(hot_pinch_loc + 1, cold_pinch_loc):
            pt.loc[j, col_H_NP] = 0

    # Remove pocket segments above the Pinch
    pt, hot_pinch_loc, cold_pinch_loc = _remove_pockets_on_one_side_of_the_pinch(
        pt, col_H_NP, col_H, hot_pinch_loc, cold_pinch_loc, True
    )

    # Remove pocket segments below the Pinch
    pt, hot_pinch_loc, cold_pinch_loc = _remove_pockets_on_one_side_of_the_pinch(
        pt, col_H_NP, col_H, hot_pinch_loc, cold_pinch_loc, False
    )

    return pt


def _remove_pockets_on_one_side_of_the_pinch(
    pt: ProblemTable,
    col_H_NP: str = PT.H_NET_NP.value,
    col_H: str = PT.H_NET.value,
    hot_pinch_loc: int = None,
    cold_pinch_loc: int = None,
    is_above_pinch: bool = True,
) -> Tuple[ProblemTable, ProblemTable]:
    """Iteratively eliminate pockets above or below the pinch by flattening enthalpy spans."""

    # Settings for removing pocket segments for above/below the pinch
    if is_above_pinch:
        i = 0
        pinch_loc = hot_pinch_loc
        sgn = 1
    else:
        i = len(pt) - 1
        pinch_loc = cold_pinch_loc
        sgn = -1

    T_vals, H_vals, H_NP_vals = pt.col[PT.T.value], pt.col[col_H], pt.col[col_H_NP]

    if H_vals[i] < tol:
        # No heating or cooling required
        return pt, hot_pinch_loc, cold_pinch_loc

    for _ in range(i, pinch_loc, sgn):
        di = sgn
        n_int_added = 0
        if H_vals[i] < H_vals[i + sgn] - tol:
            i_0 = i
            i = _pocket_exit_index(H_vals, i, pinch_loc, sgn)

            if i != pinch_loc:
                T0 = linear_interpolation(
                    H_vals[i_0], H_vals[i], H_vals[i + sgn], T_vals[i], T_vals[i + sgn]
                )
                pt, n_int_added = insert_temperature_interval_into_pt(pt, T0)

            if n_int_added > 0:
                T_vals, H_vals, H_NP_vals = (
                    pt.col[PT.T.value],
                    pt.col[col_H],
                    pt.col[col_H_NP],
                )
                if is_above_pinch:
                    hot_pinch_loc += n_int_added
                    cold_pinch_loc += n_int_added
                else:
                    i_0 += n_int_added

            j_rng = range(i_0 + 1, i + 1) if is_above_pinch else range(i + 1, i_0)
            for j in j_rng:
                H_NP_vals[j] = H_vals[i_0]

            di = n_int_added * sgn

        i += di
        if (pinch_loc - i) * sgn <= 0:
            break

    return pt, hot_pinch_loc, cold_pinch_loc


def _pocket_exit_index(H_vals: np.ndarray, i_0: int, pinch_loc: int, sgn: int) -> int:
    """Return index where a pocket terminates when marching in direction ``sgn``."""
    if sgn > 0:
        for i in range(i_0 + 1, pinch_loc + 1):
            if H_vals[i_0] >= H_vals[i] + tol:
                return i - 1
        return pinch_loc
    else:
        for i in range(i_0 - 1, pinch_loc - 1, -1):
            if H_vals[i_0] >= H_vals[i] + tol:
                return i + 1
        return pinch_loc


def _calc_GCC_with_vertical_heat_transfer(pt: ProblemTable) -> ProblemTable:
    """Returns the extreme GCC where heat transfer on the composite curves is vertical (not horizontal)."""
    # Top section of the vGCC
    hcc_max = pt.loc[0, PT.H_HOT.value]
    pt.col[PT.H_NET_V.value] = np.where(
        pt.col[PT.H_COLD.value] > hcc_max, pt.col[PT.H_COLD.value] - hcc_max, 0.0
    )
    # Bottom section of the vGCC
    cu_tar = pt.loc[-1, PT.H_NET.value]
    pt.col[PT.H_NET_V.value] = np.where(
        pt.col[PT.H_HOT.value] < cu_tar,
        cu_tar - pt.col[PT.H_HOT.value],
        pt.col[PT.H_NET_V.value],
    )
    return pt


def _calc_GCC_needing_utility(pt: ProblemTable, selected_gcc: str = PT.H_NET_NP.value) -> ProblemTable:
    """Return the actual GCC based on utility usage and heat transfer direction settings."""
    pt.col[PT.H_NET_A.value] = pt.col[selected_gcc]
    return pt


def _calc_GGC_pockets(pt: ProblemTable) -> ProblemTable:
    """Store GCC pocket contribution (difference between real and pocket-free profiles)."""
    pt.col[PT.H_NET_PK.value] = pt.col[PT.H_NET.value] - pt.col[PT.H_NET_NP.value]
    return pt


def _calc_seperated_heat_load_profiles(
    pt: ProblemTable,
    col_H_net: str = PT.H_NET_A.value,
    col_H_hot_net: str = PT.H_HOT_NET.value,
    col_H_cold_net: str = PT.H_COLD_NET.value,
    col_RCP_net: str = PT.RCP_UT_NET.value,
    col_RCP_hot_net: str = PT.RCP_HOT_UT.value,
    col_RCP_cold_net: str = PT.RCP_COLD_UT.value,
    is_process_stream: bool = True,
) -> ProblemTable:
    """Determines the gross required heating or cooling profile of a system from the GCC."""

    # Calculate ΔH differences
    dh_diff = pt.delta_col(col_H_net, 1)

    # Determine whether each row corresponds to a hot-side or cold-side enthalpy change
    if is_process_stream:
        is_hot = dh_diff <= 0
        is_cold = ~is_hot
    else:
        is_cold = dh_diff <= 0
        is_hot = ~is_cold

    # Compute cumulative enthalpy change
    pt.col[col_H_hot_net] = np.cumsum(-dh_diff * is_hot)
    pt.col[col_H_cold_net] = np.cumsum(-dh_diff * is_cold)

    # Handle RCP (HTR x CP)
    pt.col[col_RCP_hot_net] = pt.col[col_RCP_net] * is_hot
    pt.col[col_RCP_cold_net] = pt.col[col_RCP_net] * is_cold

    # Normalize hot profile to start at x=0 and cold profile to end at x=0
    if is_process_stream:
        pt.col[col_H_hot_net] *= -1
        HUt_max = -pt.loc[-1, col_H_cold_net]
        pt.col[col_H_cold_net] += HUt_max
    else:
        pt.col[col_H_cold_net] *= -1
        HUt_max = -pt.loc[-1, col_H_hot_net]
        pt.col[col_H_hot_net] += HUt_max      

    return pt


def _calc_balanced_CC(pt: ProblemTable) -> ProblemTable:
    """Creates the balanced Composite Curve (CC) using both process and utility streams."""

    pt.col[PT.H_HOT_BAL.value] = pt.col[PT.H_HOT.value] + pt.col[PT.H_HOT_UT.value]
    pt.col[PT.H_COLD_BAL.value] = pt.col[PT.H_COLD.value] + pt.col[PT.H_COLD_UT.value]

    return pt


#######################################################################################################
# Helper functions: assisted integration
#######################################################################################################


def _calc_GCC_with_partial_pockets(
    pt: ProblemTable, dt_cut: float = 10, dt_cut_min: float = 0
) -> ProblemTable:
    """Modify PT in-place to reflect assisted GCC and return the GCC_AI result."""

    # pt.col[PT.H_NET_PK.value] = pt.col[PT.H_NET.value] - pt.col[PT.H_NET_NP.value]

    # if np.sum(pt.col[PT.H_NET_PK.value]) < tol * len(pt):
    #     pt.col[PT.H_NET_AI.value] = pt.col[PT.H_NET.value]
    #     return pt

    # i = len(pt)
    # while i > 0:
    #     if pt.loc[i - 1, PT.H_NET_PK.value] > tol:
    #         i_lb = i
    #         for i in range(i, 0, -1):
    #             if pt.loc[i, PT.H_NET_PK.value] < tol:
    #                 break
    #         i_ub = i
    #         _compute_pocket_temperature_differences

    #     else:
    #         i += 1

    # pt.col[PT.H_NET_AI.value] = pt.col[PT.H_NET.value] - pt.col[PT.H_NET_PK.value]
    return pt


#######################################################################################################
# Helper functions: _target_utility
#######################################################################################################


def _assign_utility(
    pt: ProblemTable,
    col_T: Enum,
    col_H: Enum,
    u_ls: List[Stream],
    pinch_row: int,
    is_hot_ut: bool,
    real_T: bool,
) -> List[Stream]:
    """Assigns utility heat duties based on vertical heat transfer across a pinch."""
    col_T_values = pt.col[col_T]
    col_H_values = pt.col[col_H]
    if is_hot_ut:
        T_segment = col_T_values[: pinch_row + 1]
        H_segment = col_H_values[: pinch_row + 1]
        segment_limit = H_segment[0]
    else:
        T_segment = col_T_values[pinch_row - 1:]
        H_segment = col_H_values[pinch_row - 1:]
        segment_limit = H_segment[-1]

    Q_assigned = 0.0
    for u in reversed(u_ls) if is_hot_ut else u_ls:
        Ts, Tt = (
            ((u.t_max, u.t_min) if real_T else (u.t_max_star, u.t_min_star))
            if is_hot_ut
            else ((u.t_min, u.t_max) if real_T else (u.t_min_star, u.t_max_star))
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
