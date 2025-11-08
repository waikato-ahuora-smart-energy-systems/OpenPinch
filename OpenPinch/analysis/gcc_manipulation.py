"""Determine various forms of the grand composite curve."""

from typing import Dict

import numpy as np

from ..classes import *
from ..lib import *
from ..utils import *


__all__ = [
    "get_additional_GCCs",
    "get_GCC_without_pockets", 
    "get_GCC_with_partial_pockets", 
    "get_GCC_with_vertical_heat_transfer",
    "get_GCC_needing_utility",
    "get_GGC_pockets",
    "get_seperated_gcc_heat_load_profiles",
]

# TODO: Implement exergy targeting through the exergetic GCC approach. 

#######################################################################################################
# Public API
#######################################################################################################


def get_additional_GCCs(
    pt: ProblemTable,
    is_direct_integration: bool = True,
    do_vert_cc_calc: bool = False,
    do_assisted_ht_calc: bool = False,
) -> ProblemTable:        
    # Calculate various GCC profiles
    if is_direct_integration:
        get_GCC_without_pockets(pt)
        
        if do_vert_cc_calc:
            pt.update(
                get_GCC_with_vertical_heat_transfer(
                    pt.col[PT.H_COLD.value],
                    pt.col[PT.H_HOT.value],
                    pt.col[PT.H_NET.value],
                )
            )

        if do_assisted_ht_calc:
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
            pt.col[PT.H_NET_A.value]
        )
    )
    return pt    


def get_GCC_without_pockets(
    pt: ProblemTable, col_H_NP: str = PT.H_NET_NP.value, col_H: str = PT.H_NET.value
) -> Tuple[ProblemTable, ProblemTable]:
    """Flatten GCC pockets by inserting breakpoints so the profile becomes monotonic."""
    pt.col[col_H_NP] = pt.col[col_H]
    
    hot_pinch_loc, cold_pinch_loc, valid = pt.pinch_idx(col_H)
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


def get_GCC_with_partial_pockets(
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

def get_GCC_with_vertical_heat_transfer(
    h_cold: np.ndarray,
    h_hot: np.ndarray,
    h_net: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return the extreme GCC where heat transfer on the composite curves is vertical."""
    h_cold = np.asarray(h_cold)
    h_hot = np.asarray(h_hot)
    h_net = np.asarray(h_net)

    hcc_max = h_hot[0]
    base = np.where(h_cold > hcc_max, h_cold - hcc_max, 0.0)

    cu_tar = h_net[-1]
    h_net_v = np.where(
        h_hot < cu_tar,
        cu_tar - h_hot,
        base,
    )
    return {PT.H_NET_V.value: h_net_v}


def get_GCC_needing_utility(
    h_net: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return the actual GCC."""
    return {PT.H_NET_A.value: h_net}


def get_GGC_pockets(pt: ProblemTable) -> Dict[str, np.ndarray]:
    """Store GCC pocket contribution (difference between real and pocket-free profiles)."""
    h_net_pk = np.subtract(pt.col[PT.H_NET.value], pt.col[PT.H_NET_NP.value])
    pt.col[PT.H_NET_PK.value] = h_net_pk
    return {
        PT.H_NET_PK.value: h_net_pk
    }


def get_seperated_gcc_heat_load_profiles(
    H_net,
    rcp_net: np.ndarray = None,
    is_process_stream: bool = True,
) -> Dict[str, np.ndarray]:
    """Determines the net required heating or cooling profile of a system from the GCC."""

    # Calculate ΔH differences
    dh_diff = delta_with_zero_at_start(H_net)

    # Determine whether each row corresponds to a hot-side or cold-side enthalpy change
    if is_process_stream:
        is_hot = dh_diff <= 0
        is_cold = ~is_hot
    else:
        is_cold = dh_diff <= 0
        is_hot = ~is_cold

    # Compute cumulative enthalpy change
    hot_profile = np.cumsum(-dh_diff * is_hot)
    cold_profile = np.cumsum(-dh_diff * is_cold)

    # Handle RCP (HTR x CP)
    if not is_process_stream:
        rcp_hot = rcp_net * is_hot
        rcp_cold = rcp_net * is_cold

    # Normalize hot profile to start at x=0 and cold profile to end at x=0
    if is_process_stream:
        hot_profile *= -1
        hut_max = -cold_profile[-1]
        cold_profile = cold_profile + hut_max
    else:
        cold_profile *= -1
        hut_max = -hot_profile[-1]
        hot_profile = hot_profile + hut_max

    return {
        PT.H_NET_HOT.value: hot_profile,
        PT.H_NET_COLD.value: cold_profile,
    } if is_process_stream else {
        PT.H_HOT_UT.value: hot_profile,
        PT.H_COLD_UT.value: cold_profile,
        PT.RCP_HOT_UT.value: rcp_hot,
        PT.RCP_COLD_UT.value: rcp_cold,        
    }


#######################################################################################################
# Helper functions
#######################################################################################################


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
                n_int_added = pt.insert_temperature_interval(T0)

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


# def Calc_GCC_AI(z, pt_real, GCC_N):
#     """Returns a simplified array for the assisted integration GCC.
#     """
#     GCC_AI = [ [ None for j in range(len(pt_real[0]))] for i in range(2)]
#     for i in range(len(pt_real[0])):
#         GCC_AI[0][i] = pt_real[0][i]
#         GCC_AI[1][i] = pt_real[PT.H_NET.value][i] - GCC_N[1][i]
#     return GCC_AI
