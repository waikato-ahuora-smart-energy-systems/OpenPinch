"""Heuristics for estimating the minimum number of heat exchangers for the pinch design method."""

import numpy as np

from ..classes import *
from ..lib.enums import *
from ..utils import *
from ..utils.miscellaneous import *

__all__ = ["get_min_number_hx"]

#######################################################################################################
# Public API --- TODO: check implementation
#######################################################################################################


def get_min_number_hx(z, pt_df: ProblemTable, bcc_star_df: ProblemTable) -> int:
    """
    Estimates the minimum number of heat exchangers required for the pinch problem using vectorized interval logic.

    Args:
        z: Zone with hot/cold streams and utilities.
        pt_df (ProblemTable): Problem table DataFrame with temperature column.
        bcc_star_df (ProblemTable): Balanced Composite Curve data with 'CCC' and 'HCC'.

    Returns:
        int: Minimum number of exchangers.
    """
    T_vals = pt_df.iloc[:, 0].values
    CCC = bcc_star_df["CCC"].values
    HCC = bcc_star_df["HCC"].values

    num_hx = 0
    i = 0
    while i < len(T_vals) - 1:
        if abs(CCC[i + 1] - HCC[i + 1]) > tol:
            break
        i += 1

    i_1 = i
    i += 1

    while i < len(T_vals):
        i_0 = i_1
        if abs(CCC[i] - HCC[i]) < tol or i == len(T_vals) - 1:
            i_1 = i
            T_high, T_low = T_vals[i_0], T_vals[i_1]

            def count_crossing(streams):
                """Count process streams whose adjusted temperatures intersect interval [T_low, T_high]."""
                t_max = np.array([s.t_max_star for s in streams])
                t_min = np.array([s.t_min_star for s in streams])
                return np.sum(
                    ((t_max > T_low + tol) & (t_max <= T_high + tol))
                    | ((t_min >= T_low - tol) & (t_min < T_high - tol))
                    | ((t_min < T_low - tol) & (t_max > T_high + tol))
                )

            num_hx += count_crossing(z.hot_streams)
            num_hx += count_crossing(z.cold_streams)

            def count_utility_crossing(utilities):
                """Count utility streams whose adjusted temperatures intersect interval [T_low, T_high]."""
                t_max = np.array([u.t_max_star for u in utilities])
                t_min = np.array([u.t_min_star for u in utilities])
                return np.sum(
                    (t_max > T_low + tol) & (t_max <= T_high + tol)
                    | (t_min >= T_low - tol) & (t_min < T_high - tol)
                )

            num_hx += count_utility_crossing(z.hot_utilities)
            num_hx += count_utility_crossing(z.cold_utilities)
            num_hx -= 1

            j = i_1
            while j < len(T_vals) - 1:
                if abs(CCC[j + 1] - HCC[j + 1]) > tol:
                    break
                j += 1

            i = j
            i_1 = j

        i += 1

    return int(num_hx)
