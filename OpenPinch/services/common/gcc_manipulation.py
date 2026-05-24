"""Determine various forms of the grand composite curve."""

from typing import Tuple

import numpy as np

from ...classes.problem_table import ProblemTable
from ...lib.config import tol
from ...lib.enums import PT
from ...lib.problem_table_types import ProblemTableUpdateKwargs
from ...utils.miscellaneous import delta_with_zero_at_start, linear_interpolation

__all__ = [
    "get_additional_GCCs",
    "get_GCC_without_pockets",
    "get_GCC_with_partial_pockets",
    "get_GCC_with_vertical_heat_transfer",
    "get_GCC_needing_utility",
    "get_GGC_pockets",
    "get_seperated_gcc_heat_load_profiles",
]


################################################################################
# Public API
################################################################################


def get_additional_GCCs(
    pt: ProblemTable,
    do_vert_cc_calc: bool = False,
    do_assisted_ht_calc: bool = False,
    assisted_ht_dt_cut: float = 10.0,
    assisted_ht_dt_cut_min: float = 0.0,
    is_process_stream: bool = True,
) -> ProblemTable:
    """Populate derived GCC variants used by utility and integration targeting.

    Parameters
    ----------
    pt:
        Problem table containing baseline ``H_net`` and Composite Curve columns.
    do_vert_cc_calc:
        Enable vertical heat-transfer transformation for assisted integration.
    do_assisted_ht_calc:
        Enable pocket extraction and assisted heat-transfer GCC adjustments.

    Returns
    -------
    ProblemTable
        Updated problem table with additional GCC-related columns.
    """
    # Calculate various GCC profiles
    get_GCC_without_pockets(pt)

    actual_h_net = pt[PT.H_NET_NP]

    if do_vert_cc_calc:
        pt.update(
            **get_GCC_with_vertical_heat_transfer(
                T_col=pt[PT.T],
                h_cold=pt[PT.H_COLD],
                h_hot=pt[PT.H_HOT],
                h_net=pt[PT.H_NET],
            )
        )
        actual_h_net = pt[PT.H_NET_V]

    if do_assisted_ht_calc:
        pt.update(
            **get_GCC_with_partial_pockets(
                T_col=pt[PT.T],
                h_net=pt[PT.H_NET],
                h_net_np=pt[PT.H_NET_NP],
                dt_cut=assisted_ht_dt_cut,
                dt_cut_min=assisted_ht_dt_cut_min,
            )
        )
        actual_h_net = pt[PT.H_NET_AI]
    else:
        pt.update(
            **get_GGC_pockets(
                T_col=pt[PT.T],
                h_net=pt[PT.H_NET],
                h_net_np=pt[PT.H_NET_NP],
            )
        )
        pt.update(
            T_col=pt[PT.T],
            updates={PT.H_NET_AI: np.asarray(pt[PT.H_NET_NP], dtype=float)},
        )

    pt.update(
        **get_GCC_needing_utility(
            T_col=pt[PT.T],
            h_net=actual_h_net,
        )
    )
    pt.update(
        **get_seperated_gcc_heat_load_profiles(
            T_col=pt[PT.T],
            H_net=pt[PT.H_NET_A],
            is_process_stream=is_process_stream,
        )
    )
    return pt


def get_GCC_without_pockets(
    pt: ProblemTable,
    col_H_NP: str | PT = PT.H_NET_NP,
    col_H: str | PT = PT.H_NET,
) -> Tuple[ProblemTable, ProblemTable]:
    """Flatten GCC pockets by inserting breakpoints so the profile becomes monotonic."""
    pt[col_H_NP] = pt[col_H]

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
    T_col: np.ndarray,
    h_net: np.ndarray,
    h_net_np: np.ndarray,
    dt_cut: float = 10,
    dt_cut_min: float = 0,
) -> ProblemTableUpdateKwargs:
    """Return GCC updates for the assisted profile after partially cutting pockets."""
    dt_cut = max(float(dt_cut), float(dt_cut_min))

    T_col = np.asarray(T_col, dtype=float)
    h_net = np.asarray(h_net, dtype=float)
    h_net_np = np.asarray(h_net_np, dtype=float)
    h_net_pk = _normalise_pocket_profile(np.subtract(h_net, h_net_np))
    pocket_pt = ProblemTable(
        {
            PT.T: T_col,
            PT.H_NET: h_net,
            PT.H_NET_NP: h_net_np,
            PT.H_NET_PK: h_net_pk,
            PT.H_NET_AI: h_net - h_net_pk,
        }
    )
    if np.nansum(pocket_pt[PT.H_NET_PK]) <= tol * len(pocket_pt):
        return _assisted_gcc_updates(pocket_pt)

    scan_idx = 0
    while True:
        h_pockets = pocket_pt[PT.H_NET_PK]
        run_bounds = _next_positive_run(h_pockets, start_idx=scan_idx)
        if run_bounds is None:
            break

        run_start, run_stop = run_bounds
        if run_start == 0 or run_stop >= len(pocket_pt):
            scan_idx = run_stop
            continue

        t_vals = pocket_pt[PT.T]
        i_upper = run_start - 1
        i_lower = run_stop
        if t_vals[i_upper] - t_vals[i_lower] < dt_cut - tol:
            scan_idx = run_stop
            continue

        if i_upper + 2 < i_lower:
            i_upper, i_lower = _shrink_pocket_to_dt_cut_zone(
                t_vals=t_vals,
                h_pockets=h_pockets,
                i_upper=i_upper,
                i_lower=i_lower,
                dt_cut=dt_cut,
            )

        h_cut = _solve_assisted_pocket_cut_height(
            t_vals=t_vals,
            h_pockets=h_pockets,
            i_upper=i_upper,
            i_lower=i_lower,
            dt_cut=dt_cut,
        )
        if h_cut is None or h_cut <= tol:
            scan_idx = run_stop
            continue

        cut_temps = _cut_temperatures_needing_insertion(
            t_vals=t_vals,
            h_pockets=h_pockets,
            i_upper=i_upper,
            i_lower=i_lower,
            h_cut=h_cut,
            dt_cut=dt_cut,
        )
        if cut_temps.size > 0:
            pocket_pt.insert_temperature_interval(cut_temps)
            h_pockets = pocket_pt[PT.H_NET_PK]
            run_bounds = _next_positive_run(h_pockets, start_idx=max(i_upper, 0))
            if run_bounds is None:
                break
            run_start, run_stop = run_bounds

        _apply_h_cut_to_positive_run(
            h_pockets=pocket_pt[PT.H_NET_PK],
            run_start=run_start,
            run_stop=run_stop,
            h_cut=h_cut,
        )
        scan_idx = run_stop

    pocket_pt[PT.H_NET_PK] = _normalise_pocket_profile(pocket_pt[PT.H_NET_PK])
    pocket_pt[PT.H_NET_AI] = pocket_pt[PT.H_NET] - pocket_pt[PT.H_NET_PK]
    return _assisted_gcc_updates(pocket_pt)


def get_GCC_with_vertical_heat_transfer(
    T_col: np.ndarray,
    h_cold: np.ndarray,
    h_hot: np.ndarray,
    h_net: np.ndarray,
) -> ProblemTableUpdateKwargs:
    """Return the extreme GCC with vertical heat transfer on the composite curves."""
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
    return {"T_col": T_col, "updates": {PT.H_NET_V: h_net_v}}


def get_GCC_needing_utility(
    T_col: np.ndarray,
    h_net: np.ndarray,
) -> ProblemTableUpdateKwargs:
    """Return the actual GCC."""
    return {"T_col": T_col, "updates": {PT.H_NET_A: h_net}}


def get_GGC_pockets(
    T_col: np.ndarray,
    h_net: np.ndarray,
    h_net_np: np.ndarray,
) -> ProblemTableUpdateKwargs:
    """Store GCC pocket contribution.

    This is the difference between the real and pocket-free GCC profiles.
    """
    h_net_pk = _normalise_pocket_profile(np.subtract(h_net, h_net_np))
    return {
        "T_col": np.asarray(T_col, dtype=float),
        "updates": {PT.H_NET_PK: h_net_pk},
    }


def get_seperated_gcc_heat_load_profiles(
    T_col: np.ndarray,
    H_net: np.ndarray,
    rcp_net: np.ndarray = None,
    is_process_stream: bool = True,
) -> ProblemTableUpdateKwargs:
    """Determine the net required heating or cooling profile from the GCC."""
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

    updates = (
        {
            PT.H_NET_HOT: hot_profile,
            PT.H_NET_COLD: cold_profile,
        }
        if is_process_stream
        else {
            PT.H_HOT_UT: hot_profile,
            PT.H_COLD_UT: cold_profile,
            PT.RCP_HOT_UT: rcp_hot,
            PT.RCP_COLD_UT: rcp_cold,
        }
    )
    return {"T_col": T_col, "updates": updates}


################################################################################
# Helper functions
################################################################################


def _remove_pockets_on_one_side_of_the_pinch(
    pt: ProblemTable,
    col_H_NP: str | PT = PT.H_NET_NP,
    col_H: str | PT = PT.H_NET,
    hot_pinch_loc: int = None,
    cold_pinch_loc: int = None,
    is_above_pinch: bool = True,
) -> Tuple[ProblemTable, ProblemTable]:
    """Iteratively eliminate pockets above or below the pinch."""

    # Settings for removing pocket segments for above/below the pinch
    if is_above_pinch:
        i = 0
        pinch_loc = hot_pinch_loc
        sgn = 1
    else:
        i = len(pt) - 1
        pinch_loc = cold_pinch_loc
        sgn = -1

    T_vals, H_vals, H_NP_vals = pt[PT.T], pt[col_H], pt[col_H_NP]

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
                    pt[PT.T],
                    pt[col_H],
                    pt[col_H_NP],
                )
                if is_above_pinch:
                    hot_pinch_loc += n_int_added
                    cold_pinch_loc += n_int_added
                    pinch_loc += n_int_added
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


def _assisted_gcc_updates(pocket_pt: ProblemTable) -> ProblemTableUpdateKwargs:
    """Return aligned assisted GCC updates from a local pocket ProblemTable."""
    return {
        "T_col": pocket_pt[PT.T],
        "updates": {
            PT.H_NET_PK: pocket_pt[PT.H_NET_PK],
            PT.H_NET_AI: pocket_pt[PT.H_NET_AI],
        },
    }


def _normalise_pocket_profile(h_pockets: np.ndarray) -> np.ndarray:
    """Clamp a pocket profile to non-negative values and zero near-tolerance noise."""
    h_pockets = np.clip(np.asarray(h_pockets, dtype=float), 0.0, None)
    h_pockets[np.abs(h_pockets) <= tol] = 0.0
    return h_pockets


def _next_positive_run(
    h_pockets: np.ndarray,
    start_idx: int = 0,
) -> tuple[int, int] | None:
    """Return ``(start, stop)`` for the next contiguous positive pocket run."""
    positive_idx = np.flatnonzero(h_pockets[start_idx:] > tol)
    if positive_idx.size == 0:
        return None

    run_start = start_idx + int(positive_idx[0])
    non_positive_idx = np.flatnonzero(h_pockets[run_start:] <= tol)
    run_stop = (
        run_start + int(non_positive_idx[0])
        if non_positive_idx.size > 0
        else h_pockets.size
    )
    return run_start, run_stop


def _shrink_pocket_to_dt_cut_zone(
    t_vals: np.ndarray,
    h_pockets: np.ndarray,
    i_upper: int,
    i_lower: int,
    dt_cut: float,
) -> tuple[int, int]:
    """Restrict a pocket to the top and bottom segments that bracket the cut region."""
    k = i_lower
    j_top = i_upper + 1
    for j_top in range(i_upper + 1, i_lower):
        while k > j_top and h_pockets[j_top] - h_pockets[k] > tol:
            k -= 1
        if abs(t_vals[j_top] - t_vals[k]) <= dt_cut + tol or j_top + 1 >= k:
            break
    i_upper = j_top - 1

    k = i_upper
    j_bottom = i_lower - 1
    for j_bottom in range(i_lower - 1, i_upper, -1):
        while k < j_bottom and h_pockets[j_bottom] - h_pockets[k] > tol:
            k += 1
        if abs(t_vals[j_bottom] - t_vals[k]) <= dt_cut + tol or j_bottom <= k:
            break
    i_lower = j_bottom + 1
    return i_upper, i_lower


def _solve_assisted_pocket_cut_height(
    t_vals: np.ndarray,
    h_pockets: np.ndarray,
    i_upper: int,
    i_lower: int,
    dt_cut: float,
) -> float | None:
    """Return the cut height solving ``T_hot(H) - T_cold(H) = dt_cut``."""
    dh_top = h_pockets[i_upper] - h_pockets[i_upper + 1]
    dh_bottom = h_pockets[i_lower - 1] - h_pockets[i_lower]
    if abs(dh_top) <= tol or abs(dh_bottom) <= tol:
        return None

    m1 = (t_vals[i_upper] - t_vals[i_upper + 1]) / dh_top
    c1 = t_vals[i_upper] - m1 * h_pockets[i_upper]
    m2 = (t_vals[i_lower - 1] - t_vals[i_lower]) / dh_bottom
    c2 = t_vals[i_lower] - m2 * h_pockets[i_lower]

    denom = m2 - m1
    if abs(denom) <= tol:
        return None

    h_cut = (c1 - c2 - dt_cut) / denom
    if not np.isfinite(h_cut):
        return None

    hot_bounds = sorted((h_pockets[i_upper], h_pockets[i_upper + 1]))
    cold_bounds = sorted((h_pockets[i_lower - 1], h_pockets[i_lower]))
    h_min = max(hot_bounds[0], cold_bounds[0])
    h_max = min(hot_bounds[1], cold_bounds[1])

    if h_cut < h_min - tol or h_cut > h_max + tol:
        return None
    return min(max(h_cut, h_min), h_max)


def _cut_temperatures_needing_insertion(
    t_vals: np.ndarray,
    h_pockets: np.ndarray,
    i_upper: int,
    i_lower: int,
    h_cut: float,
    dt_cut: float,
) -> np.ndarray:
    """Return new breakpoint temperatures for the residual ``h_cut`` intersections."""
    if not (
        i_upper < i_lower - 1
        and t_vals[i_upper + 1] - t_vals[i_lower - 1] - dt_cut < -tol
    ):
        return np.empty(0, dtype=float)

    cut_temps = [
        temp
        for temp in (
            _segment_cut_temperature(
                t_vals=t_vals,
                h_pockets=h_pockets,
                left_idx=i_upper,
                right_idx=i_upper + 1,
                h_cut=h_cut,
            ),
            _segment_cut_temperature(
                t_vals=t_vals,
                h_pockets=h_pockets,
                left_idx=i_lower - 1,
                right_idx=i_lower,
                h_cut=h_cut,
            ),
        )
        if temp is not None
    ]
    if not cut_temps:
        return np.empty(0, dtype=float)
    return np.asarray(cut_temps, dtype=float)


def _segment_cut_temperature(
    t_vals: np.ndarray,
    h_pockets: np.ndarray,
    left_idx: int,
    right_idx: int,
    h_cut: float,
) -> float | None:
    """Return the breakpoint temperature where one segment crosses ``h_cut``."""
    if (
        abs(h_pockets[left_idx] - h_cut) <= tol
        or abs(h_pockets[right_idx] - h_cut) <= tol
    ):
        return None

    return linear_interpolation(
        h_cut,
        h_pockets[right_idx],
        h_pockets[left_idx],
        t_vals[right_idx],
        t_vals[left_idx],
    )


def _apply_h_cut_to_positive_run(
    h_pockets: np.ndarray,
    run_start: int,
    run_stop: int,
    h_cut: float,
) -> None:
    """Apply the cut height across one positive run and clip exhausted tails to zero."""
    segment = slice(run_start, run_stop)
    h_pockets[segment] = _normalise_pocket_profile(h_pockets[segment] - h_cut)
