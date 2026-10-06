"""Target multiple utilities over a heating or cooling profile from the pinch."""

from __future__ import annotations

import warnings
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
        _warn_on_unmet_utility_demand(
            hot_required=abs(float(pt[ProblemTableLabel.H_NET_COLD][0])),
            cold_required=abs(float(pt[ProblemTableLabel.H_NET_HOT][-1])),
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
            idx=idx,
            scale=_process_duty_scale(pt),
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


# Unmet utility demand (kW) below this is rounding, not a shortfall.
_UNMET_DEMAND_ABS_TOL = 1e-3

_DEFAULT_UTILITY_FLAG = "_is_default_utility"


def mark_default_utility(stream) -> None:
    """Flag ``stream`` as a default HU/CU added during input preparation."""
    setattr(stream, _DEFAULT_UTILITY_FLAG, True)


def is_default_utility(stream) -> bool:
    """Return whether ``stream`` is a default HU/CU (a backup utility)."""
    return bool(getattr(stream, _DEFAULT_UTILITY_FLAG, False))


def _process_duty_scale(pt: ProblemTable) -> float:
    """Largest hot or cold composite duty of the table, in kW (0 if absent)."""
    scale = 0.0
    for label in (ProblemTableLabel.H_HOT, ProblemTableLabel.H_COLD):
        if label.value in pt.columns:
            values = np.abs(np.asarray(pt[label], dtype=float))
            if values.size and np.isfinite(values).any():
                scale = max(scale, float(np.nanmax(values)))
    return scale


def _warn_on_unmet_utility_demand(
    *,
    hot_required: float,
    cold_required: float,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    idx: int | None,
    scale: float = 0.0,
) -> None:
    """Warn when the assigned utility duty falls short of the target.

    A shortfall means the utilities given cannot meet the demand, for example
    because of ``maximum_heat_flow`` caps or a fixed segmented profile. The
    reported utility totals and costs then understate the target. ``scale``
    is the size of the problem (kW): a residue far below it, left by the
    cascade arithmetic, is not a shortfall.
    """
    for label, required, utilities in (
        ("hot", hot_required, hot_utilities),
        ("cold", cold_required, cold_utilities),
    ):
        assigned = sum(
            float(StreamCollection._value_at_idx(utility.heat_flow, idx))
            for utility in utilities
        )
        shortfall = required - assigned
        # Ignore numerical residue: below 1 W, or a millionth of the target or
        # of the problem's duty.
        if shortfall > max(
            tol, _UNMET_DEMAND_ABS_TOL, 1e-6 * max(required, float(scale))
        ):
            warnings.warn(
                f"The {label} utilities meet {assigned:.6g} of the {required:.6g} "
                f"{label} utility target; {shortfall:.6g} is unmet. Add a "
                f"{label} utility or raise its maximum_heat_flow.",
                UserWarning,
                stacklevel=2,
            )


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


def _apply_utility_duties(
    utilities: StreamCollection,
    duties: tuple[float, ...],
    *,
    idx: int | None,
) -> StreamCollection:
    """Replace the selected-period duty of every utility."""
    for utility, duty in zip(utilities, duties, strict=True):
        assigned_duty = float(duty) if duty > tol else 0.0
        if utility.has_segments:
            utility._set_segmented_total_heat_flow_at_idx(assigned_duty, idx=idx)
        else:
            utility.set_value_attr_at_idx(
                attr_name="heat_flow",
                value=assigned_duty,
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

    if (
        T_segment.ndim != 1
        or H_segment.ndim != 1
        or len(T_segment) != len(H_segment)
        or not np.isfinite(H_segment).all()
        or np.any(H_segment < -tol)
    ):
        raise ValueError(
            "Error in utility targeting. Please report the data that produced "
            "this error."
        )

    utilities = tuple(u_ls)
    duties = [0.0] * len(utilities)
    levels = []
    for u in utilities:
        if is_real_temperatures:
            t_lo, t_hi = u.minimum_temperature, u.maximum_temperature
        else:
            t_lo, t_hi = u.shifted_minimum_temperature, u.shifted_maximum_temperature
        if is_hot_ut:
            levels.append((float(t_hi[idx]), float(t_lo[idx])))
        else:
            levels.append((float(t_lo[idx]), float(t_hi[idx])))
    # Use the least valuable utility first: the coldest hot utility and the
    # hottest cold utility, by shifted level in this period. The collection's
    # own order compares whole multi-period values and can differ from this.
    # A default HU/CU added during input preparation is a backup for what the
    # given utilities cannot supply (for example because of maximum_heat_flow
    # caps), so it always goes last. A user utility named HU/CU is not one.
    collection_order = (
        range(len(utilities) - 1, -1, -1) if is_hot_ut else range(len(utilities))
    )
    indices = sorted(
        collection_order,
        key=lambda i: (
            is_default_utility(utilities[i]),
            *(
                (levels[i][0], levels[i][1])
                if is_hot_ut
                else (-levels[i][0], -levels[i][1])
            ),
        ),
    )
    Q_assigned = 0.0
    for utility_index in indices:
        u = utilities[utility_index]
        Ts, Tt = levels[utility_index]

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
