"""Internal postprocessing helpers for HPR residual utility accounting."""

from __future__ import annotations

import numpy as np

from ....analysis.targeting.grand_composite import (
    get_GCC_without_pockets,
    get_seperated_gcc_heat_load_profiles,
)
from ....analysis.targeting.utilities import (
    _apply_utility_duties,
    target_utilities_for_load_profiles,
)
from ....domain.configuration import tol
from ....domain.enums import ProblemTableLabel
from ....domain.hpr import HPRResidualProfile
from ....domain.problem_table import ProblemTable


def _get_hpr_residual_utility_summary(
    *,
    pt: ProblemTable,
    base_target,
    period_idx: int,
    is_direct: bool,
    is_heat_pumping: bool,
) -> dict:
    residual_net = _get_hpr_residual_net_profile(
        pt=pt,
        is_direct=is_direct,
        is_heat_pumping=is_heat_pumping,
    )
    utility_T_vals, utility_net = _get_hpr_residual_utility_net_profile(
        T_vals=pt[ProblemTableLabel.T],
        residual_net=residual_net,
    )
    hot_profile, cold_profile = _get_hpr_residual_load_profiles(
        pt=pt,
        T_vals=utility_T_vals,
        residual_net=utility_net,
        is_direct=is_direct,
        is_heat_pumping=is_heat_pumping,
    )
    hot_utilities, cold_utilities = _retarget_hpr_residual_utilities(
        T_vals=utility_T_vals,
        residual_net=utility_net,
        hot_profile=hot_profile,
        cold_profile=cold_profile,
        hot_utilities=base_target.hot_utilities.copy(deep=True),
        cold_utilities=base_target.cold_utilities.copy(deep=True),
        is_real_temperatures=not is_direct,
        period_idx=period_idx,
    )
    hot_utility_target = float(
        hot_utilities.sum_stream_attribute("heat_flow", idx=period_idx)
    )
    cold_utility_target = float(
        cold_utilities.sum_stream_attribute("heat_flow", idx=period_idx)
    )
    base_heat_recovery = float(getattr(base_target, "heat_recovery_target", 0.0) or 0.0)
    base_hot_utility_target = float(
        getattr(base_target, "hot_utility_target", 0.0) or 0.0
    )
    base_cold_utility_target = float(
        getattr(base_target, "cold_utility_target", 0.0) or 0.0
    )
    heat_recovery_target = (
        base_heat_recovery + (base_cold_utility_target - cold_utility_target)
        if is_direct
        else base_heat_recovery + (base_hot_utility_target - hot_utility_target)
    )
    heat_recovery_limit = getattr(base_target, "heat_recovery_limit", None)
    hot_pinch, cold_pinch = ProblemTable(
        {
            ProblemTableLabel.T: utility_T_vals,
            ProblemTableLabel.H_NET: utility_net,
        }
    ).pinch_temperatures(col_H=ProblemTableLabel.H_NET)
    profile_temperatures, profile_net, profile_heating, profile_cooling = (
        _deduplicate_residual_profile_rows(
            temperatures=utility_T_vals,
            net=utility_net,
            heating=np.maximum(hot_profile, 0.0),
            cooling=np.abs(cold_profile),
        )
    )

    return {
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "hot_utility_target": hot_utility_target,
        "cold_utility_target": cold_utility_target,
        "heat_recovery_target": heat_recovery_target,
        "heat_recovery_limit": heat_recovery_limit,
        "degree_of_int": (
            (heat_recovery_target / heat_recovery_limit)
            if isinstance(heat_recovery_limit, float | int) and heat_recovery_limit > 0
            else (1.0 if heat_recovery_limit == 0 else None)
        ),
        "utility_cost": _compute_utility_cost(
            hot_utilities, cold_utilities, period_idx=period_idx
        ),
        "hot_pinch": hot_pinch,
        "cold_pinch": cold_pinch,
        "residual_profile": HPRResidualProfile(
            temperatures=tuple(profile_temperatures),
            net=tuple(profile_net),
            heating=tuple(profile_heating),
            cooling=tuple(profile_cooling),
            temperature_basis="shifted" if is_direct else "real",
        ),
    }


def _deduplicate_residual_profile_rows(
    *,
    temperatures: np.ndarray,
    net: np.ndarray,
    heating: np.ndarray,
    cooling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse equivalent duplicate temperatures at the detached contract edge."""
    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (temperatures, net, heating, cooling)
    )
    if any(values.ndim != 1 for values in arrays) or len({len(v) for v in arrays}) != 1:
        raise ValueError("HPR residual profile columns must be aligned 1D arrays.")
    if not len(arrays[0]) or any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("HPR residual profile columns must be finite and nonempty.")

    kept = [0]
    for index in range(1, len(arrays[0])):
        previous = kept[-1]
        if abs(arrays[0][previous] - arrays[0][index]) <= tol:
            if any(
                not np.isclose(values[previous], values[index], rtol=1e-9, atol=tol)
                for values in arrays[1:]
            ):
                raise ValueError(
                    "Duplicate HPR residual temperatures contain conflicting values."
                )
            continue
        kept.append(index)
    indices = np.asarray(kept, dtype=int)
    return tuple(values[indices] for values in arrays)


def _get_hpr_residual_net_profile(
    *,
    pt: ProblemTable,
    is_direct: bool,
    is_heat_pumping: bool,
) -> np.ndarray:
    base_col = ProblemTableLabel.H_NET_W_AIR
    hpr_col = (
        ProblemTableLabel.H_NET_HP if is_heat_pumping else ProblemTableLabel.H_NET_RFRG
    )
    return np.asarray(pt[base_col] - pt[hpr_col], dtype=float)


def _get_hpr_residual_utility_net_profile(
    *,
    T_vals: np.ndarray,
    residual_net: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual_net = np.asarray(residual_net, dtype=float)
    if not residual_net.size or not np.isfinite(residual_net).all():
        raise ValueError("HPR residual cascade must be finite and nonempty.")
    # Reintegrate the combined cascade before finding its pinch. In particular,
    # a tiny positive minimum must not turn a valid pinch into an absent one.
    residual_net = residual_net - residual_net.min()
    residual_pt = ProblemTable(
        {ProblemTableLabel.T: T_vals, ProblemTableLabel.H_NET: residual_net}
    )
    get_GCC_without_pockets(residual_pt)
    return (
        np.asarray(residual_pt[ProblemTableLabel.T], dtype=float),
        np.asarray(residual_pt[ProblemTableLabel.H_NET_NP], dtype=float),
    )


def _get_hpr_residual_load_profiles(
    *,
    pt: ProblemTable,
    T_vals: np.ndarray,
    residual_net: np.ndarray,
    is_direct: bool,
    is_heat_pumping: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if is_direct:
        updates = get_seperated_gcc_heat_load_profiles(
            T_col=T_vals,
            H_net=residual_net,
            is_process_stream=True,
        )["updates"]
        process_hot_profile = np.asarray(
            updates[ProblemTableLabel.H_NET_HOT], dtype=float
        )
        process_cold_profile = np.asarray(
            updates[ProblemTableLabel.H_NET_COLD], dtype=float
        )
        hot_profile = process_cold_profile
        cold_profile = process_hot_profile
        hot_after_col = (
            ProblemTableLabel.H_NET_HOT_AFTR_HP
            if is_heat_pumping
            else ProblemTableLabel.H_NET_HOT_AFTR_RFRG
        )
        cold_after_col = (
            ProblemTableLabel.H_NET_COLD_AFTR_HP
            if is_heat_pumping
            else ProblemTableLabel.H_NET_COLD_AFTR_RFRG
        )
        stored_profiles = {
            hot_after_col: process_hot_profile,
            cold_after_col: process_cold_profile,
        }
    else:
        rcp_net = (
            np.asarray(pt[ProblemTableLabel.RCP_UT_NET], dtype=float)
            if (
                ProblemTableLabel.RCP_UT_NET.value in pt.columns
                and len(pt[ProblemTableLabel.RCP_UT_NET]) == len(T_vals)
            )
            else np.zeros_like(residual_net)
        )
        updates = get_seperated_gcc_heat_load_profiles(
            T_col=T_vals,
            H_net=residual_net,
            rcp_net=rcp_net,
            is_process_stream=False,
        )["updates"]
        hot_profile = np.asarray(updates[ProblemTableLabel.H_HOT_UT], dtype=float)
        cold_profile = np.asarray(updates[ProblemTableLabel.H_COLD_UT], dtype=float)
        hot_after_col = (
            ProblemTableLabel.H_NET_HOT_UT_AFTR_HP
            if is_heat_pumping
            else ProblemTableLabel.H_NET_HOT_UT_AFTR_RFRG
        )
        cold_after_col = (
            ProblemTableLabel.H_NET_COLD_UT_AFTR_HP
            if is_heat_pumping
            else ProblemTableLabel.H_NET_COLD_UT_AFTR_RFRG
        )
        stored_profiles = {
            hot_after_col: hot_profile,
            cold_after_col: cold_profile,
        }

    pt.update(T_col=T_vals, updates=stored_profiles)
    return hot_profile, cold_profile


def _retarget_hpr_residual_utilities(
    *,
    T_vals: np.ndarray,
    residual_net: np.ndarray,
    hot_profile: np.ndarray,
    cold_profile: np.ndarray,
    hot_utilities,
    cold_utilities,
    is_real_temperatures: bool,
    period_idx: int,
):
    _apply_utility_duties(hot_utilities, (0.0,) * len(hot_utilities), idx=period_idx)
    _apply_utility_duties(cold_utilities, (0.0,) * len(cold_utilities), idx=period_idx)
    pinch_idx = ProblemTable(
        {ProblemTableLabel.T: T_vals, ProblemTableLabel.H_NET: residual_net}
    ).pinch_idx(ProblemTableLabel.H_NET)
    return target_utilities_for_load_profiles(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        T_vals=T_vals,
        H_net_cold=hot_profile,
        H_net_hot=cold_profile,
        pinch_idx=pinch_idx,
        is_real_temperatures=is_real_temperatures,
        idx=period_idx,
    )


def _compute_utility_cost(hot_utilities, cold_utilities, *, period_idx: int) -> float:
    utility_cost = 0.0
    for utility in hot_utilities + cold_utilities:
        if utility.utility_cost is None:
            continue
        utility_cost += float(utility.utility_cost[period_idx])
    return utility_cost
