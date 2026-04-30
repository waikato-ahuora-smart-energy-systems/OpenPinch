"""Preprocessing helpers for heat-pump and refrigeration targeting."""

from typing import Tuple

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.stream_collection import StreamCollection
from ...lib.config import Configuration
from ...lib.enums import PT
from ...lib.schema import HPRTargetInputs
from ...utils.miscellaneous import clean_composite_curve, linear_interpolation

from .shared import _create_stream_collection_of_background_profile


__all__ = [
    "_construct_HPRTargetInputs",
    "_apply_temperature_shift_for_hpr_stream_dtmin_cont",
    "_get_reduced_bckgrd_cascade_till_Q_target",
    "_get_z_ambient",
    "_get_simplified_bckgrd_cascade_and_z_amb",
    "_add_T_amb_interval",
    "_extend_profile_with_temperature_margin",
]


def _construct_HPRTargetInputs(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    *,
    is_heat_pumping: bool = True,
    zone_config: Configuration = Configuration(),
    debug: bool = False,
) -> HPRTargetInputs:
    T_vals, H_hot, H_cold = T_vals.copy(), H_hot.copy(), H_cold.copy()
    T_hot, T_cold = _apply_temperature_shift_for_hpr_stream_dtmin_cont(
        T_vals, zone_config.DT_CONT_HP
    )

    for T_arr, H_arr, is_cold in [(T_hot, H_hot, False), (T_cold, H_cold, True)]:
        if (is_cold and is_heat_pumping) or (not is_cold and not is_heat_pumping):
            T_arr, H_arr = _get_reduced_bckgrd_cascade_till_Q_target(
                Q_hpr_target, T_arr, H_arr, is_cold=is_cold
            )
        T_arr, H_arr, z_amb_arr = _get_simplified_bckgrd_cascade_and_z_amb(
            T_vals=T_arr,
            H_vals=H_arr,
            zone_config=zone_config,
            is_cold=is_cold,
        )
        s = _create_stream_collection_of_background_profile(T_arr, H_arr)
        if is_cold:
            T_cold, H_cold, z_amb_cold, s_cold = T_arr, H_arr, z_amb_arr, s
        else:
            T_hot, H_hot, z_amb_hot, s_hot = T_arr, H_arr, z_amb_arr, s

    inputs = {
        "Q_hpr_target": Q_hpr_target,
        "Q_heat_max": H_cold[0],
        "Q_cool_max": -H_hot[-1],
        "T_hot": T_hot,
        "H_hot": H_hot,
        "z_amb_hot": z_amb_hot,
        "T_cold": T_cold,
        "H_cold": H_cold,
        "z_amb_cold": z_amb_cold,
        "dt_range_max": max(T_cold[0], T_hot[0]) - min(T_cold[-1], T_hot[-1]),
        "is_heat_pumping": bool(is_heat_pumping),
        "eta_penalty": 0.001,
        "rho_penalty": 10,
        "bckgrd_hot_streams": s_hot,
        "bckgrd_cold_streams": s_cold,
        "debug": debug,
        "hpr_type": zone_config.HPR_TYPE,
        "n_cond": zone_config.N_COND,
        "n_evap": zone_config.N_EVAP,
        "eta_comp": zone_config.ETA_COMP,
        "eta_exp": zone_config.ETA_EXP,
        "eta_ii_hpr_carnot": zone_config.ETA_II_HPR_CARNOT,
        "eta_ii_he_carnot": zone_config.ETA_II_HE_CARNOT
        if zone_config.ALLOW_INTEGRATED_EXPANDER
        else 0.0,
        "dtcont_hp": zone_config.DT_CONT_HP,
        "dt_hp_ihx": zone_config.DT_HPR_IHX,
        "dt_cascade_hx": zone_config.DT_HPR_CASCADE_HX,
        "T_env": zone_config.T_ENV,
        "dt_env_cont": zone_config.DT_ENV_CONT,
        "dt_phase_change": zone_config.DT_PHASE_CHANGE,
        "refrigerant_ls": [r.strip().upper() for r in zone_config.REFRIGERANTS],
        "do_refrigerant_sort": zone_config.DO_REFRIGERANT_SORT,
        "heat_to_power_ratio": zone_config.PRICE_RATIO_HEAT_TO_ELE,
        "cold_to_power_ratio": zone_config.PRICE_RATIO_COLD_TO_ELE,
        "max_multi_start": zone_config.MAX_HP_MULTISTART,
        "bb_minimiser": zone_config.BB_MINIMISER,
        "allow_integrated_expander": zone_config.ALLOW_INTEGRATED_EXPANDER,
        "initialise_simulated_cycle": zone_config.INITIALISE_SIMULATED_CYCLE,
    }
    return HPRTargetInputs(**inputs)


def _apply_temperature_shift_for_hpr_stream_dtmin_cont(
    T_vals: np.ndarray,
    dtmin_hp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    return T_vals - dtmin_hp, T_vals + dtmin_hp


def _get_reduced_bckgrd_cascade_till_Q_target(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *,
    is_cold: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if is_cold:
        if H_vals[0] < Q_hpr_target:
            return T_vals, H_vals

        i = H_vals.size - np.searchsorted(H_vals[::-1], Q_hpr_target, side="left") - 1
        if i == T_vals.size - 1:
            raise ValueError("Target for heat pumping cannot be zero.")
        T_vals[i] = linear_interpolation(
            Q_hpr_target, H_vals[i], H_vals[i + 1], T_vals[i], T_vals[i + 1]
        )
        H_vals[i] = Q_hpr_target
        return T_vals[i:], H_vals[i:]

    if -H_vals[-1] < Q_hpr_target:
        return T_vals, H_vals

    i = np.searchsorted(-H_vals, Q_hpr_target, side="left")
    if i == 0:
        raise ValueError("Target for refrigeration cannot be zero.")
    T_vals[i] = linear_interpolation(
        -Q_hpr_target, H_vals[i], H_vals[i - 1], T_vals[i], T_vals[i - 1]
    )
    H_vals[i] = -Q_hpr_target
    return T_vals[: i + 1], H_vals[: i + 1]


def _get_z_ambient(
    T_vals: np.ndarray,
    T_amb_star: float,
    is_cold: bool,
) -> Tuple[np.ndarray, float]:
    if is_cold:
        return np.where(T_vals > T_amb_star, 1.0, 0.0)
    return np.where(T_vals < T_amb_star, -1.0, 0.0)


def _get_simplified_bckgrd_cascade_and_z_amb(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    zone_config: Configuration,
    *,
    is_cold: bool,
) -> Tuple[np.ndarray, float]:
    sign = 1 if is_cold else -1
    T_amb_star = zone_config.T_ENV + (
        zone_config.DT_ENV_CONT + zone_config.DT_CONT_HP
    ) * sign
    T_vals, H_vals = _add_T_amb_interval(
        T_vals, H_vals, T_amb_star, zone_config.DT_PHASE_CHANGE, is_cold
    )
    z_amb = _get_z_ambient(T_vals=T_vals, T_amb_star=T_amb_star, is_cold=is_cold)
    H_vals += z_amb

    T_vals, H_vals = clean_composite_curve(T_vals, H_vals)

    z_amb = _get_z_ambient(
        T_vals=T_vals,
        T_amb_star=zone_config.T_ENV
        + (zone_config.DT_ENV_CONT + zone_config.DT_CONT_HP) * sign,
        is_cold=is_cold,
    )
    H_vals -= z_amb

    T_vals, H_vals, z_amb = _extend_profile_with_temperature_margin(
        T_vals, H_vals, z_amb, dt_margin=10.0
    )
    return T_vals, H_vals, z_amb


def _add_T_amb_interval(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    T_amb: float,
    dt_phase_change: float,
    is_cold: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    H_label = PT.H_NET_COLD.value if is_cold else PT.H_NET_HOT.value
    pt = ProblemTable({PT.T.value: T_vals, H_label: H_vals})
    T_amb_ls = (
        [T_amb, T_amb + dt_phase_change]
        if is_cold
        else [T_amb, T_amb - dt_phase_change]
    )
    pt.insert_temperature_interval(T_amb_ls)
    return pt.col[PT.T.value], pt.col[H_label]


def _extend_profile_with_temperature_margin(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    z_amb: np.ndarray,
    *,
    dt_margin: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if T_vals.size == 0:
        return T_vals, H_vals

    T_ext = np.empty(T_vals.size + 2, dtype=T_vals.dtype)
    H_ext = np.empty(H_vals.size + 2, dtype=H_vals.dtype)
    z_ext = np.empty(z_amb.size + 2, dtype=z_amb.dtype)

    T_ext[0] = T_vals[0] + dt_margin
    T_ext[1:-1] = T_vals
    T_ext[-1] = T_vals[-1] - dt_margin

    H_ext[0] = H_vals[0]
    H_ext[1:-1] = H_vals
    H_ext[-1] = H_vals[-1]

    z_ext[0] = z_amb[0]
    z_ext[1:-1] = z_amb
    z_ext[-1] = z_amb[-1]
    return T_ext, H_ext, z_ext

