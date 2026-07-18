"""Preprocessing helpers for heat pump and refrigeration targeting."""

from typing import Tuple

import numpy as np

from ....analysis.graphs.composite import clean_composite_curve
from ....analysis.numerics import delta_vals, linear_interpolation
from ....contracts.hpr import HeatPumpTargetInputs
from ....domain.configuration import Configuration, tol
from ....domain.enums import ProblemTableLabel
from ....domain.problem_table import ProblemTable
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection

__all__ = ["construct_HPRTargetInputs"]


################################################################################
# Public API
################################################################################


def construct_HPRTargetInputs(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    *,
    is_heat_pumping: bool = True,
    config: Configuration,
    period_idx: int = 0,
    debug: bool = False,
) -> HeatPumpTargetInputs:
    """Prepare normalised background cascades and solver arguments for HPR targeting."""
    T_vals, H_hot, H_cold = T_vals.copy(), H_hot.copy(), H_cold.copy()
    hpr = config.hpr
    costing = config.costing
    environment = config.environment
    thermal = config.thermal
    T_hot, T_cold = _apply_temperature_shift_for_hpr_stream_dtmin_cont(
        T_vals, hpr.dt_cont
    )

    T_hot, H_hot, z_amb_hot, s_hot = _prepare_hpr_background_profile(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_hot,
        H_vals=H_hot,
        config=config,
        is_heat_pumping=is_heat_pumping,
        is_cold=False,
    )
    T_cold, H_cold, z_amb_cold, s_cold = _prepare_hpr_background_profile(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_cold,
        H_vals=H_cold,
        config=config,
        is_heat_pumping=is_heat_pumping,
        is_cold=True,
    )

    return HeatPumpTargetInputs(
        # Derived targeting state.
        Q_hpr_target=Q_hpr_target,
        Q_heat_max=H_cold[0],
        Q_cool_max=-H_hot[-1],
        dt_range_max=max(T_cold[0], T_hot[0]) - min(T_cold[-1], T_hot[-1]),
        T_hot=T_hot,
        H_hot=H_hot,
        z_amb_hot=z_amb_hot,
        T_cold=T_cold,
        H_cold=H_cold,
        z_amb_cold=z_amb_cold,
        bckgrd_hot_streams=s_hot,
        bckgrd_cold_streams=s_cold,
        is_heat_pumping=bool(is_heat_pumping),
        debug=debug,
        period_idx=period_idx,
        # Direct config pass-through.
        hpr_type=hpr.type,
        hpr_comp_fixed_cost=costing.hpr_comp_fixed_cost,
        hpr_comp_variable_cost=costing.hpr_comp_variable_cost,
        hpr_comp_cost_exp=costing.hpr_comp_cost_exp,
        hpr_hx_fixed_cost=costing.hpr_hx_duty_fixed_cost,
        hpr_hx_variable_cost=costing.hpr_hx_duty_variable_cost,
        hpr_hx_cost_exp=costing.hpr_hx_duty_cost_exp,
        n_cond=hpr.n_cond,
        n_evap=hpr.n_evap,
        n_mvr=hpr.mvr_count,
        refrigerant_ls=hpr.normalised_refrigerants,
        mvr_fluid_ls=hpr.normalised_mvr_fluids,
        do_refrigerant_sort=hpr.refrigerant_sort_enabled,
        eta_comp=hpr.eta_comp,
        eta_mvr_comp=hpr.mvr_eta_comp,
        eta_motor=hpr.mvr_eta_motor,
        eta_exp=hpr.eta_exp,
        eta_ii_hpr_carnot=hpr.eta_ii_carnot,
        eta_ii_he_carnot=hpr.effective_eta_ii_he_carnot,
        dtcont_hp=hpr.dt_cont,
        dt_hp_ihx=hpr.dt_ihx,
        dt_cascade_hx=hpr.dt_cascade_hx,
        T_env=environment.temperature,
        dt_env_cont=hpr.dt_env_cont,
        dt_phase_change=thermal.dt_phase_change,
        eta_penalty=hpr.eta_penalty,
        rho_penalty=hpr.rho_penalty,
        max_multi_start=hpr.max_multistart,
        bb_minimiser=hpr.bb_minimiser,
        allow_integrated_expander=hpr.integrated_expander_enabled,
        initialise_simulated_cycle=hpr.initialise_simulated_cycle,
        heat_to_power_ratio=costing.hpr_price_ratio_heat_to_ele,
        cold_to_power_ratio=costing.hpr_price_ratio_cold_to_ele,
        ele_price=costing.hpr_ele_price,
        annual_op_time=costing.annual_op_time,
        discount_rate=costing.discount_rate,
        serv_life=costing.service_life,
    )


################################################################################
# Helper Functions
################################################################################


def _prepare_hpr_background_profile(
    *,
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    config: Configuration,
    is_heat_pumping: bool,
    is_cold: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StreamCollection]:
    should_trim_to_target = ((is_cold) and (is_heat_pumping)) or (
        (not is_cold) and (not is_heat_pumping)
    )
    if should_trim_to_target:
        T_vals, H_vals = _get_reduced_bckgrd_cascade_till_Q_target(
            Q_hpr_target, T_vals, H_vals, is_cold=is_cold
        )

    T_vals, H_vals, z_amb = _get_simplified_bckgrd_cascade_and_z_amb(
        T_vals=T_vals,
        H_vals=H_vals,
        config=config,
        is_cold=is_cold,
    )
    return (
        T_vals,
        H_vals,
        z_amb,
        _create_stream_collection_of_background_profile(T_vals, H_vals),
    )


def _create_stream_collection_of_background_profile(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> StreamCollection:
    """Convert a temperature-enthalpy profile into piecewise stream segments."""
    s = StreamCollection()

    T_vals = np.asarray(T_vals)
    H_vals = np.abs(np.asarray(H_vals))

    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    dh_vals = delta_vals(H_vals)
    dh_indices = np.argwhere(np.abs(dh_vals) > tol).flatten()
    for i in dh_indices:
        if dh_vals[i] > tol:
            s.add(
                Stream(
                    supply_temperature=T_vals[i + 1],
                    target_temperature=T_vals[i],
                    heat_flow=dh_vals[i],
                )
            )
        elif -dh_vals[i] > tol:
            s.add(
                Stream(
                    supply_temperature=T_vals[i],
                    target_temperature=T_vals[i + 1],
                    heat_flow=-dh_vals[i],
                )
            )
    return s


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
    config: Configuration,
    *,
    is_cold: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sign = 1 if is_cold else -1
    T_amb_star = (
        config.environment.temperature
        + (config.hpr.dt_env_cont + config.hpr.dt_cont) * sign
    )
    T_vals, H_vals = _add_T_amb_interval(
        T_vals, H_vals, T_amb_star, config.thermal.dt_phase_change, is_cold
    )
    z_amb = _get_z_ambient(T_vals=T_vals, T_amb_star=T_amb_star, is_cold=is_cold)
    H_vals += z_amb

    T_vals, H_vals = clean_composite_curve(T_vals, H_vals)

    z_amb = _get_z_ambient(
        T_vals=T_vals,
        T_amb_star=config.environment.temperature
        + (config.hpr.dt_env_cont + config.hpr.dt_cont) * sign,
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
    H_label = ProblemTableLabel.H_NET_COLD if is_cold else ProblemTableLabel.H_NET_HOT
    pt = ProblemTable({ProblemTableLabel.T: T_vals, H_label: H_vals})
    T_amb_ls = (
        [T_amb, T_amb + dt_phase_change]
        if is_cold
        else [T_amb, T_amb - dt_phase_change]
    )
    pt.insert_temperature_interval(T_amb_ls)
    return pt[ProblemTableLabel.T], pt[H_label]


def _extend_profile_with_temperature_margin(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    z_amb: np.ndarray,
    *,
    dt_margin: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if T_vals.size == 0:
        return T_vals, H_vals, z_amb

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
