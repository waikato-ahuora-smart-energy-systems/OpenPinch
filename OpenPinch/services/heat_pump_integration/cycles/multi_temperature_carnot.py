"""Multi-temperature Carnot HP targeting."""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from ....lib.schemas.hpr import HeatPumpTargetInputs, HPRBackendResult, HPRParsedState
from ....utils.decorators import timing_decorator
from ..common.encoding import AMBIENT_X_BOUNDS, map_x_arr_to_T_arr, map_x_to_Q_amb
from ..common.layout import HPRoptVectorLayout
from ..common.shared import (
    calc_carnot_heat_engine_eta,
    calc_carnot_heat_pump_cop,
    compute_entropic_mean_temperature,
    evaluate_carnot_hpr_result,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
    solve_hpr_placement,
    tol,
)

__all__ = [
    "optimise_multi_temperature_carnot_heat_pump_placement",
]


#######################################################################################################
# Public API
#######################################################################################################


@timing_decorator
def optimise_multi_temperature_carnot_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise multi-temperature Carnot stages for the prepared HPR case."""
    x0_ls, bnds = _get_multi_temperature_carnot_hp_opt_setup(args)
    return solve_hpr_placement(
        f_obj=_compute_multi_temperature_carnot_cycle_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


#######################################################################################################
# Helper Functions
#######################################################################################################


def _get_multi_temperature_carnot_hp_opt_setup(
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, list]:
    layout = HPRoptVectorLayout(n_cond=int(args.n_cond), n_evap=int(args.n_evap))
    return layout.pack(
        x_amb=0.0,
        x_cond=[0.0] * layout.n_cond,
        x_evap=[0.0] * layout.n_evap,
    ), layout.build_bounds(
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
    )


def _parse_multi_temperature_carnot_cycle_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    parts = HPRoptVectorLayout(n_cond=args.n_cond, n_evap=args.n_evap).unpack(x)
    x_amb = parts["x_amb"]
    x_cond = parts["x_cond"]
    x_evap = parts["x_evap"]

    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    T_cond = map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    return HPRParsedState(
        T_cond=T_cond,
        T_evap=T_evap,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
    )


def _get_unique_idx_mtc(is_on: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i_c, i_e = np.nonzero(is_on)
    return np.unique(i_c), np.unique(i_e)


def _get_heat_engine_and_recovery_duty(
    is_on: np.ndarray,
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    Qc_pool: np.ndarray,
    Qe_pool: np.ndarray,
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray, np.ndarray]:
    Qc_he = np.zeros_like(Qc_pool)
    Qe_he = np.zeros_like(Qe_pool)
    if np.any(is_on):  # Heat engine or heat recovery
        i_c, i_e = _get_unique_idx_mtc(is_on)
        Qc_pool_sum, Qe_pool_sum = Qc_pool[i_c].sum(), Qe_pool[i_e].sum()
        if Qc_pool_sum * Qe_pool_sum > tol:
            if 1 > args.eta_ii_he_carnot > 0:
                T_h = compute_entropic_mean_temperature(T_evap[i_e], Qe_pool[i_e])
                T_l = compute_entropic_mean_temperature(T_cond[i_c], Qc_pool[i_c])
                eta_he = calc_carnot_heat_engine_eta(T_h, T_l, args.eta_ii_he_carnot)
                Qe_used = min(Qe_pool_sum, Qc_pool_sum / max(1.0 - eta_he, tol))
                Qc_used = Qe_used * (1 - eta_he)
            else:
                Qe_used = Qc_used = min(Qe_pool_sum, Qc_pool_sum)

            Qc_he[i_c] = Qc_pool[i_c] * (Qc_used / Qc_pool_sum)
            Qe_he[i_e] = Qe_pool[i_e] * (Qe_used / Qe_pool_sum)

    return Qc_he, Qe_he


def _get_heat_pump_duty(
    is_on: np.ndarray,
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    Qc_pool: np.ndarray,
    Qe_pool: np.ndarray,
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray, np.ndarray]:
    Qc_hpr = np.zeros_like(Qc_pool)
    Qe_hpr = np.zeros_like(Qe_pool)
    if np.any(is_on):  # Heat pump
        i_c, i_e = _get_unique_idx_mtc(is_on)
        Qc_pool_sum, Qe_pool_sum = Qc_pool[i_c].sum(), Qe_pool[i_e].sum()
        if (Qc_pool_sum * Qe_pool_sum > tol) and (1 > args.eta_ii_hpr_carnot >= 0):
            T_h = compute_entropic_mean_temperature(T_cond[i_c], Qc_pool[i_c])
            T_l = compute_entropic_mean_temperature(T_evap[i_e], Qe_pool[i_e])
            if T_h > T_l and args.eta_ii_hpr_carnot > 0.0:
                cop = calc_carnot_heat_pump_cop(T_h, T_l, args.eta_ii_hpr_carnot)
                Qe_used = min(Qe_pool_sum, Qc_pool_sum * (cop - 1.0) / cop)
                w_hpr = Qe_used / (cop - 1.0)
                Qc_used = Qe_used + w_hpr
                Qc_hpr[i_c] = Qc_pool[i_c] * (Qc_used / Qc_pool_sum)
                Qe_hpr[i_e] = Qe_pool[i_e] * (Qe_used / Qe_pool_sum)

    return Qc_hpr, Qe_hpr


def _opt_wrapper_for_heat_pump(
    x: np.float64,
    is_on: np.ndarray,
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_cond: np.ndarray,
    Q_evap: np.ndarray,
    Qc_he: np.ndarray,
    Qe_he: np.ndarray,
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray, np.ndarray]:
    Qc_hpr, Qe_hpr = _get_heat_pump_duty(
        is_on=is_on,
        T_cond=T_cond,
        T_evap=T_evap,
        Qc_pool=Q_cond - Qc_he * x,
        Qe_pool=Q_evap - Qe_he * x,
        args=args,
    )
    w_he = (Qe_he.sum() - Qc_he.sum()) * x
    w_hpr = Qc_hpr.sum() - Qe_hpr.sum()
    return w_hpr - w_he


def _get_multi_temperature_carnot_stage_duties_and_work(
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    H_hot_with_amb: np.ndarray,
    H_cold_with_amb: np.ndarray,
    args: HeatPumpTargetInputs,
) -> dict:
    Q_cond = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )
    Q_evap = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap, args.T_hot, H_hot_with_amb, is_cond=False
    )

    cop = 1.0

    T_diff = np.subtract.outer(T_cond, T_evap)
    is_hp = T_diff > tol

    Qc_he, Qe_he = _get_heat_engine_and_recovery_duty(
        is_on=~is_hp,
        T_cond=T_cond,
        T_evap=T_evap,
        Qc_pool=Q_cond,
        Qe_pool=Q_evap,
        args=args,
    )

    if np.any(~is_hp):
        fun = lambda x: _opt_wrapper_for_heat_pump(
            x=x,
            is_on=is_hp,
            T_cond=T_cond,
            T_evap=T_evap,
            Q_cond=Q_cond,
            Q_evap=Q_evap,
            Qc_he=Qc_he,
            Qe_he=Qe_he,
            args=args,
        )
        res = minimize_scalar(
            fun=fun,
            bounds=(0, 1),
            # method='brent',
        )
        Qc_he *= res.x
        Qe_he *= res.x

    Qc_hpr, Qe_hpr = _get_heat_pump_duty(
        is_on=is_hp,
        T_cond=T_cond,
        T_evap=T_evap,
        Qc_pool=Q_cond - Qc_he,
        Qe_pool=Q_evap - Qe_he,
        args=args,
    )

    w_he = Qe_he.sum() - Qc_he.sum()
    w_hpr = Qc_hpr.sum() - Qe_hpr.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        cop = np.where(
            w_hpr > 0,
            Qc_hpr.sum() / w_hpr,
            1.0,
        ).item()

    if not np.isclose(
        Qc_he.sum() + Qc_hpr.sum() + w_he, Qe_he.sum() + Qe_hpr.sum() + w_hpr, atol=tol
    ):
        raise ValueError(
            "Energy balance not satisfied in multi-temperature Carnot cycle calculation."
        )

    return {
        "w_hpr": w_hpr,
        "w_he": w_he,
        "heat_recovery": Qc_he,
        "cop": cop,
        "Qc": Qc_hpr + Qc_he,
        "Qe": Qe_hpr + Qe_he,
        "Qc_hpr": Qc_hpr,
        "Qe_hpr": Qe_hpr,
        "Qc_he": Qc_he,
        "Qe_he": Qe_he,
    }


def _compute_multi_temperature_carnot_cycle_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> HPRBackendResult:
    state_vars = _parse_multi_temperature_carnot_cycle_state_variables(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)

    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars.Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars.Q_amb_hot

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond=state_vars.T_cond,
        T_evap=state_vars.T_evap,
        H_hot_with_amb=H_hot_with_amb,
        H_cold_with_amb=H_cold_with_amb,
        args=args,
    )

    work = cycle_results["w_hpr"] - cycle_results["w_he"]
    return evaluate_carnot_hpr_result(
        args=args,
        state=state_vars,
        w_net=work,
        w_hpr=cycle_results["w_hpr"],
        w_he=cycle_results["w_he"],
        heat_recovery=cycle_results["heat_recovery"],
        cop_h=cycle_results["cop"],
        Q_cond_total=cycle_results["Qc_hpr"] + cycle_results["Qc_he"],
        Q_evap_total=cycle_results["Qe_hpr"] + cycle_results["Qe_he"],
        Q_cond=cycle_results["Qc_hpr"],
        Q_evap=cycle_results["Qe_hpr"],
        Q_cond_he=cycle_results["Qc_he"],
        Q_evap_he=cycle_results["Qe_he"],
        debug=debug,
    )
