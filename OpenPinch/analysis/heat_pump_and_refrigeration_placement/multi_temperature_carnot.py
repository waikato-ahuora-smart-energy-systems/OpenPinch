"""Multi-temperature Carnot HP targeting."""

from typing import Tuple

import numpy as np

from ...lib.schema import HPRTargetInputs, HPRTargetOutputs
from ...utils.decorators import timing_decorator

from .encoding import map_x_arr_to_T_arr, map_x_to_Q_amb
from .shared import (
    calc_carnot_heat_engine_eta,
    calc_carnot_heat_pump_cop,
    calc_hpr_obj,
    compute_entropic_mean_temperature,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
    get_carnot_hpr_cycle_streams,
    solve_hpr_placement,
    g_ineq_penalty,
    plot_multi_hp_profiles_from_results,
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
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    res = solve_hpr_placement(
        f_obj=_compute_multi_temperature_carnot_cycle_obj,
        x0_ls=_get_x0_for_multi_temperature_carnot_hp_opt(args),
        bnds=_get_bounds_for_multi_temperature_carnot_hp_opt(args),
        args=args,
    )
    res.update(
        get_carnot_hpr_cycle_streams(
            res["T_cond"], res["Q_cond"], res["T_evap"], res["Q_evap"], args
        )
    )
    return HPRTargetOutputs.model_validate(res)


#######################################################################################################
# Helper Functions
#######################################################################################################


def _get_x0_for_multi_temperature_carnot_hp_opt(args: HPRTargetInputs) -> list:
    n_cond, n_evap = int(args.n_cond), int(args.n_evap)
    return [0.0] + [0.0] * n_cond + [0.0] * n_evap


def _get_bounds_for_multi_temperature_carnot_hp_opt(args: HPRTargetInputs) -> list:
    n_cond, n_evap = int(args.n_cond), int(args.n_evap)
    return [(-1.0, 10.0)] + [(0.0, 1.0)] * n_cond + [(0.0, 1.0)] * n_evap


def _parse_multi_temperature_carnot_cycle_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    x_amb = x[0]
    a = 1 + int(args.n_cond)
    x_cond = x[1:a]
    b = a + int(args.n_evap)
    x_evap = x[a:b]

    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    T_cond = map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    return {
        "T_cond": T_cond,
        "T_evap": T_evap,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
    }


def _get_unique_idx_mtc(is_on: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i_c, i_e = np.nonzero(is_on)
    return np.unique(i_c), np.unique(i_e)


def _get_multi_temperature_carnot_stage_duties_and_work(
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    H_hot_with_amb: np.ndarray,
    H_cold_with_amb: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    Q_cond = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )
    Q_evap = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap, args.T_hot, H_hot_with_amb, is_cond=False
    )

    Qc_he = np.zeros_like(Q_cond)
    Qe_he = np.zeros_like(Q_evap)
    Qc_hpr = np.zeros_like(Q_cond)
    Qe_hpr = np.zeros_like(Q_evap)
    w_he = 0.0
    w_hpr = 0.0
    cop = 1.0

    T_diff = np.subtract.outer(T_cond, T_evap)

    is_hp = (T_diff > tol)
    if np.any(~is_hp): # Heat engine or heat recovery
        i_c, i_e = _get_unique_idx_mtc(~is_hp)
        Qc_pool, Qe_pool = Q_cond, Q_evap
        Qc_pool_sum, Qe_pool_sum = Qc_pool.sum(), Qe_pool.sum()
        if (Qc_pool_sum * Qe_pool_sum > tol):
            if 1 > args.eta_ii_he_carnot > 0:
                T_h = compute_entropic_mean_temperature(T_evap[i_e], Qe_pool[i_e])
                T_l = compute_entropic_mean_temperature(T_cond[i_c], Qc_pool[i_c])
                eta_he = calc_carnot_heat_engine_eta(T_h, T_l, args.eta_ii_he_carnot)
                Qe_used = min(Qe_pool_sum, Qc_pool_sum / max(1.0 - eta_he, tol))
                w_he = (Qe_used * eta_he)
                Qc_used = Qe_used - w_he
            else:
                Qe_used = Qc_used = min(Qe_pool_sum, Qc_pool_sum)
            
            Qc_he[i_c] = Qc_pool * (Qc_used / Qc_pool_sum)
            Qe_he[i_e] = Qe_pool * (Qe_used / Qe_pool_sum)

    if np.any(is_hp): # Heat pump
        i_c, i_e = _get_unique_idx_mtc(is_hp)
        Qc_pool = Q_cond - Qc_he
        Qe_pool = Q_evap - Qe_he       
        Qc_pool_sum, Qe_pool_sum = Qc_pool.sum(), Qe_pool.sum()
        if (Qc_pool_sum * Qe_pool_sum > tol) and (1 > args.eta_ii_hpr_carnot >= 0):
            T_h = compute_entropic_mean_temperature(T_cond[i_c], Qc_pool[i_c])
            T_l = compute_entropic_mean_temperature(T_evap[i_e], Qe_pool[i_e])
            if T_h > T_l and args.eta_ii_hpr_carnot > 0.0:
                cop = calc_carnot_heat_pump_cop(T_h, T_l, args.eta_ii_hpr_carnot)
                Qe_used = min(Qe_pool_sum, Qc_pool_sum * (cop - 1.0) / cop)
                w_hpr = Qe_used / (cop - 1.0)
                Qc_used = Qe_used + w_hpr
                Qc_hpr[i_c] = Qc_pool * (Qc_used / Qc_pool_sum)
                Qe_hpr[i_e] = Qe_pool * (Qe_used / Qe_pool_sum)

    Qc = Qc_he + Qc_hpr
    Qe = Qe_he + Qe_hpr
    heat_recovery = Qc_he.sum()

    if not np.isclose(Qc.sum() + w_he, Qe.sum() + w_hpr, atol=tol):
        raise ValueError(
            "Energy balance not satisfied in multi-temperature Carnot cycle calculation."
        )

    return {
        "w_hpr": w_hpr,
        "w_he": w_he,
        "heat_recovery": heat_recovery,
        "cop": cop,
        "Qc": Qc,
        "Qe": Qe,
    }


def _compute_multi_temperature_carnot_cycle_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    state_vars = _parse_multi_temperature_carnot_cycle_state_variables(x, args)

    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars["Q_amb_cold"]
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars["Q_amb_hot"]

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond=state_vars["T_cond"],
        T_evap=state_vars["T_evap"],
        H_hot_with_amb=H_hot_with_amb,
        H_cold_with_amb=H_cold_with_amb,
        args=args,
    )

    work = cycle_results["w_hpr"] - cycle_results["w_he"]
    Q_ext_heat = max(np.abs(H_cold_with_amb[0]) - cycle_results["Qc"].sum(), 0.0)
    Q_ext_cold = max(np.abs(H_hot_with_amb[-1]) - cycle_results["Qe"].sum(), 0.0)

    p = (
        g_ineq_penalty(g=Q_ext_cold, rho=args.rho_penalty, form="square")
        if not args.is_heat_pumping
        else 0.0
    )
    obj = calc_hpr_obj(
        work=work,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:
        res = get_carnot_hpr_cycle_streams(
            state_vars["T_cond"],
            cycle_results["Qc"],
            state_vars["T_evap"],
            cycle_results["Qe"],
            args,
        )
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            res["hpr_hot_streams"],
            res["hpr_cold_streams"],
            title=f"Obj {float(obj):.5f} = {(work / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )

    return {
        "obj": obj,
        "utility_tot": work + Q_ext_heat + Q_ext_cold,
        "w_net": work,
        "w_hpr": cycle_results["w_hpr"],
        "w_he": cycle_results["w_he"],
        "heat_recovery": cycle_results["heat_recovery"],
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "Q_cond": cycle_results["Qc"],
        "T_evap": state_vars["T_evap"],
        "Q_evap": cycle_results["Qe"],
        "cop_h": cycle_results["cop"],
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "success": True,
    }
