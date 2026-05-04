"""Multiple simple Carnot HP targeting."""

from typing import Tuple

import numpy as np

from ...lib.schema import HPRTargetInputs, HPRTargetOutputs
from ...utils.decorators import timing_decorator

from .encoding import map_x_arr_to_T_arr
from .shared import (
    calc_carnot_heat_engine_eta,
    calc_carnot_heat_pump_cop,
    calc_hpr_obj,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
    get_carnot_hpr_cycle_streams,
    solve_hpr_placement,
    g_ineq_penalty,
    plot_multi_hp_profiles_from_results,
    tol,
)


__all__ = [
    "optimise_multi_simple_carnot_heat_pump_placement",
]


#######################################################################################################
# Public API
#######################################################################################################


@timing_decorator
def optimise_multi_simple_carnot_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    args.n_cond = args.n_evap = max(args.n_cond, args.n_evap)
    res = solve_hpr_placement(
        f_obj=_compute_multi_simple_carnot_hp_opt_obj,
        x0_ls=[0.0 for _ in range(args.n_cond + args.n_evap + 1)],
        bnds=[(0.0, 1.0) for _ in range(args.n_cond + args.n_evap)] + [(-1.0, 4.0)],
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


def _parse_multi_simple_carnot_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_cond = map_x_arr_to_T_arr(x[: args.n_cond], args.T_cold[0], args.T_cold[-1])
    T_evap = args.T_hot[-1] - np.array(x[args.n_cond : -1]) * (
        args.T_hot[-1] - args.T_hot[0]
    )
    scale = max(args.Q_heat_max, args.Q_cool_max)
    Q_amb_hot = min(scale * x[-1], 0.0) * -1
    Q_amb_cold = max(scale * x[-1], 0.0)
    return {
        "T_cond": T_cond,
        "T_evap": T_evap,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
    }


def _get_multi_simple_carnot_stage_duties_and_work(
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    H_hot_with_amb: np.ndarray,
    H_cold_with_amb: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, list]:
    Q_cond = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )

    Qc_he = np.zeros_like(Q_cond)
    Qe_he = np.zeros_like(Q_cond)
    Qc_hx = np.zeros_like(Q_cond)
    Qe_hx = np.zeros_like(Q_cond)
    Qc_hpr = np.zeros_like(Q_cond)
    Qe_hpr = np.zeros_like(Q_cond)
    w_hpr = np.zeros_like(Q_cond)
    w_he = np.zeros_like(Q_cond)
    q_diff = [0.0]

    T_diff = T_cond - T_evap
    T_cond_abs = T_cond + 273.15
    T_evap_abs = T_evap + 273.15

    is_hp = T_diff >= tol
    if np.any(is_hp):
        cop_hp = calc_carnot_heat_pump_cop(
            T_cond_abs[is_hp], T_evap_abs[is_hp], args.eta_ii_hpr_carnot
        )
        Qc_hpr[is_hp] = Q_cond[is_hp]
        w_hpr[is_hp] = Qc_hpr[is_hp] / cop_hp
        Qe_hpr[is_hp] = Qc_hpr[is_hp] - w_hpr[is_hp]

    is_he = (T_diff <= -tol) & (args.eta_ii_he_carnot >= tol)
    if np.any(is_he):
        eff_he = calc_carnot_heat_engine_eta(
            T_evap_abs[is_he], T_cond_abs[is_he], args.eta_ii_he_carnot
        )
        Qc_he[is_he] = Q_cond[is_he]
        w_he[is_he] = Qc_he[is_he] * eff_he / (1 - eff_he)
        Qe_he[is_he] = Qc_he[is_he] + w_he[is_he]

    is_hx = ((T_diff > -tol) | (args.eta_ii_he_carnot < tol)) & (T_diff < tol)
    if np.any(is_hx):
        Qc_hx[is_hx] = Q_cond[is_hx]
        Qe_hx[is_hx] = Qc_hx[is_hx]

    sort_idx = np.argsort(-T_evap, kind="stable")
    Q_allocated = 0.0
    for i in sort_idx:
        Q_stage = Qe_he[i] + Qe_hx[i] + Qe_hpr[i]
        if Q_stage < tol:
            Qc_he[i] = 0.0
            Qe_he[i] = 0.0
            Qc_hx[i] = 0.0
            Qe_hx[i] = 0.0
            Qc_hpr[i] = 0.0
            Qe_hpr[i] = 0.0
            w_he[i] = 0.0
            w_hpr[i] = 0.0
            continue

        Q_available = (
            get_Q_vals_at_T_hpr_from_bckgrd_profile(
                np.array([T_evap[i]]), args.T_hot, H_hot_with_amb, is_cond=False
            )[0]
            - Q_allocated
        )
        if Q_available < tol:
            scale = 0.0
        elif Q_stage > Q_available:
            scale = Q_available / Q_stage
        else:
            scale = 1.0

        Qc_he[i] *= scale
        Qe_he[i] *= scale
        Qc_hx[i] *= scale
        Qe_hx[i] *= scale
        Qc_hpr[i] *= scale
        Qe_hpr[i] *= scale
        w_he[i] *= scale
        w_hpr[i] *= scale
        Q_allocated += Qe_he[i] + Qe_hx[i] + Qe_hpr[i]

    Qc = Qc_he + Qc_hx + Qc_hpr
    Qe = Qe_he + Qe_hx + Qe_hpr
    heat_recovery = Qc_hx.sum()

    if not np.isclose(Qc.sum() + w_he.sum(), Qe.sum() + w_hpr.sum(), atol=tol):
        raise ValueError(
            "Energy balance not satisfied in multiple simple Carnot cycle calculation."
        )

    return {
        "w_hpr": w_hpr,
        "w_he": w_he,
        "heat_recovery": heat_recovery,
        "Qc": Qc,
        "Qe": Qe,
        "q_diff": q_diff,
    }


def _compute_multi_simple_carnot_hp_opt_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    state_vars = _parse_multi_simple_carnot_hp_state_variables(x, args)
    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars["Q_amb_cold"]
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars["Q_amb_hot"]

    cycle_results = _get_multi_simple_carnot_stage_duties_and_work(
        T_cond=state_vars["T_cond"],
        T_evap=state_vars["T_evap"],
        H_hot_with_amb=H_hot_with_amb,
        H_cold_with_amb=H_cold_with_amb,
        args=args,
    )

    work = cycle_results["w_hpr"].sum() - cycle_results["w_he"].sum()
    cop = (
        cycle_results["Qc"].sum() / cycle_results["w_hpr"].sum()
        if cycle_results["w_hpr"].sum() > tol
        else 0
    )
    eta_he = (
        cycle_results["w_he"].sum() / (cycle_results["Qc"].sum() + 1e-6)
        if cycle_results["Qc"].sum() != 0
        else 0
    )

    Q_ext_heat = max(np.abs(H_cold_with_amb[0]) - cycle_results["Qc"].sum(), 0.0)
    Q_ext_cold = max(np.abs(H_hot_with_amb[-1]) - cycle_results["Qe"].sum(), 0.0)
    p = g_ineq_penalty(g=cycle_results["q_diff"], rho=args.rho_penalty, form="square")

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
            title=f"Obj {float(obj):.5f} = {(work.sum() / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )

    return {
        "obj": obj,
        "utility_tot": work.sum() + Q_ext_heat,
        "w_net": work,
        "w_hpr": cycle_results["w_hpr"],
        "w_he": cycle_results["w_he"],
        "heat_recovery": cycle_results["heat_recovery"],
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "Q_cond": cycle_results["Qc"],
        "T_evap": state_vars["T_evap"],
        "Q_evap": cycle_results["Qe"],
        "cop_h": cop,
        "eta_he": eta_he,
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "success": True,
    }
