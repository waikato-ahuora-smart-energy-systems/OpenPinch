"""Cascade vapour-compression HP targeting."""

import numpy as np

from ...classes.cascade_vapour_compression_cycle import CascadeVapourCompressionCycle
from ...lib.enums import PT
from ...lib.schema import HPRTargetInputs, HPRTargetOutputs
from ...utils.decorators import timing_decorator

from .encoding import (
    _map_Q_amb_to_x,
    _map_Q_arr_to_x_arr,
    _map_T_arr_to_x_arr,
    _map_x_arr_to_DT_arr,
    _map_x_arr_to_Q_arr,
    _map_x_arr_to_T_arr,
    _map_x_to_Q_amb,
)
from .multi_temperature_carnot import (
    _optimise_multi_temperature_carnot_heat_pump_placement,
)
from .shared import (
    _append_unspecified_final_cascade_cooling_duty,
    _calc_obj,
    _get_ambient_air_stream,
    get_process_heat_cascade,
    g_ineq_penalty,
    plot_multi_hp_profiles_from_results,
)


__all__ = [
    "_optimise_cascade_heat_pump_placement",
    "_get_x0_for_cascade_hp_opt",
    "_get_bounds_for_cascade_hp_opt",
    "_parse_cascade_hp_state_variables",
    "_compute_cascade_hp_system_obj",
]


@timing_decorator
def _optimise_cascade_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    num_stages = int(args.n_cond + args.n_evap - 1)
    init_res = (
        _optimise_multi_temperature_carnot_heat_pump_placement(args)
        if args.initialise_simulated_cycle
        else None
    )
    from .shared import _validate_vapour_hp_refrigerant_ls, _solve_hpr_placement

    args.refrigerant_ls = _validate_vapour_hp_refrigerant_ls(num_stages, args)
    res = _solve_hpr_placement(
        f_obj=_compute_cascade_hp_system_obj,
        x0_ls=_get_x0_for_cascade_hp_opt(init_res, args),
        bnds=_get_bounds_for_cascade_hp_opt(args),
        args=args,
    )
    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_cascade_hp_opt(
    init_res: HPRTargetOutputs,
    args: HPRTargetInputs,
) -> np.ndarray:
    if init_res is None:
        return None

    n_cool = max(int(args.n_evap) - 1, 0)
    Q_cool_ex = args.Q_cool_max + init_res.Q_amb_hot
    Q_heat_ex = args.Q_heat_max + init_res.Q_amb_cold

    x_amb = _map_Q_amb_to_x(
        init_res.Q_amb_hot, init_res.Q_amb_cold, max(Q_heat_ex, Q_cool_ex)
    )
    x_cond = _map_T_arr_to_x_arr(
        init_res.T_cond, args.T_cold[0], args.T_cold[-1]
    ).tolist()
    x_evap = _map_T_arr_to_x_arr(
        init_res.T_evap, args.T_hot[-1], args.T_hot[0]
    ).tolist()
    x_subcool = [0.0] * int(args.n_cond)
    x_heat = _map_Q_arr_to_x_arr(init_res.Q_cond, Q_heat_ex).tolist()
    x_cool = _map_Q_arr_to_x_arr(init_res.Q_evap[:n_cool], Q_cool_ex).tolist()
    x_ihx = [0.0] * (int(args.n_cond) + int(args.n_evap) - 1)
    return np.asarray(
        [x_amb] + x_cond + x_evap + x_subcool + x_heat + x_cool + x_ihx,
        dtype=np.float64,
    )


def _get_bounds_for_cascade_hp_opt(args: HPRTargetInputs) -> list:
    n_cond = n_heat = int(args.n_cond)
    n_evap = int(args.n_evap)
    n_cool = n_evap - 1
    n_units = n_cond + n_evap - 1
    return (
        [(-1.0, 10.0)]
        + [(0.0, 1.0)] * n_cond
        + [(0.0, 1.0)] * n_evap
        + [(0.0, 1.0)] * n_cond
        + [(0.0, 1.0)] * n_heat
        + [(0.0, 1.0)] * n_cool
        + [(0.0, 1.0)] * n_units
    )


def _parse_cascade_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    x_amb = x[0]
    a = 1 + int(args.n_cond)
    x_cond = x[1:a]
    b = a + int(args.n_evap)
    x_evap = x[a:b]
    c = b + int(args.n_cond)
    x_subcool = x[b:c]
    d = c + int(args.n_cond)
    x_heat = x[c:d]
    e = d + int(args.n_evap) - 1
    x_cool = x[d:e]
    f = e + (int(args.n_cond) + int(args.n_evap) - 1)
    x_ihx = x[e:f]

    Q_amb_hot, Q_amb_cold = _map_x_to_Q_amb(
        x_amb, max(args.Q_heat_max, args.Q_cool_max)
    )
    T_cond = _map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = _map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = _map_x_arr_to_DT_arr(x_subcool, T_cond, args.T_cold[0])
    Q_heat = _map_x_arr_to_Q_arr(x_heat, args.Q_heat_max + Q_amb_hot)
    Q_cool = _append_unspecified_final_cascade_cooling_duty(
        x_cool * (args.Q_cool_max + Q_amb_cold)
    )
    T_cond_all = np.sort(np.concatenate([T_cond - dT_subcool, T_evap[:-1]]))[::-1]
    T_evap_all = np.sort(np.concatenate([(T_cond - dT_subcool)[1:], T_evap]))[::-1]
    dT_ihx_gas_side = _map_x_arr_to_DT_arr(x_ihx, T_cond_all, T_evap_all)
    return {
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": Q_heat,
        "T_evap": T_evap,
        "Q_cool": Q_cool,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
        "dT_ihx_gas_side": dT_ihx_gas_side,
    }


def _compute_cascade_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    state_vars = _parse_cascade_hp_state_variables(x, args)

    hp = CascadeVapourCompressionCycle()
    hp.solve(
        T_evap=state_vars["T_evap"],
        T_cond=state_vars["T_cond"],
        Q_heat=state_vars["Q_heat"],
        Q_cool=state_vars["Q_cool"],
        dT_subcool=state_vars["dT_subcool"],
        dT_ihx_gas_side=state_vars["dT_ihx_gas_side"],
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dt_cascade_hx=args.dt_cascade_hx,
    )
    if not hp.solved:
        return {"obj": np.inf, "success": False}

    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars["Q_amb_hot"]
    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars["Q_amb_cold"]

    hpr_hot_streams = hp.build_stream_collection(
        include_cond=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )
    hpr_cold_streams = hp.build_stream_collection(
        include_evap=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )

    pt_cond = get_process_heat_cascade(
        hot_streams=hpr_hot_streams,
        cold_streams=args.bckgrd_cold_streams
        + _get_ambient_air_stream(Q_amb_cold=state_vars["Q_amb_cold"], args=args),
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams
        + _get_ambient_air_stream(Q_amb_hot=state_vars["Q_amb_hot"], args=args),
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    w_hpr = hp.work
    cop = hp.Q_heat_arr.sum() / w_hpr if w_hpr > 0 else 1.0

    Q_ext_heat = pt_cond.col[PT.H_NET.value][0]
    Q_ext_cold = pt_evap.col[PT.H_NET.value][-1]
    g = [pt_cond.col[PT.H_NET.value][-1], pt_evap.col[PT.H_NET.value][0], hp.penalty]
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")

    obj = _calc_obj(
        work=w_hpr,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            hpr_hot_streams,
            hpr_cold_streams,
            title=f"Obj {float(obj):.5f} = {(w_hpr / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )

    return {
        "obj": obj,
        "utility_tot": w_hpr + Q_ext_heat + Q_ext_cold,
        "w_net": w_hpr,
        "w_hpr": hp.work_arr,
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "dT_subcool": state_vars["dT_subcool"],
        "Q_heat": hp.Q_heat_arr,
        "T_evap": state_vars["T_evap"],
        "Q_cool": hp.Q_cool_arr,
        "cop_h": cop,
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "hpr_hot_streams": hpr_hot_streams,
        "hpr_cold_streams": hpr_cold_streams,
        "model": hp,
    }
