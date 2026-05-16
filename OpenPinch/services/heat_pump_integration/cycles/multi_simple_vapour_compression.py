"""Multiple simple vapour-compression HP targeting."""

import numpy as np

from ....classes.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from ....lib.enums import PT
from ....lib.schemas.hpr import HeatPumpTargetInputs, HeatPumpTargetOutputs
from ....utils.decorators import timing_decorator
from ..common.encoding import (
    MAX_AMBIENT_X_ABS,
    map_Q_amb_to_x,
    map_Q_arr_to_x_arr,
    map_T_arr_to_x_arr,
    map_x_arr_to_DT_arr,
    map_x_arr_to_Q_arr,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
)
from ..common.layout import HPRoptVectorLayout
from ..common.shared import (
    calc_hpr_obj,
    g_ineq_penalty,
    get_ambient_air_stream,
    get_process_heat_cascade,
    plot_multi_hp_profiles_from_results,
    solve_hpr_placement,
    validate_vapour_hp_refrigerant_ls,
)
from .multi_simple_carnot import optimise_multi_simple_carnot_heat_pump_placement

__all__ = [
    "optimise_multi_simple_heat_pump_placement",
]


#######################################################################################################
# Public API
#######################################################################################################


@timing_decorator
def optimise_multi_simple_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HeatPumpTargetOutputs:
    """Optimise multiple parallel vapour-compression stages for the HPR case."""
    num_stages = args.n_cond = args.n_evap = int(max(args.n_cond, args.n_evap))
    init_res = (
        optimise_multi_simple_carnot_heat_pump_placement(args)
        if args.initialise_simulated_cycle
        else None
    )
    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(num_stages, args)
    x0_ls, bnds = _get_multi_single_hp_opt_setup(init_res, args)
    res = solve_hpr_placement(
        f_obj=_compute_multi_simple_hp_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )
    return HeatPumpTargetOutputs.model_validate(res)


def _get_multi_single_hp_opt_setup(
    init_res: HeatPumpTargetOutputs,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray | None, list]:
    n_units = int(args.n_cond)
    layout = HPRoptVectorLayout(
        n_cond=n_units,
        n_evap=int(args.n_evap),
        n_subcool=n_units,
        n_heat=n_units,
        n_ihx=n_units,
    )
    bnds = layout.build_bounds(
        x_amb=(-MAX_AMBIENT_X_ABS, MAX_AMBIENT_X_ABS),
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

    Q_cool_ex = args.Q_cool_max + init_res.Q_amb_hot
    Q_heat_ex = args.Q_heat_max + init_res.Q_amb_cold

    x_amb = map_Q_amb_to_x(
        init_res.Q_amb_hot,
        init_res.Q_amb_cold,
        max(args.Q_heat_max, args.Q_cool_max),
    )
    x_cond = map_T_arr_to_x_arr(
        init_res.T_cond, args.T_cold[0], args.T_cold[-1]
    ).tolist()
    x_evap = map_T_arr_to_x_arr(
        init_res.T_evap[::-1], args.T_hot[-1], args.T_hot[0]
    ).tolist()
    x_subcool = [0.0] * int(args.n_cond)
    x_heat = map_Q_arr_to_x_arr(init_res.Q_cond, Q_heat_ex).tolist()
    x_ihx = [0.0] * int(args.n_cond)
    return layout.pack(
        x_amb=x_amb,
        x_cond=x_cond,
        x_evap=x_evap,
        x_subcool=x_subcool,
        x_heat=x_heat,
        x_ihx=x_ihx,
    ), bnds


#######################################################################################################
# Helper Functions
#######################################################################################################


def _parse_multi_simple_hp_state_temperatures(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> dict:
    n_units = int(args.n_cond)
    parts = HPRoptVectorLayout(
        n_cond=n_units,
        n_evap=int(args.n_evap),
        n_subcool=n_units,
        n_heat=n_units,
        n_ihx=n_units,
    ).unpack(x)
    x_amb = parts["x_amb"]
    x_cond = parts["x_cond"]
    x_evap = parts["x_evap"]
    x_subcool = parts["x_subcool"]
    x_heat = parts["x_heat"]
    x_ihx = parts["x_ihx"]

    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    T_cond = map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = map_x_arr_to_DT_arr(x_subcool, T_cond, T_evap)
    Q_cond = map_x_arr_to_Q_arr(x_heat, args.Q_heat_max + Q_amb_cold)
    dT_ihx_gas_side = map_x_arr_to_DT_arr(x_ihx, T_cond, T_evap)
    return {
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": Q_cond,
        "T_evap": T_evap,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
        "dT_ihx_gas_side": dT_ihx_gas_side,
    }


def _compute_multi_simple_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    debug: bool = False,
) -> dict:
    state_vars = _parse_multi_simple_hp_state_temperatures(x, args)

    T_diff = (
        state_vars["T_cond"]
        - state_vars["dT_subcool"]
        - state_vars["T_evap"]
        - args.dtcont_hp
    )
    if T_diff.min() < 0:
        return {"obj": np.inf, "success": False}

    H_hot_with_amb = (
        args.H_hot + getattr(args, "z_amb_hot", 0.0) * state_vars["Q_amb_hot"]
    )
    H_cold_with_amb = (
        args.H_cold + getattr(args, "z_amb_cold", 0.0) * state_vars["Q_amb_cold"]
    )

    hp = ParallelVapourCompressionCycles()
    hp.solve(
        T_evap=state_vars["T_evap"],
        T_cond=state_vars["T_cond"],
        dT_subcool=state_vars["dT_subcool"],
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dT_ihx_gas_side=state_vars["dT_ihx_gas_side"],
        Q_heat=state_vars["Q_heat"],
    )

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
        + get_ambient_air_stream(Q_amb_cold=state_vars["Q_amb_cold"], args=args),
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams
        + get_ambient_air_stream(Q_amb_hot=state_vars["Q_amb_hot"], args=args),
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    w_hpr = hp.work
    cop = hp.Q_heat_arr.sum() / w_hpr if w_hpr > 0 else 1.0

    Q_ext_heat = max(float(pt_cond.col[PT.H_NET.value][0]), 0.0)
    Q_ext_cold = max(float(pt_evap.col[PT.H_NET.value][0]), 0.0)
    penalty_terms = np.atleast_1d(np.asarray(hp.penalty, dtype=float)).sum()
    g = np.maximum(
        np.array([pt_cond.col[PT.H_NET.value][-1], penalty_terms], dtype=float),
        0.0,
    )
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    if not args.is_heat_pumping:
        p += g_ineq_penalty(g=Q_ext_cold, rho=args.rho_penalty, form="square")

    obj = calc_hpr_obj(
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
        "dT_superheat": state_vars.get("dT_superheat"),
        "Q_cool": hp.Q_cool_arr,
        "cop_h": cop,
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "hpr_hot_streams": hpr_hot_streams,
        "hpr_cold_streams": hpr_cold_streams,
        "model": hp,
    }
