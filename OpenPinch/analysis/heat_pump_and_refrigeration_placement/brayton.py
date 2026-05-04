"""Brayton HP targeting."""

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from ...classes.stream_collection import StreamCollection
from ...classes.brayton_heat_pump import SimpleBraytonHeatPumpCycle
from ...lib.enums import PT
from ...lib.schema import HPRTargetInputs, HPRTargetOutputs
from ...utils.decorators import timing_decorator

from .shared import (
    calc_hpr_obj,
    get_process_heat_cascade,
)


__all__ = [
    "optimise_brayton_heat_pump_placement",
]


#######################################################################################################
# Public API
#######################################################################################################


@timing_decorator
def optimise_brayton_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    args.n_cond = args.n_evap = 1
    args.refrigerant_ls = ["air"]

    opt = minimize(
        fun=lambda x: _compute_brayton_hp_system_obj(x, args)["obj"],
        x0=_get_x0_for_brayton_hp_opt(args),
        method="SLSQP",
        bounds=_get_bounds_for_brayton_hp_opt(args),
        options={"disp": False, "maxiter": 1000},
        tol=1e-7,
    )

    if not opt.success:
        raise ValueError(f"Brayton heat pump targeting failed: {opt.message}")

    res = _compute_brayton_hp_system_obj(opt.x, args)
    res["success"] = opt.success
    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_brayton_hp_opt(args: HPRTargetInputs) -> list:
    return [
        0.0,
        abs(args.T_cold[0] - args.T_hot[0]) / args.dt_range_max,
        abs(args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max,
        1.0,
    ]


#######################################################################################################
# Helper Functions
#######################################################################################################


def _get_bounds_for_brayton_hp_opt(args: HPRTargetInputs) -> list:
    return [
        (-0.2, 1.0),
        (0.01, 1.5),
        (0.01, 1.5),
        (0.01, 1.0),
    ]


def _parse_brayton_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray]:
    T_comp_out = np.array(args.T_cold[0] + x[0] * args.dt_range_max, dtype=np.float64)
    dT_comp = np.array(x[1] * args.dt_range_max, dtype=np.float64)
    dT_gc = np.array(x[2] * args.dt_range_max, dtype=np.float64)
    Q_heat = np.array(x[3] * args.Q_hpr_target, dtype=np.float64)
    return [T_comp_out], [dT_comp], [dT_gc], [Q_heat]


def _create_brayton_hp_list(
    T_comp_out: np.ndarray,
    dT_gc: np.ndarray,
    Q_gc: np.ndarray,
    dT_comp: np.ndarray,
    args: HPRTargetInputs,
) -> List[SimpleBraytonHeatPumpCycle]:
    hp_list = []
    n_hp = args.n_cond
    for i in range(n_hp):
        hp = SimpleBraytonHeatPumpCycle()
        hp.solve(
            T_comp_out=T_comp_out[i],
            T_comp_in=T_comp_out[i] - dT_comp[i],
            dT_gc=dT_gc[i],
            Q_heat=Q_gc[i],
            eta_comp=args.eta_comp,
            eta_exp=args.eta_exp,
            is_recuperated=False,
            refrigerant=args.refrigerant_ls[0],
        )
        hp_list.append(hp)
    return hp_list


def _compute_brayton_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> float:
    T_comp_out, dT_comp, dT_gc, Q_heat = _parse_brayton_hp_state_variables(x, args)

    hp_list = _create_brayton_hp_list(
        T_comp_out=T_comp_out,
        dT_comp=dT_comp,
        dT_gc=dT_gc,
        Q_gc=Q_heat,
        args=args,
    )

    hpr_hot_streams = _build_simulated_hpr_streams(hp_list, include_cond=True)
    hpr_cold_streams = _build_simulated_hpr_streams(hp_list, include_evap=True)

    T_exp_out = hp_list[0].cycle_states[3]["T"]

    pt_gas_cooler = get_process_heat_cascade(
        hot_streams=hpr_hot_streams,
        cold_streams=args.bckgrd_cold_streams,
        is_shifted=True,
    )
    pt_gas_heater = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams,
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    w_hpr = sum([hp.work_net for hp in hp_list])
    c = (
        pt_gas_cooler.col[PT.H_NET.value][-1] + pt_gas_heater.col[PT.H_NET.value][0]
    ) * 10
    Q_ext = pt_gas_cooler.col[PT.H_NET.value][0]
    Q_cool = np.array([hp.Q_cool for hp in hp_list])
    cop = (args.Q_hpr_target - Q_ext) / (w_hpr + 1e-9)
    # Q_amb = _calc_Q_amb(Q_cool.sum(), np.abs(args.H_hot[-1]), args.Q_amb_max)
    obj = calc_hpr_obj(
        work=w_hpr,
        Q_ext_heat=Q_ext,
        Q_ext_cold=0.0,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=c,
    )

    return {
        "obj": obj,
        "utility_tot": w_hpr + Q_ext,
        "w_net": w_hpr,
        "Q_ext": Q_ext,
        "T_comp_out": np.array(T_comp_out),
        "dT_gc": np.array(dT_gc),
        "Q_heat": np.array(Q_heat),
        "T_evap": np.array(T_exp_out),
        "dT_comp": np.array(dT_comp),
        "Q_cool": np.array(Q_cool),
        "cop_h": cop,
        "Q_amb_hot": Q_amb if args.is_heat_pumping else 0.0,
        "Q_amb_cold": 0.0 if args.is_heat_pumping else Q_amb,
        "hpr_hot_streams": hpr_hot_streams,
        "hpr_cold_streams": hpr_cold_streams,
        "model": hp_list,
    }


def _build_simulated_hpr_streams(
    hp_list,
    *,
    is_process_stream: bool = False,
    include_cond: bool = False,
    include_evap: bool = False,
    dtcont_hp: float = 0.0,
) -> StreamCollection:
    hp_streams = StreamCollection()
    for hp in hp_list:
        hp_streams.add_many(
            hp.build_stream_collection(
                include_cond=include_cond,
                include_evap=include_evap,
                is_process_stream=is_process_stream,
                dtcont=dtcont_hp,
            )
        )
    return hp_streams


# def _calc_Q_amb(
#     Q_evap_total: float,
#     H_hot_limit: float,
#     Q_amb_max: float,
# ) -> float:
#     return max(Q_evap_total - (H_hot_limit - Q_amb_max), 0.0)
