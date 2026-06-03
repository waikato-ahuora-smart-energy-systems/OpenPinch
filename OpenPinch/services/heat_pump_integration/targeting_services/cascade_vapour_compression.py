"""Cascade vapour-compression HP targeting."""

import numpy as np

from ..unit_models.cascade_vapour_compression_cycle import CascadeVapourCompressionCycle
from ....lib.schemas.hpr import HeatPumpTargetInputs, HPRBackendResult, HPRParsedState
from ....utils.decorators import timing_decorator
from ..common.encoding import (
    AMBIENT_X_BOUNDS,
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
    _append_unspecified_final_cascade_cooling_duty,
    evaluate_vapour_hpr_result,
    solve_hpr_placement,
    validate_vapour_hp_refrigerant_ls,
)
from .multi_temperature_carnot import (
    optimise_multi_temperature_carnot_heat_pump_placement,
)

__all__ = [
    "optimise_cascade_heat_pump_placement",
]


################################################################################
# Public API
################################################################################


@timing_decorator
def optimise_cascade_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise a cascade vapour-compression placement for the prepared HPR case."""
    num_stages = int(args.n_cond + args.n_evap - 1)
    init_res = (
        optimise_multi_temperature_carnot_heat_pump_placement(args)
        if args.initialise_simulated_cycle
        else None
    )

    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(num_stages, args)
    x0_ls, bnds = _get_cascade_hp_opt_setup(init_res, args)
    return solve_hpr_placement(
        f_obj=_compute_cascade_hp_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


################################################################################
# Helper Functions
################################################################################


def _get_cascade_hp_opt_setup(
    init_res: HPRBackendResult | None,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray | None, list]:
    n_cond = int(args.n_cond)
    n_evap = int(args.n_evap)
    layout = HPRoptVectorLayout(
        n_cond=n_cond,
        n_evap=n_evap,
        n_subcool=n_cond,
        n_heat=n_cond,
        n_cool=max(n_evap - 1, 0),
        n_ihx=n_cond + n_evap - 1,
    )
    bnds = layout.build_bounds(
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat=(0.0, 1.0),
        x_cool=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

    n_cool = max(int(args.n_evap) - 1, 0)
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
    x_cool = map_Q_arr_to_x_arr(init_res.Q_evap[:n_cool], Q_cool_ex).tolist()
    x_ihx = [0.0] * (int(args.n_cond) + int(args.n_evap) - 1)
    return layout.pack(
        x_amb=x_amb,
        x_cond=x_cond,
        x_evap=x_evap,
        x_subcool=x_subcool,
        x_heat=x_heat,
        x_cool=x_cool,
        x_ihx=x_ihx,
    ), bnds


def _parse_cascade_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    n_cond = int(args.n_cond)
    n_evap = int(args.n_evap)
    parts = HPRoptVectorLayout(
        n_cond=n_cond,
        n_evap=n_evap,
        n_subcool=n_cond,
        n_heat=n_cond,
        n_cool=max(n_evap - 1, 0),
        n_ihx=n_cond + n_evap - 1,
    ).unpack(x)
    x_amb = parts["x_amb"]
    x_cond = parts["x_cond"]
    x_evap = parts["x_evap"]
    x_subcool = parts["x_subcool"]
    x_heat = parts["x_heat"]
    x_cool = parts["x_cool"]
    x_ihx = parts["x_ihx"]

    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    T_cond = map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = map_x_arr_to_DT_arr(x_subcool, T_cond, args.T_cold[0])
    Q_heat = map_x_arr_to_Q_arr(x_heat, args.Q_heat_max + Q_amb_cold)
    Q_cool = _append_unspecified_final_cascade_cooling_duty(
        x_cool * (args.Q_cool_max + Q_amb_hot)
    )
    T_cond_all = np.sort(np.concatenate([T_cond - dT_subcool, T_evap[:-1]]))[::-1]
    T_evap_all = np.sort(np.concatenate([(T_cond - dT_subcool)[1:], T_evap]))[::-1]
    dT_ihx_gas_side = map_x_arr_to_DT_arr(x_ihx, T_cond_all, T_evap_all)
    return HPRParsedState(
        T_cond=T_cond,
        dT_subcool=dT_subcool,
        Q_heat=Q_heat,
        T_evap=T_evap,
        Q_cool=Q_cool,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        dT_ihx_gas_side=dT_ihx_gas_side,
    )


def _compute_cascade_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> HPRBackendResult:
    state_vars = _parse_cascade_hp_state_variables(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)

    try:
        hp = CascadeVapourCompressionCycle()
        hp.solve(
            T_evap=state_vars.T_evap,
            T_cond=state_vars.T_cond,
            Q_heat=state_vars.Q_heat,
            Q_cool=state_vars.Q_cool,
            dT_subcool=state_vars.dT_subcool,
            dT_ihx_gas_side=state_vars.dT_ihx_gas_side,
            eta_comp=args.eta_comp,
            refrigerant=args.refrigerant_ls,
            dt_cascade_hx=args.dt_cascade_hx,
        )
        if not hp.solved:
            return HPRBackendResult.failure(
                reason="Cascade vapour-compression cycle failed to solve.",
                Q_amb_hot=state_vars.Q_amb_hot,
                Q_amb_cold=state_vars.Q_amb_cold,
            )

        hpr_streams = hp.build_stream_collection(
            include_cond=True,
            include_evap=True,
            is_process_stream=False,
            dtcont=args.dtcont_hp,
        )
        w_hpr = hp.work
        cop = hp.Q_heat_arr.sum() / w_hpr if w_hpr > 0 else 1.0
        return evaluate_vapour_hpr_result(
            args=args,
            state=state_vars,
            work=w_hpr,
            work_arr=hp.work_arr,
            Q_heat=hp.Q_heat_arr,
            Q_cool=hp.Q_cool_arr,
            cop_h=cop,
            hpr_streams=hpr_streams,
            model=hp,
            penalty_terms=[hp.penalty],
            dT_subcool=state_vars.dT_subcool,
            debug=debug,
        )
    except Exception as exc:
        return HPRBackendResult.failure(
            reason=str(exc),
            Q_amb_hot=state_vars.Q_amb_hot,
            Q_amb_cold=state_vars.Q_amb_cold,
        )
