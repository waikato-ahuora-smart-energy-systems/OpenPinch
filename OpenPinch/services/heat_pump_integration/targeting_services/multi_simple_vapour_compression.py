"""Multiple simple vapour-compression HP targeting."""

from __future__ import annotations

import numpy as np

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
    evaluate_vapour_hpr_result,
    solve_hpr_placement,
    validate_vapour_hp_refrigerant_ls,
)
from ..unit_models.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from .multi_simple_carnot import optimise_multi_simple_carnot_heat_pump_placement

__all__ = [
    "optimise_multi_simple_heat_pump_placement",
]


################################################################################
# Public API
################################################################################


@timing_decorator
def optimise_multi_simple_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise multiple parallel vapour-compression stages for the HPR case."""
    num_stages = args.n_cond = args.n_evap = int(max(args.n_cond, args.n_evap))
    init_res = (
        optimise_multi_simple_carnot_heat_pump_placement(args)
        if args.initialise_simulated_cycle
        else None
    )
    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(num_stages, args)
    x0_ls, bnds = _get_multi_single_hp_opt_setup(init_res, args)
    return solve_hpr_placement(
        f_obj=_compute_multi_simple_hp_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


def _get_multi_single_hp_opt_setup(
    init_res: HPRBackendResult | None,
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
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

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


################################################################################
# Helper Functions
################################################################################


def _parse_multi_simple_hp_state_temperatures(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
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
    return HPRParsedState(
        T_cond=T_cond,
        dT_subcool=dT_subcool,
        Q_heat=Q_cond,
        T_evap=T_evap,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        dT_ihx_gas_side=dT_ihx_gas_side,
    )


def _compute_multi_simple_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    debug: bool = False,
) -> HPRBackendResult:
    state_vars = _parse_multi_simple_hp_state_temperatures(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)

    T_diff = (
        state_vars.T_cond - state_vars.dT_subcool - state_vars.T_evap - args.dtcont_hp
    )
    if T_diff.min() < 0:
        return HPRBackendResult.failure(
            reason="Invalid simulated vapour candidate with negative temperature lift.",
            Q_amb_hot=state_vars.Q_amb_hot,
            Q_amb_cold=state_vars.Q_amb_cold,
        )

    try:
        hp = ParallelVapourCompressionCycles()
        hp.solve(
            T_evap=state_vars.T_evap,
            T_cond=state_vars.T_cond,
            dT_subcool=state_vars.dT_subcool,
            eta_comp=args.eta_comp,
            refrigerant=args.refrigerant_ls,
            dT_ihx_gas_side=state_vars.dT_ihx_gas_side,
            Q_heat=state_vars.Q_heat,
        )
        if not hp.solved:
            return HPRBackendResult.failure(
                reason="Parallel vapour-compression cycle failed to solve.",
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
            dT_superheat=state_vars.dT_superheat,
            debug=debug,
        )
    except Exception as exc:
        return HPRBackendResult.failure(
            reason=str(exc),
            Q_amb_hot=state_vars.Q_amb_hot,
            Q_amb_cold=state_vars.Q_amb_cold,
        )
