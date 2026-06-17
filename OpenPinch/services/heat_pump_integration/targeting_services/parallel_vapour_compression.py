"""Parallel vapour-compression HP targeting."""

from __future__ import annotations

import numpy as np

from ....lib.schemas.hpr import HeatPumpTargetInputs, HPRBackendResult, HPRParsedState
from ....utils.decorators import timing_decorator
from ..common.encoding import (
    AMBIENT_X_BOUNDS,
    encode_base_and_duty_splits,
    map_Q_amb_to_x,
    map_T_arr_to_x_arr,
    map_x_arr_to_DT_arr,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
)
from ..common.layout import HPRoptVectorLayout
from ..common.shared import (
    evaluate_vapour_hpr_result,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
    solve_hpr_placement,
    validate_vapour_hp_refrigerant_ls,
)
from ..unit_models.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from .parallel_carnot import optimise_parallel_carnot_heat_pump_placement

__all__ = [
    "optimise_parallel_heat_pump_placement",
]


################################################################################
# Public API
################################################################################


@timing_decorator
def optimise_parallel_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise multiple parallel vapour-compression stages for the HPR case."""
    num_stages = args.n_cond = args.n_evap = int(max(args.n_cond, args.n_evap))
    init_res = (
        optimise_parallel_carnot_heat_pump_placement(args)
        if args.initialise_simulated_cycle
        else None
    )
    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(num_stages, args)
    x0_ls, bnds = _get_parallel_hp_opt_setup(init_res, args)
    return solve_hpr_placement(
        f_obj=_compute_parallel_hp_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


def _get_parallel_hp_opt_setup(
    init_res: HPRBackendResult | None,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray | None, list]:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_units = int(args.n_cond)
    layout = HPRoptVectorLayout(
        n_cond=n_units,
        n_evap=int(args.n_evap),
        n_subcool=n_units,
        n_heat_base=1 if is_heat_pumping else 0,
        n_cool_base=0 if is_heat_pumping else 1,
        n_heat_split=n_units if is_heat_pumping else 0,
        n_cool_split=0 if is_heat_pumping else n_units,
        n_ihx=n_units,
    )
    bnds = layout.build_bounds(
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat_base=(0.0, 1.0),
        x_cool_base=(0.0, 1.0),
        x_heat_split=(0.0, 1.0),
        x_cool_split=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

    Q_primary_ex = (
        args.Q_heat_max + init_res.Q_amb_cold
        if is_heat_pumping
        else args.Q_cool_max + init_res.Q_amb_hot
    )

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
    init_primary_duty = init_res.Q_cond if is_heat_pumping else init_res.Q_evap
    _, x_primary_base, x_primary_split = encode_base_and_duty_splits(
        init_primary_duty,
        Q_primary_ex,
    )
    x_primary_split = x_primary_split.tolist()
    x_ihx = [0.0] * int(args.n_cond)
    pack_kwargs = {
        "x_amb": x_amb,
        "x_cond": x_cond,
        "x_evap": x_evap,
        "x_subcool": x_subcool,
        "x_ihx": x_ihx,
    }
    if is_heat_pumping:
        pack_kwargs["x_heat_base"] = [x_primary_base]
        pack_kwargs["x_heat_split"] = x_primary_split
    else:
        pack_kwargs["x_cool_base"] = [x_primary_base]
        pack_kwargs["x_cool_split"] = x_primary_split
    return layout.pack(**pack_kwargs), bnds


################################################################################
# Helper Functions
################################################################################


def _parse_parallel_hp_state_temperatures(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_units = int(args.n_cond)
    parts = HPRoptVectorLayout(
        n_cond=n_units,
        n_evap=int(args.n_evap),
        n_subcool=n_units,
        n_heat_base=1 if is_heat_pumping else 0,
        n_cool_base=0 if is_heat_pumping else 1,
        n_heat_split=n_units if is_heat_pumping else 0,
        n_cool_split=0 if is_heat_pumping else n_units,
        n_ihx=n_units,
    ).unpack(x)
    x_amb = parts["x_amb"]
    x_cond = parts["x_cond"]
    x_evap = parts["x_evap"]
    x_subcool = parts["x_subcool"]
    x_heat_base = parts["x_heat_base"]
    x_cool_base = parts["x_cool_base"]
    x_heat_split = parts["x_heat_split"]
    x_cool_split = parts["x_cool_split"]
    x_ihx = parts["x_ihx"]

    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    H_cold_with_amb = args.H_cold + args.z_amb_cold * Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * Q_amb_hot
    T_cond = map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = map_x_arr_to_DT_arr(x_subcool, T_cond, T_evap)
    Q_heat_base = (
        float(x_heat_base[0]) * (args.Q_heat_max + Q_amb_cold)
        if is_heat_pumping
        else None
    )
    Q_cool_base = (
        None
        if is_heat_pumping
        else float(x_cool_base[0]) * (args.Q_cool_max + Q_amb_hot)
    )
    Q_heat_available = (
        get_Q_vals_at_T_hpr_from_bckgrd_profile(
            T_cond,
            args.T_cold,
            H_cold_with_amb,
            is_cond=True,
        )
        if is_heat_pumping
        else None
    )
    Q_cool_available = (
        None
        if is_heat_pumping
        else get_Q_vals_at_T_hpr_from_bckgrd_profile(
            T_evap,
            args.T_hot,
            H_hot_with_amb,
            is_cond=False,
        )
    )
    dT_ihx_gas_side = map_x_arr_to_DT_arr(x_ihx, T_cond, T_evap)
    return HPRParsedState(
        T_cond=T_cond,
        dT_subcool=dT_subcool,
        T_evap=T_evap,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        dT_ihx_gas_side=dT_ihx_gas_side,
        Q_heat_base=Q_heat_base,
        Q_cool_base=Q_cool_base,
        x_heat_split=x_heat_split if is_heat_pumping else None,
        x_cool_split=None if is_heat_pumping else x_cool_split,
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
    )


def _compute_parallel_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    debug: bool = False,
) -> HPRBackendResult:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    state_vars = _parse_parallel_hp_state_temperatures(x, args)
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
            dtcont=args.dtcont_hp,
            dT_subcool=state_vars.dT_subcool,
            eta_comp=args.eta_comp,
            refrigerant=args.refrigerant_ls,
            dT_ihx_gas_side=state_vars.dT_ihx_gas_side,
            Q_heat_base=state_vars.Q_heat_base,
            x_heat_split=state_vars.x_heat_split,
            Q_heat_available=state_vars.Q_heat_available,
            Q_cool_base=state_vars.Q_cool_base,
            x_cool_split=state_vars.x_cool_split,
            Q_cool_available=state_vars.Q_cool_available,
            is_heat_pump=is_heat_pumping,
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
        primary_duty = hp.Q_heat_arr.sum() if is_heat_pumping else hp.Q_cool_arr.sum()
        cop = primary_duty / w_hpr if w_hpr > 0 else 1.0
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
