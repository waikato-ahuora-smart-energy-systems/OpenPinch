"""Cascade vapour-compression HP targeting."""

from __future__ import annotations

import numpy as np

from ....contracts.hpr import HeatPumpTargetInputs, HPRBackendResult, HPRParsedState
from ..common._shared.ambient_preallocation import preallocate_direct_ambient_duties
from ..common._shared.streams import get_Q_vals_at_T_hpr_from_bckgrd_profile
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
    validate_vapour_hp_refrigerant_ls,
)
from ..cycles.cascade_vapour_compression_cycle import CascadeVapourCompressionCycle
from ..optimisation_adapter import solve_hpr_placement
from .cascade_carnot import (
    optimise_cascade_carnot_heat_pump_placement,
)

__all__ = [
    "optimise_cascade_heat_pump_placement",
]


################################################################################
# Public API
################################################################################


def optimise_cascade_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise a cascade vapour-compression placement for the prepared HPR case."""
    num_stages = int(args.n_cond + args.n_evap - 1)
    init_res = (
        optimise_cascade_carnot_heat_pump_placement(args)
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
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_cond = int(args.n_cond)
    n_evap = int(args.n_evap)
    n_heat, n_cool = _cascade_process_control_counts(
        n_cond=n_cond,
        n_evap=n_evap,
        is_heat_pumping=is_heat_pumping,
    )
    layout = HPRoptVectorLayout(
        n_cond=n_cond,
        n_evap=n_evap,
        n_subcool=n_cond,
        n_heat_base=1 if n_heat else 0,
        n_cool_base=1 if n_cool else 0,
        n_heat_split=n_heat,
        n_cool_split=n_cool,
        n_ihx=n_cond + n_evap - 1,
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

    ambient = preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=init_res.Q_amb_hot,
        Q_amb_cold=init_res.Q_amb_cold,
    )
    Q_cool_ex = ambient.Q_cool_capacity
    Q_heat_ex = ambient.Q_heat_capacity

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
    init_heat = init_res.Q_cond if is_heat_pumping else init_res.Q_cond[:n_heat]
    _, x_heat_base, x_heat_split = encode_base_and_duty_splits(init_heat, Q_heat_ex)
    x_heat_split = x_heat_split.tolist()
    init_cool = init_res.Q_evap[:n_cool] if is_heat_pumping else init_res.Q_evap
    _, x_cool_base, x_cool_split = encode_base_and_duty_splits(
        init_cool,
        Q_cool_ex,
    )
    x_cool_split = x_cool_split.tolist()
    x_ihx = [0.0] * (int(args.n_cond) + int(args.n_evap) - 1)
    pack_kwargs = {
        "x_amb": x_amb,
        "x_cond": x_cond,
        "x_evap": x_evap,
        "x_subcool": x_subcool,
        "x_ihx": x_ihx,
    }
    if n_heat:
        pack_kwargs["x_heat_base"] = [x_heat_base]
        pack_kwargs["x_heat_split"] = x_heat_split
    if n_cool:
        pack_kwargs["x_cool_base"] = [x_cool_base]
        pack_kwargs["x_cool_split"] = x_cool_split
    return layout.pack(**pack_kwargs), bnds


def _parse_cascade_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_cond = int(args.n_cond)
    n_evap = int(args.n_evap)
    n_heat, n_cool = _cascade_process_control_counts(
        n_cond=n_cond,
        n_evap=n_evap,
        is_heat_pumping=is_heat_pumping,
    )
    parts = HPRoptVectorLayout(
        n_cond=n_cond,
        n_evap=n_evap,
        n_subcool=n_cond,
        n_heat_base=1 if n_heat else 0,
        n_cool_base=1 if n_cool else 0,
        n_heat_split=n_heat,
        n_cool_split=n_cool,
        n_ihx=n_cond + n_evap - 1,
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
    ambient = preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
    )
    H_cold_with_amb = ambient.H_cold_with_residual_ambient(args)
    H_hot_with_amb = ambient.H_hot_with_residual_ambient(args)
    T_cond = map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = map_x_arr_to_DT_arr(x_subcool, T_cond, args.T_cold[0])
    Q_heat_base = float(x_heat_base[0]) * ambient.Q_heat_capacity if n_heat else None
    Q_cool_base = float(x_cool_base[0]) * ambient.Q_cool_capacity if n_cool else None
    Q_heat_available = (
        get_Q_vals_at_T_hpr_from_bckgrd_profile(
            T_cond if is_heat_pumping else T_cond[:n_heat],
            ambient.T_cold_residual,
            H_cold_with_amb,
            is_cond=True,
        )
        if n_heat
        else None
    )
    Q_cool_available = (
        get_Q_vals_at_T_hpr_from_bckgrd_profile(
            T_evap[:n_cool] if is_heat_pumping else T_evap,
            ambient.T_hot_residual,
            H_hot_with_amb,
            is_cond=False,
        )
        if n_cool
        else None
    )
    T_cond_all = np.sort(np.concatenate([T_cond - dT_subcool, T_evap[:-1]]))[::-1]
    T_evap_all = np.sort(np.concatenate([(T_cond - dT_subcool)[1:], T_evap]))[::-1]
    dT_ihx_gas_side = map_x_arr_to_DT_arr(x_ihx, T_cond_all, T_evap_all)
    return HPRParsedState(
        T_cond=T_cond,
        dT_subcool=dT_subcool,
        T_evap=T_evap,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        Q_amb_hot_direct=ambient.Q_amb_hot_direct,
        Q_amb_cold_direct=ambient.Q_amb_cold_direct,
        Q_amb_hot_residual=ambient.Q_amb_hot_residual,
        Q_amb_cold_residual=ambient.Q_amb_cold_residual,
        dT_ihx_gas_side=dT_ihx_gas_side,
        Q_heat_base=Q_heat_base,
        Q_cool_base=Q_cool_base,
        x_heat_split=x_heat_split if n_heat else None,
        x_cool_split=x_cool_split if n_cool else None,
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
    )


def _cascade_process_control_counts(
    *,
    n_cond: int,
    n_evap: int,
    is_heat_pumping: bool,
) -> tuple[int, int]:
    if is_heat_pumping:
        return n_cond, max(n_evap - 1, 0)
    return max(n_cond - 1, 0), n_evap


def _compute_cascade_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> HPRBackendResult:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    state_vars = _parse_cascade_hp_state_variables(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)

    try:
        hp = CascadeVapourCompressionCycle()
        hp.solve(
            T_evap=state_vars.T_evap,
            T_cond=state_vars.T_cond,
            dtcont=args.dtcont_hp,
            Q_heat_base=state_vars.Q_heat_base,
            x_heat_split=state_vars.x_heat_split,
            Q_heat_available=state_vars.Q_heat_available,
            Q_cool_base=state_vars.Q_cool_base,
            x_cool_split=state_vars.x_cool_split,
            Q_cool_available=state_vars.Q_cool_available,
            dT_subcool=state_vars.dT_subcool,
            dT_ihx_gas_side=state_vars.dT_ihx_gas_side,
            eta_comp=args.eta_comp,
            refrigerant=args.refrigerant_ls,
            dt_cascade_hx=args.dt_cascade_hx,
            is_heat_pump=is_heat_pumping,
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
            debug=debug,
        )
    except Exception as exc:
        return HPRBackendResult.failure(
            reason=str(exc),
            Q_amb_hot=state_vars.Q_amb_hot,
            Q_amb_cold=state_vars.Q_amb_cold,
        )
