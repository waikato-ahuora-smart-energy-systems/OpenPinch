"""Parallel Carnot HP targeting."""

from __future__ import annotations

import numpy as np

from ....lib.schemas.hpr import HeatPumpTargetInputs, HPRBackendResult, HPRParsedState
from ....utils.decorators import timing_decorator
from ..common.encoding import (
    AMBIENT_X_BOUNDS,
    encode_duty_splits,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
)
from ..common.layout import HPRoptVectorLayout
from ..common.shared import (
    evaluate_carnot_hpr_result,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
    solve_hpr_placement,
)
from ..unit_models.carnot_cycles import ParallelCarnotCycles

__all__ = [
    "optimise_parallel_carnot_heat_pump_placement",
]


################################################################################
# Public API
################################################################################


@timing_decorator
def optimise_parallel_carnot_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise parallel simple Carnot stages for a screening-level HPR solve."""
    args.n_cond = args.n_evap = max(args.n_cond, args.n_evap)
    x0_ls, bnds = _get_parallel_carnot_hp_opt_setup(args)
    return solve_hpr_placement(
        f_obj=_compute_parallel_carnot_hp_opt_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


################################################################################
# Helper Functions
################################################################################


def _get_parallel_carnot_hp_opt_setup(
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, list]:
    n_stages = int(args.n_cond)
    layout = HPRoptVectorLayout(
        n_cond=args.n_cond,
        n_evap=args.n_evap,
        n_heat_base=1,
        n_heat_split=n_stages,
    )
    x_heat_split = encode_duty_splits(
        np.full(n_stages, 1.0 / max(n_stages, 1)),
        1.0,
    )
    return layout.pack(
        x_amb=0.0,
        x_cond=[0.0] * layout.n_cond,
        x_evap=[0.0] * layout.n_evap,
        x_heat_base=[1.0],
        x_heat_split=x_heat_split,
    ), layout.build_bounds(
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_heat_base=(0.0, 1.0),
        x_heat_split=(0.0, 1.0),
    )


def _parse_parallel_carnot_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    parts = HPRoptVectorLayout(
        n_cond=args.n_cond,
        n_evap=args.n_evap,
        n_heat_base=1,
        n_heat_split=int(args.n_cond),
    ).unpack(x)
    T_cond = map_x_arr_to_T_arr(parts["x_cond"], args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(parts["x_evap"], args.T_hot[-1], args.T_hot[0])
    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(
        parts["x_amb"], max(args.Q_heat_max, args.Q_cool_max)
    )
    H_cold_with_amb = args.H_cold + args.z_amb_cold * Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * Q_amb_hot
    Q_heat_base = float(parts["x_heat_base"][0]) * (args.Q_heat_max + Q_amb_cold)
    Q_heat_available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond,
        args.T_cold,
        H_cold_with_amb,
        is_cond=True,
    )
    Q_cool_available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap,
        args.T_hot,
        H_hot_with_amb,
        is_cond=False,
    )
    return HPRParsedState(
        T_cond=T_cond,
        T_evap=T_evap,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        Q_heat_base=Q_heat_base,
        x_heat_split=parts["x_heat_split"],
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
    )


def _compute_parallel_carnot_hp_opt_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> HPRBackendResult:
    state_vars = _parse_parallel_carnot_hp_state_variables(x, args)
    cycle = ParallelCarnotCycles()
    cycle.solve(
        T_cond=state_vars.T_cond,
        T_evap=state_vars.T_evap,
        Q_heat_base=state_vars.Q_heat_base,
        x_heat_split=state_vars.x_heat_split,
        Q_heat_available=state_vars.Q_heat_available,
        Q_cool_available=state_vars.Q_cool_available,
        eta_ii_hpr_carnot=args.eta_ii_hpr_carnot,
        eta_ii_he_carnot=args.eta_ii_he_carnot,
        args=args,
    )
    return evaluate_carnot_hpr_result(
        args=args,
        state=state_vars,
        w_net=cycle.work,
        w_hpr=cycle.w_hpr,
        w_he=cycle.w_he,
        heat_recovery=cycle.heat_recovery,
        cop_h=cycle.COP_h,
        eta_he=cycle.eta_he,
        Q_cond_total=cycle.Q_cond,
        Q_evap_total=cycle.Q_evap,
        penalty_terms=cycle.penalty,
        debug=debug,
    )
