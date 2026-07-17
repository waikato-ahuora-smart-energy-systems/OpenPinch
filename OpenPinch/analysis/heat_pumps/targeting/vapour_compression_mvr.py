"""Vapour-compression plus MVR cascade HP targeting."""

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
from ..cycles.vapour_compression_mvr_cascade import VapourCompressionMvrCascade
from ..optimisation_adapter import solve_hpr_placement
from .cascade_carnot import (
    optimise_cascade_carnot_heat_pump_placement,
)

__all__ = [
    "optimise_vapour_compression_mvr_heat_pump_placement",
]

MAX_MVR_STAGE_LIFT = 20.0


def optimise_vapour_compression_mvr_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise a VC low-stage plus MVR high-stage placement."""
    if not getattr(args, "is_heat_pumping", True):
        raise ValueError(
            "Vapour compression with MVR cascade is heat-pump-only; "
            "refrigeration targets are not supported."
        )

    n_vc = _num_vc_stages(args)
    init_res = (
        optimise_cascade_carnot_heat_pump_placement(args)
        if args.initialise_simulated_cycle
        else None
    )
    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(n_vc, args)
    args.mvr_fluid_ls = _normalise_fluid_list(getattr(args, "mvr_fluid_ls", ["Water"]))
    x0_ls, bnds = _get_vc_mvr_opt_setup(init_res, args)
    return solve_hpr_placement(
        f_obj=_compute_vc_mvr_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


def _num_vc_stages(args: HeatPumpTargetInputs) -> int:
    return int(max(args.n_cond, args.n_evap))


def _num_mvr_stages(args: HeatPumpTargetInputs) -> int:
    n_mvr = int(getattr(args, "n_mvr", 1))
    if n_mvr < 1:
        raise ValueError("VC+MVR targeting requires at least one MVR stage.")
    return n_mvr


def _vc_mvr_layout(args: HeatPumpTargetInputs) -> HPRoptVectorLayout:
    n_vc = _num_vc_stages(args)
    n_mvr = _num_mvr_stages(args)
    return HPRoptVectorLayout(
        n_cond=n_vc,
        n_evap=n_vc,
        n_subcool=n_mvr + n_vc,
        n_heat_base=1,
        n_heat_split=n_vc,
        n_ihx=n_vc,
        n_misc=1 + n_mvr + max(n_mvr - 1, 0),
    )


def _get_vc_mvr_opt_setup(
    init_res: HPRBackendResult | None,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray | None, list]:
    layout = _vc_mvr_layout(args)
    bnds = layout.build_bounds(
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat_base=(0.0, 1.0),
        x_heat_split=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
        x_misc=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

    n_vc = _num_vc_stages(args)
    n_mvr = _num_mvr_stages(args)
    ambient = preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=init_res.Q_amb_hot,
        Q_amb_cold=init_res.Q_amb_cold,
    )
    Q_primary_ex = ambient.Q_heat_capacity
    x_amb = map_Q_amb_to_x(
        init_res.Q_amb_hot,
        init_res.Q_amb_cold,
        max(args.Q_heat_max, args.Q_cool_max),
    )
    T_cond_seed = _fit_stage_array(init_res.T_cond, n_vc)
    T_evap_seed = _fit_stage_array(init_res.T_evap[::-1], n_vc)
    Q_heat_seed = _fit_stage_array(init_res.Q_cond, n_vc)
    x_cond = map_T_arr_to_x_arr(T_cond_seed, args.T_cold[0], args.T_cold[-1])
    x_evap = map_T_arr_to_x_arr(T_evap_seed, args.T_hot[-1], args.T_hot[0])
    x_subcool = np.zeros(n_mvr + n_vc, dtype=float)
    _, x_heat_base, x_heat_split = encode_base_and_duty_splits(
        Q_heat_seed,
        Q_primary_ex,
    )
    x_ihx = np.zeros(n_vc, dtype=float)
    x_mvr_source_split = np.array([0.5], dtype=float)
    x_mvr_lift = np.full(n_mvr, 0.5, dtype=float)
    x_mvr_process_split = np.zeros(max(n_mvr - 1, 0), dtype=float)
    x_misc = np.concatenate([x_mvr_source_split, x_mvr_lift, x_mvr_process_split])
    return layout.pack(
        x_amb=x_amb,
        x_cond=x_cond,
        x_evap=x_evap,
        x_subcool=x_subcool,
        x_heat_base=[x_heat_base],
        x_heat_split=x_heat_split,
        x_ihx=x_ihx,
        x_misc=x_misc,
    ), bnds


def _parse_vc_mvr_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    n_mvr = _num_mvr_stages(args)
    parts = _vc_mvr_layout(args).unpack(x)
    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(
        parts["x_amb"],
        max(args.Q_heat_max, args.Q_cool_max),
    )
    T_cond_vc = map_x_arr_to_T_arr(
        parts["x_cond"],
        args.T_cold[0],
        args.T_cold[-1],
    )
    T_evap_vc = map_x_arr_to_T_arr(
        parts["x_evap"],
        args.T_hot[-1],
        args.T_hot[0],
    )
    ambient = preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
    )
    H_cold_with_amb = ambient.H_cold_with_residual_ambient(args)
    Q_heat_base = float(parts["x_heat_base"][0]) * ambient.Q_heat_capacity
    Q_heat_available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond_vc,
        ambient.T_cold_residual,
        H_cold_with_amb,
        is_cond=True,
    )
    x_subcool_mvr = parts["x_subcool"][:n_mvr]
    x_subcool_vc = parts["x_subcool"][n_mvr:]
    dT_subcool_vc = map_x_arr_to_DT_arr(x_subcool_vc, T_cond_vc, T_evap_vc)
    dT_ihx_gas_side = map_x_arr_to_DT_arr(parts["x_ihx"], T_cond_vc, T_evap_vc)
    misc = parts["x_misc"]
    x_mvr_source_split = misc[0]
    dT_lift_mvr = misc[1 : 1 + n_mvr] * MAX_MVR_STAGE_LIFT
    x_mvr_process_split = misc[1 + n_mvr :]
    T_evap_mvr, T_cond_mvr = _derive_provisional_mvr_temperatures(
        T_cond_vc=T_cond_vc,
        dT_subcool_vc=dT_subcool_vc,
        dT_lift_mvr=dT_lift_mvr,
        dt_cascade_hx=args.dt_cascade_hx,
    )
    dT_subcool_mvr = x_subcool_mvr * dT_lift_mvr
    return HPRParsedState(
        T_cond=np.concatenate([T_cond_mvr, T_cond_vc]),
        dT_subcool=np.concatenate([dT_subcool_mvr, dT_subcool_vc]),
        T_evap=np.concatenate([T_evap_mvr, T_evap_vc]),
        Q_cool=None,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        Q_amb_hot_direct=ambient.Q_amb_hot_direct,
        Q_amb_cold_direct=ambient.Q_amb_cold_direct,
        Q_amb_hot_residual=ambient.Q_amb_hot_residual,
        Q_amb_cold_residual=ambient.Q_amb_cold_residual,
        dT_ihx_gas_side=dT_ihx_gas_side,
        Q_heat_base=Q_heat_base,
        x_heat_split=parts["x_heat_split"],
        Q_heat_available=Q_heat_available,
        x_mvr_source_split=float(x_mvr_source_split),
        x_mvr_process_split=x_mvr_process_split,
    )


def _compute_vc_mvr_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> HPRBackendResult:
    if not getattr(args, "is_heat_pumping", True):
        return HPRBackendResult.failure(
            reason=(
                "Vapour compression with MVR cascade is heat-pump-only; "
                "refrigeration targets are not supported."
            )
        )

    state_vars = _parse_vc_mvr_state_variables(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)
    parsed = _unpack_vc_mvr_state(state_vars, args)

    try:
        hp = VapourCompressionMvrCascade()
        hp.solve(
            T_evap_vc=parsed["T_evap_vc"],
            T_cond_vc=parsed["T_cond_vc"],
            dT_lift_mvr=parsed["dT_lift_mvr"],
            Q_heat_base=state_vars.Q_heat_base,
            x_heat_split=state_vars.x_heat_split,
            Q_heat_available=state_vars.Q_heat_available,
            mvr_source_split=parsed["mvr_source_split"],
            mvr_process_split=parsed["mvr_process_split"],
            dT_subcool_vc=parsed["dT_subcool_vc"],
            dT_subcool_mvr=parsed["dT_subcool_mvr"],
            dT_ihx_gas_side_vc=state_vars.dT_ihx_gas_side,
            eta_comp=args.eta_comp,
            eta_mvr_comp=args.eta_mvr_comp,
            eta_motor=args.eta_motor,
            refrigerant=args.refrigerant_ls,
            mvr_fluid=args.mvr_fluid_ls,
            dt_cascade_hx=args.dt_cascade_hx,
            dtcont=args.dtcont_hp,
        )
        if not hp.solved:
            return _finite_failed_vc_mvr_result(
                reason="VC+MVR cascade candidate is infeasible.",
                state=state_vars,
                work=hp.work,
                penalty_terms=hp.penalty,
                args=args,
            )

        solved_state = _build_solved_vc_mvr_state(state_vars, hp, args)
        hpr_streams = hp.build_stream_collection(
            include_cond=True,
            include_evap=True,
            is_process_stream=False,
            dtcont=args.dtcont_hp,
        )
        w_hpr = hp.work
        cop = hp.Q_heat / w_hpr if w_hpr > 0 else 1.0
        Q_heat_ordered, Q_cool_ordered = _order_vc_mvr_result_duties(hp, args)
        return evaluate_vapour_hpr_result(
            args=args,
            state=solved_state,
            work=w_hpr,
            work_arr=_order_vc_mvr_work(hp, args),
            Q_heat=Q_heat_ordered,
            Q_cool=Q_cool_ordered,
            cop_h=cop,
            hpr_streams=hpr_streams,
            model=hp,
            penalty_terms=hp.penalty,
            dT_subcool=solved_state.dT_subcool,
            debug=debug,
        )
    except Exception as exc:
        if debug:
            raise
        parsed = _unpack_vc_mvr_state(state_vars, args)
        fallback_work = max(float(state_vars.Q_heat_base or 0.0), 1.0)
        return _finite_failed_vc_mvr_result(
            reason=str(exc),
            state=state_vars,
            work=fallback_work,
            penalty_terms=[fallback_work],
            args=args,
        )


def _finite_failed_vc_mvr_result(
    *,
    reason: str,
    state: HPRParsedState,
    work: float,
    penalty_terms: list[float],
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    work = max(float(work), 1.0)
    penalty = float(np.maximum(np.asarray(penalty_terms, dtype=float), 0.0).sum())
    obj = (work + penalty) / max(float(args.Q_hpr_target), 1.0)
    return HPRBackendResult(
        obj=float(obj),
        utility_tot=float(work + penalty),
        w_net=float(work),
        w_hpr=float(work),
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=state.Q_amb_hot,
        Q_amb_cold=state.Q_amb_cold,
        success=False,
        failure_reason=reason,
    )


def _fit_stage_array(values, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(size, dtype=float)
    if arr.size >= size:
        return arr[:size]
    return np.concatenate([arr, np.full(size - arr.size, arr[-1], dtype=float)])


def _normalise_fluid_list(values) -> list[str]:
    if isinstance(values, str):
        values = [values]
    fluids = [str(value).strip() for value in values if str(value).strip()]
    return fluids or ["Water"]


def _derive_provisional_mvr_temperatures(
    *,
    T_cond_vc: np.ndarray,
    dT_subcool_vc: np.ndarray,
    dT_lift_mvr: np.ndarray,
    dt_cascade_hx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Provisional serial MVR temperatures before the VC profile is solved."""
    T_evap_mvr = np.empty_like(dT_lift_mvr, dtype=float)
    T_cond_mvr = np.empty_like(dT_lift_mvr, dtype=float)
    T_evap_mvr[0] = T_cond_vc[0] - dT_subcool_vc[0] - float(dt_cascade_hx)
    for j, lift in enumerate(dT_lift_mvr):
        T_cond_mvr[j] = T_evap_mvr[j] + lift
        if j + 1 < dT_lift_mvr.size:
            T_evap_mvr[j + 1] = T_cond_mvr[j]
    return T_evap_mvr, T_cond_mvr


def _build_solved_vc_mvr_state(
    state: HPRParsedState,
    hp: VapourCompressionMvrCascade,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    """Replace provisional MVR temperatures with solved cascade temperatures."""
    n_vc = _num_vc_stages(args)
    n_mvr = _num_mvr_stages(args)
    parsed = _unpack_vc_mvr_state(state, args)
    dT_subcool_mvr = np.array(
        [cycle.dT_subcool for cycle in hp.mvr_cycles],
        dtype=float,
    )
    dT_subcool_vc = np.array(
        [cycle.dT_subcool for cycle in hp.vc_cycles],
        dtype=float,
    )
    if dT_subcool_mvr.size != n_mvr or dT_subcool_vc.size != n_vc:
        raise ValueError("Solved VC+MVR stage counts do not match targeting inputs.")
    return state.model_copy(
        update={
            "T_cond": np.concatenate([hp.T_cond_mvr, parsed["T_cond_vc"]]),
            "T_evap": np.concatenate([hp.T_evap_mvr, parsed["T_evap_vc"]]),
            "dT_subcool": np.concatenate([dT_subcool_mvr, dT_subcool_vc]),
        }
    )


def _order_vc_mvr_result_duties(
    hp: VapourCompressionMvrCascade,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return result duties in the shared [MVR, VC] temperature order."""
    n_vc = _num_vc_stages(args)
    return (
        np.concatenate([hp.Q_heat_arr[n_vc:], hp.Q_heat_arr[:n_vc]]),
        np.concatenate([hp.Q_cool_arr[n_vc:], hp.Q_cool_arr[:n_vc]]),
    )


def _order_vc_mvr_work(
    hp: VapourCompressionMvrCascade,
    args: HeatPumpTargetInputs,
) -> np.ndarray:
    """Return per-stage work in the shared [MVR, VC] temperature order."""
    n_vc = _num_vc_stages(args)
    return np.concatenate([hp.work_arr[n_vc:], hp.work_arr[:n_vc]])


def _unpack_vc_mvr_state(
    state: HPRParsedState,
    args: HeatPumpTargetInputs,
) -> dict[str, np.ndarray]:
    n_vc = _num_vc_stages(args)
    n_mvr = _num_mvr_stages(args)
    T_cond = state.T_cond
    T_evap = state.T_evap
    dT_subcool = state.dT_subcool
    if T_cond is None or T_evap is None or dT_subcool is None:
        raise ValueError("VC+MVR parsed state must include stage temperature arrays.")
    expected_temperature_size = n_mvr + n_vc
    if T_cond.size != expected_temperature_size:
        raise ValueError(
            "VC+MVR parsed condenser temperatures must include VC and MVR stages."
        )
    if T_evap.size != expected_temperature_size:
        raise ValueError(
            "VC+MVR parsed evaporator temperatures must include VC and MVR stages."
        )
    if state.x_heat_split is None or state.x_heat_split.size != n_vc:
        raise ValueError(
            "VC+MVR parsed heat state must include VC heat split fractions."
        )
    if dT_subcool.size != expected_temperature_size:
        raise ValueError("VC+MVR parsed subcooling must include MVR and VC stages.")
    return {
        "T_cond_mvr": T_cond[:n_mvr],
        "T_cond_vc": T_cond[n_mvr:],
        "T_evap_mvr": T_evap[:n_mvr],
        "T_evap_vc": T_evap[n_mvr:],
        "dT_subcool_mvr": dT_subcool[:n_mvr],
        "dT_subcool_vc": dT_subcool[n_mvr:],
        "mvr_source_split": float(state.x_mvr_source_split),
        "mvr_process_split": state.x_mvr_process_split,
        "dT_lift_mvr": T_cond[:n_mvr] - T_evap[:n_mvr],
    }
