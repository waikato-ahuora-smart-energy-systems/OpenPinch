"""Shared helpers for heat pump and refrigeration targeting."""

from typing import Any, Callable, Tuple

import numpy as np
from CoolProp.CoolProp import PropsSI

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - optional dependency guard
    go = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

from ....classes.stream import Stream
from ....classes.stream_collection import StreamCollection
from ....lib.config import tol
from ....lib.enums import PT
from ....lib.schemas.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPRParsedState,
    HPRThermoArtifacts,
)
from ....utils.blackbox_minimisers import multiminima
from ....utils.miscellaneous import (
    clean_composite_curve_ends,
    delta_vals,
    g_ineq_penalty,
)
from ...common.problem_table_analysis import (
    create_problem_table_with_t_int,
    get_process_heat_cascade,
    get_utility_heat_cascade,
)

__all__ = [
    "PropsSI",
    "solve_hpr_placement",
    "create_stream_collection_of_background_profile",
    "get_Q_vals_at_T_hpr_from_bckgrd_profile",
    "compute_entropic_mean_temperature",
    "calc_carnot_heat_pump_cop",
    "calc_carnot_heat_engine_eta",
    "_append_unspecified_final_cascade_cooling_duty",
    "_get_hpr_cascade",
    "validate_vapour_hp_refrigerant_ls",
    "_get_carnot_hpr_cycle_cascade_profile",
    "get_carnot_hpr_cycle_streams",
    "_build_latent_streams",
    "get_ambient_air_stream",
    "calc_hpr_obj",
    "evaluate_carnot_hpr_result",
    "evaluate_vapour_hpr_result",
    "plot_multi_hp_profiles_from_results",
    "get_process_heat_cascade",
    "get_utility_heat_cascade",
    "g_ineq_penalty",
    "tol",
]


def plot_multi_hp_profiles_from_results(
    T_hot: np.ndarray = None,
    H_hot: np.ndarray = None,
    T_cold: np.ndarray = None,
    H_cold: np.ndarray = None,
    hpr_hot_streams: StreamCollection = None,
    hpr_cold_streams: StreamCollection = None,
    idx: int = 0,
    title: str = None,
) -> "go.Figure":
    """Plot background source/sink profiles alongside solved HPR cycle streams."""
    go = _require_plotly()
    fig = go.Figure()

    if T_hot is not None and H_hot is not None:
        T_hot, H_hot = clean_composite_curve_ends(T_hot, H_hot)
        fig.add_trace(
            go.Scatter(
                x=H_hot,
                y=T_hot,
                mode="lines",
                name="Sink",
                line={"color": "red", "width": 2},
            )
        )

    if T_cold is not None and H_cold is not None:
        T_cold, H_cold = clean_composite_curve_ends(T_cold, H_cold)
        fig.add_trace(
            go.Scatter(
                x=H_cold,
                y=T_cold,
                mode="lines",
                name="Source",
                line={"color": "blue", "width": 2},
            )
        )

    if hpr_hot_streams is not None and hpr_cold_streams is not None:
        T_hpr_arr, H_hpr_hot, H_hpr_cold = _get_hpr_cascade(
            hpr_hot_streams,
            hpr_cold_streams,
            idx=idx,
        )
        T_hpr_hot, H_hpr_hot = clean_composite_curve_ends(T_hpr_arr, H_hpr_hot)
        T_hpr_cold, H_hpr_cold = clean_composite_curve_ends(T_hpr_arr, H_hpr_cold)
        fig.add_trace(
            go.Scatter(
                x=H_hpr_hot,
                y=T_hpr_hot,
                mode="lines",
                name="Condenser",
                line={"color": "darkred", "width": 1.8, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=H_hpr_cold,
                y=T_hpr_cold,
                mode="lines",
                name="Evaporator",
                line={"color": "darkblue", "width": 1.8, "dash": "dash"},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Heat Flow / kW",
        yaxis_title="Temperature / degC",
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", zeroline=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)")
    fig.add_vline(x=0.0, line_color="black", line_width=2)
    return fig


def _require_plotly():
    if _PLOTLY_IMPORT_ERROR is not None:
        raise ImportError(
            "Plotly is required for HPR profile plotting. "
            "Install it directly or reinstall OpenPinch with "
            "'pip install openpinch[notebook]' or 'pip install openpinch[dashboard]'."
        ) from _PLOTLY_IMPORT_ERROR
    return go


def _verify_x0_ls(x0_ls: list | float | None) -> np.ndarray | None:
    if x0_ls is None:
        return None

    if isinstance(x0_ls, (int, float)):
        x0_ls = [x0_ls]

    x0_arr = np.asarray(x0_ls, dtype=np.float64)
    if x0_arr.size == 0:
        return None
    if x0_arr.ndim == 0:
        return x0_arr.reshape(1, 1)
    if x0_arr.ndim == 1:
        return x0_arr.reshape(1, -1)
    return x0_arr


def solve_hpr_placement(
    f_obj: Callable,
    x0_ls: list | float,
    bnds: list,
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Run the configured multistart optimiser and post-process the best result."""
    local_minima_x = multiminima(
        func=f_obj,
        func_kwargs=args,
        x0_ls=_verify_x0_ls(x0_ls),
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,
        opt_kwargs={"n_runs": max(1, int(args.max_multi_start))},
    )
    if local_minima_x.size == 0:
        raise ValueError(
            "Heat pump and refrigeration targeting "
            f"({args.hpr_type}) failed to return any local minima."
        )

    result = f_obj(local_minima_x[0], args, debug=args.debug)
    if not isinstance(result, HPRBackendResult):
        raise TypeError(
            "Heat pump and refrigeration objective functions must return "
            "HPRBackendResult."
        )
    if not result.success:
        raise ValueError(
            "Heat pump and refrigeration targeting "
            f"({args.hpr_type}) failed to return an optimal result."
        )
    return result.with_updates(
        success=True,
        amb_streams=get_ambient_air_stream(result.Q_amb_hot, result.Q_amb_cold, args),
    )


def _normalise_positive_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return 0.0
    return max(float(arr.reshape(-1)[0]), 0.0)


def _build_hpr_accounting(
    *,
    work: float,
    Q_ext_heat: float,
    Q_ext_cold: float,
    args: HeatPumpTargetInputs,
    penalty_terms: Any = None,
    penalise_external_cold_when_refrigerating: bool = False,
) -> tuple[float, float, float, float]:
    """Standardize external-utility, penalty, and objective semantics."""
    Q_ext_heat = _normalise_positive_scalar(Q_ext_heat)
    Q_ext_cold = _normalise_positive_scalar(Q_ext_cold)
    penalty_terms = (
        np.asarray([], dtype=float)
        if penalty_terms is None
        else np.asarray(penalty_terms, dtype=float).reshape(-1)
    )
    positive_penalty_terms = np.maximum(penalty_terms, 0.0)
    eta_penalty = float(getattr(args, "eta_penalty", 0.01))
    rho_penalty = float(getattr(args, "rho_penalty", 10.0))
    penalty = (
        float(
            g_ineq_penalty(
                positive_penalty_terms,
                eta=eta_penalty,
                rho=rho_penalty,
                form="square",
            )
        )
        if positive_penalty_terms.size
        else 0.0
    )
    if penalise_external_cold_when_refrigerating and not getattr(
        args, "is_heat_pumping", True
    ):
        penalty += float(g_ineq_penalty(g=Q_ext_cold, rho=rho_penalty, form="square"))

    obj = float(
        calc_hpr_obj(
            work=work,
            Q_ext_heat=Q_ext_heat,
            Q_ext_cold=Q_ext_cold,
            Q_hpr_target=args.Q_hpr_target,
            heat_to_power_ratio=args.heat_to_power_ratio,
            cold_to_power_ratio=args.cold_to_power_ratio,
            penalty=penalty,
        )
    )
    return Q_ext_heat, Q_ext_cold, penalty, obj


def evaluate_carnot_hpr_result(
    *,
    args: HeatPumpTargetInputs,
    state: HPRParsedState,
    w_net: float,
    w_hpr: float | list | np.ndarray,
    Q_cond_total: np.ndarray,
    Q_evap_total: np.ndarray,
    w_he: float | list | np.ndarray | None = None,
    heat_recovery: float | list | np.ndarray | None = None,
    cop_h: float | list | np.ndarray | None = None,
    eta_he: float | list | np.ndarray | None = None,
    Q_cond: np.ndarray | None = None,
    Q_evap: np.ndarray | None = None,
    Q_cond_he: np.ndarray | None = None,
    Q_evap_he: np.ndarray | None = None,
    penalty_terms: Any = None,
    debug: bool = False,
) -> HPRBackendResult:
    """Shared Carnot-family accounting, plotting, and result assembly."""
    H_cold_with_amb = args.H_cold + args.z_amb_cold * state.Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state.Q_amb_hot

    Q_ext_heat, Q_ext_cold, penalty, obj = _build_hpr_accounting(
        work=float(w_net),
        Q_ext_heat=np.abs(H_cold_with_amb[0])
        - np.asarray(Q_cond_total, dtype=float).sum(),
        Q_ext_cold=np.abs(H_hot_with_amb[-1])
        - np.asarray(Q_evap_total, dtype=float).sum(),
        args=args,
        penalty_terms=penalty_terms,
        penalise_external_cold_when_refrigerating=True,
    )
    hpr_streams = get_carnot_hpr_cycle_streams(
        state.T_cond,
        Q_cond_total,
        state.T_evap,
        Q_evap_total,
        args,
    )
    debug_figure = None
    if debug:
        debug_figure = plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            hpr_streams.get_hot_utility_streams(),
            hpr_streams.get_cold_utility_streams(),
            title=(
                f"Obj {obj:.5f} = {(float(w_net) / args.Q_hpr_target):.5f} + "
                f"{(Q_ext_heat / args.Q_hpr_target):.5f} + "
                f"{(Q_ext_cold / args.Q_hpr_target):.5f} + "
                f"{(penalty / args.Q_hpr_target):.5f}"
            ),
            idx=args.idx,
        )

    return HPRBackendResult(
        obj=obj,
        utility_tot=float(w_net + Q_ext_heat + Q_ext_cold),
        w_net=float(w_net),
        w_hpr=w_hpr,
        w_he=w_he,
        heat_recovery=heat_recovery,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_amb_hot=state.Q_amb_hot,
        Q_amb_cold=state.Q_amb_cold,
        cop_h=cop_h,
        eta_he=eta_he,
        T_cond=state.T_cond,
        T_evap=state.T_evap,
        Q_cond=Q_cond_total if Q_cond is None else Q_cond,
        Q_evap=Q_evap_total if Q_evap is None else Q_evap,
        Q_cond_he=Q_cond_he,
        Q_evap_he=Q_evap_he,
        artifacts=HPRThermoArtifacts(
            hpr_streams=hpr_streams, debug_figure=debug_figure
        ),
    )


def evaluate_vapour_hpr_result(
    *,
    args: HeatPumpTargetInputs,
    state: HPRParsedState,
    work: float,
    work_arr: float | list | np.ndarray,
    Q_heat: np.ndarray,
    Q_cool: np.ndarray,
    cop_h: float,
    hpr_streams: StreamCollection,
    model: Any = None,
    penalty_terms: Any = None,
    dT_subcool: np.ndarray | None = None,
    dT_superheat: np.ndarray | None = None,
    debug: bool = False,
) -> HPRBackendResult:
    """Shared simulated-vapour accounting, plotting, and result assembly."""
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state.Q_amb_hot
    H_cold_with_amb = args.H_cold + args.z_amb_cold * state.Q_amb_cold

    pt_cond = get_process_heat_cascade(
        hot_streams=hpr_streams.get_hot_utility_streams(),
        cold_streams=args.bckgrd_cold_streams
        + get_ambient_air_stream(Q_amb_cold=state.Q_amb_cold, args=args),
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams
        + get_ambient_air_stream(Q_amb_hot=state.Q_amb_hot, args=args),
        cold_streams=hpr_streams.get_cold_utility_streams(),
        is_shifted=True,
    )

    Q_ext_heat, Q_ext_cold, penalty, obj = _build_hpr_accounting(
        work=float(work),
        Q_ext_heat=pt_cond[PT.H_NET][0],
        Q_ext_cold=pt_evap[PT.H_NET][-1],
        args=args,
        penalty_terms=np.concatenate(
            [
                np.array(
                    [pt_cond[PT.H_NET][-1], pt_evap[PT.H_NET][0]],
                    dtype=float,
                ),
                np.asarray(
                    [] if penalty_terms is None else penalty_terms, dtype=float
                ).reshape(-1),
            ]
        ),
        penalise_external_cold_when_refrigerating=True,
    )

    debug_figure = None
    if debug:
        debug_figure = plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            hpr_streams.get_hot_utility_streams(),
            hpr_streams.get_cold_utility_streams(),
            title=(
                f"Obj {obj:.5f} = {(float(work) / args.Q_hpr_target):.5f} + "
                f"{(Q_ext_heat / args.Q_hpr_target):.5f} + "
                f"{(Q_ext_cold / args.Q_hpr_target):.5f} + "
                f"{(penalty / args.Q_hpr_target):.5f}"
            ),
            idx=args.idx,
        )

    return HPRBackendResult(
        obj=obj,
        utility_tot=float(work + Q_ext_heat + Q_ext_cold),
        w_net=float(work),
        w_hpr=work_arr,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_amb_hot=state.Q_amb_hot,
        Q_amb_cold=state.Q_amb_cold,
        cop_h=float(cop_h),
        T_cond=state.T_cond,
        T_evap=state.T_evap,
        dT_subcool=dT_subcool,
        dT_superheat=dT_superheat,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        artifacts=HPRThermoArtifacts(
            hpr_streams=hpr_streams,
            model=model,
            debug_figure=debug_figure,
        ),
    )


def create_stream_collection_of_background_profile(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> StreamCollection:
    """Convert a temperature-enthalpy profile into piecewise stream segments."""
    s = StreamCollection()

    T_vals = np.asarray(T_vals)
    H_vals = np.abs(np.asarray(H_vals))

    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    dh_vals = delta_vals(H_vals)
    dh_indices = np.argwhere(np.abs(dh_vals) > tol).flatten()
    for i in dh_indices:
        if dh_vals[i] > tol:
            s.add(
                Stream(
                    t_supply=T_vals[i + 1],
                    t_target=T_vals[i],
                    heat_flow=dh_vals[i],
                )
            )
        elif -dh_vals[i] > tol:
            s.add(
                Stream(
                    t_supply=T_vals[i],
                    t_target=T_vals[i + 1],
                    heat_flow=-dh_vals[i],
                )
            )
    return s


def get_Q_vals_at_T_hpr_from_bckgrd_profile(
    T_hpr: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *,
    is_cond: bool = True,
) -> np.ndarray:
    """Read stage duties from a background profile at proposed HPR temperatures."""
    H_less_origin = np.interp(T_hpr, T_vals[::-1], H_vals[::-1])
    H = (
        np.concatenate((H_less_origin, np.array([0.0])))
        if is_cond
        else np.concatenate((np.array([0.0]), H_less_origin))
    )
    temp = np.roll(H, -1)
    temp[-1] = 0
    Q = H - temp
    Q_hx = Q[:-1]
    return np.where(Q_hx > 0.0, Q_hx, 0.0)


def compute_entropic_mean_temperature(
    T_arr: np.ndarray | list,
    Q_arr: np.ndarray | list,
    *,
    input_T_units: str = "C",
) -> float:
    """Return the entropic mean temperature for a distributed heat load."""
    T_arr = np.asarray(T_arr, dtype=float)
    Q_arr = np.asarray(Q_arr, dtype=float)
    unit_offset = 273.15 if input_T_units == "C" else 0
    if T_arr.var() < tol:
        return T_arr[0] + unit_offset
    S_tot = (Q_arr / (T_arr + unit_offset)).sum()
    return Q_arr.sum() / S_tot if S_tot > 0 else (T_arr.mean() + unit_offset)


def calc_carnot_heat_pump_cop(
    T_h: float | np.ndarray,
    T_l: float | np.ndarray,
    eta_ii: float,
) -> float | np.ndarray:
    """Compute a Carnot-based heating COP with a second-law efficiency factor."""
    T_h_arr = np.asarray(T_h, dtype=float)
    T_l_arr = np.asarray(T_l, dtype=float)
    delta_T = T_h_arr - T_l_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        cop = np.where(
            delta_T > 0.0,
            (T_l_arr / delta_T) * eta_ii + 1.0,
            np.inf,
        )
    return cop.item() if cop.ndim == 0 else cop


def calc_carnot_heat_engine_eta(
    T_h: float | np.ndarray,
    T_l: float | np.ndarray,
    eta_ii: float,
) -> float | np.ndarray:
    """Compute a Carnot-based heat-engine efficiency with a second-law factor."""
    T_h_arr = np.asarray(T_h, dtype=float)
    T_l_arr = np.asarray(T_l, dtype=float)
    eta = np.where(
        T_h_arr > T_l_arr,
        (1 - T_l_arr / T_h_arr) * eta_ii,
        0.0,
    )
    return eta.item() if eta.ndim == 0 else eta


def validate_vapour_hp_refrigerant_ls(
    num_stages: int,
    args: HeatPumpTargetInputs,
) -> list:
    """Return one refrigerant name per vapour-compression stage."""
    if len(args.refrigerant_ls) > 0:
        if args.do_refrigerant_sort:
            refrigerants = [
                ref
                for ref, _ in sorted(
                    ((ref, PropsSI("Tcrit", ref)) for ref in args.refrigerant_ls),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
        else:
            refrigerants = args.refrigerant_ls
        if num_stages <= len(refrigerants):
            return refrigerants[:num_stages]

        padding = [refrigerants[-1]] * (num_stages - len(refrigerants))
        return refrigerants + padding
    return ["water" for _ in range(num_stages)]


def get_carnot_hpr_cycle_streams(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    args: HeatPumpTargetInputs,
) -> StreamCollection:
    """Build one combined HPR utility stream collection for Carnot-cycle summaries."""
    dt_phase_change = float(getattr(args, "dt_phase_change", 1.0))
    return _build_latent_streams(
        T_cond, dt_phase_change, Q_cond, is_hot=True
    ) + _build_latent_streams(T_evap, dt_phase_change, Q_evap, is_hot=False)


def get_ambient_air_stream(
    Q_amb_hot: float = 0.0,
    Q_amb_cold: float = 0.0,
    args: HeatPumpTargetInputs = None,
) -> StreamCollection:
    """Build ambient-air exchange streams implied by the solved HPR result."""
    sc = StreamCollection()
    if Q_amb_hot > tol:
        sc += _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_hot]),
            dt_cont=args.dt_env_cont,
            is_hot=True,
            is_process_stream=True,
            prefix="AIR",
        )
    if Q_amb_cold > tol:
        sc += _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_cold]),
            dt_cont=args.dt_env_cont,
            is_hot=False,
            is_process_stream=True,
            prefix="AIR",
        )
    return sc


def calc_hpr_obj(
    work: float,
    Q_ext_heat: float,
    Q_ext_cold: float,
    Q_hpr_target: float,
    heat_to_power_ratio: float = 1.0,
    cold_to_power_ratio: float = 0.0,
    penalty: float = 0.0,
) -> float:
    """Return the scalar screening objective used by HPR placement solvers."""
    return (
        work
        + (Q_ext_heat * heat_to_power_ratio)
        + (Q_ext_cold * cold_to_power_ratio)
        + penalty
    ) / Q_hpr_target


################################################################################
# Helper Functions
################################################################################


def _build_latent_streams(
    T_ls: np.ndarray,
    dT_phase_change: float,
    Q_ls: np.ndarray,
    *,
    dt_cont: float = 0.0,
    is_hot: bool = True,
    is_process_stream: bool = False,
    prefix: str = "HP",
) -> StreamCollection:
    if len(T_ls) > 1:
        T_ls, Q_ls = _get_carnot_hpr_cycle_cascade_profile(
            T_ls.tolist(), Q_ls.tolist(), dT_phase_change, is_hot
        )

    sc = StreamCollection()
    for i in range(len(Q_ls)):
        sc.add(
            Stream(
                name=f"{prefix}_H{i + 1}" if is_hot else f"{prefix}_C{i + 1}",
                t_supply=T_ls[i] if is_hot else T_ls[i] - dT_phase_change,
                t_target=T_ls[i] - dT_phase_change if is_hot else T_ls[i],
                heat_flow=Q_ls[i],
                dt_cont=dt_cont,
                is_process_stream=is_process_stream,
            )
        )
    return sc


def _get_carnot_hpr_cycle_cascade_profile(
    T_hpr: list,
    Q_hpr: list,
    dT_phase_change: float,
    is_hot: bool,
    i: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    inc = 1 if is_hot else -1
    if i is None:
        i = 0 if is_hot else len(T_hpr)

    i_range = range(i, len(T_hpr) - 1) if is_hot else reversed(range(1, i))
    for i in i_range:
        if abs(T_hpr[i] - T_hpr[i + inc]) < dT_phase_change:
            T_hpr.pop(i + inc)
            Q_hpr[i] += Q_hpr[i + inc]
            Q_hpr.pop(i + inc)
            T_hpr, Q_hpr = _get_carnot_hpr_cycle_cascade_profile(
                T_hpr,
                Q_hpr,
                dT_phase_change,
                is_hot,
                i,
            )
            break

    return T_hpr, Q_hpr


def _append_unspecified_final_cascade_cooling_duty(Q_cool: np.ndarray) -> np.ndarray:
    if Q_cool.size == 0:
        return np.array(np.nan)
    return np.concatenate([Q_cool, np.array([np.nan])])


def _get_hpr_cascade(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    idx: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = idx or 0
    pt = create_problem_table_with_t_int(
        streams=hot_streams + cold_streams,
        is_shifted=False,
        idx=idx,
    )
    pt.update(
        **get_utility_heat_cascade(
            pt[PT.T],
            hot_streams,
            cold_streams,
            is_shifted=False,
            idx=idx,
        )
    )
    return pt[PT.T], pt[PT.H_HOT_UT], pt[PT.H_COLD_UT]
