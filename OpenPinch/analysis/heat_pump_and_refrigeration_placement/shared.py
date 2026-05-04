"""Shared helpers for heat-pump and refrigeration targeting."""

from typing import Callable, Tuple

import numpy as np
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI

from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...lib.config import tol
from ...lib.enums import PT
from ...lib.schema import HPRTargetInputs
from ...utils.blackbox_minimisers import multiminima
from ...utils.miscellaneous import (
    clean_composite_curve_ends,
    delta_vals,
    g_ineq_penalty,
)
from ..problem_table_analysis import (
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
    title: str = None,
) -> go.Figure:
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
            hpr_hot_streams, hpr_cold_streams
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
    fig.show()
    return fig


def solve_hpr_placement(
    f_obj: Callable,
    x0_ls: list | float,
    bnds: list,
    args: HPRTargetInputs,
) -> dict:
    if isinstance(x0_ls, (int, float)):
        x0_ls = [x0_ls]
    x0_ls = np.asarray(x0_ls, dtype=np.float64)
    if len(x0_ls.shape) == 1:
        x0_ls = [x0_ls]

    local_minima_x = multiminima(
        func=f_obj,
        func_kwargs=args,
        x0_ls=x0_ls,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,
        opt_kwargs={"n_runs": 10},
    )
    if local_minima_x.size == 0:
        raise ValueError(
            f"Heat pump and refrigeration targeting ({args.hpr_type}) failed to return any local minima."
        )

    res = f_obj(local_minima_x[0], args, debug=args.debug)
    if not res.get("success", True):
        raise ValueError(
            f"Heat pump and refrigeration targeting ({args.hpr_type}) failed to return an optimal result."
        )
    res["success"] = True
    res["amb_streams"] = get_ambient_air_stream(
        res["Q_amb_hot"], res["Q_amb_cold"], args
    )
    return res


def create_stream_collection_of_background_profile(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> StreamCollection:
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
    args: HPRTargetInputs,
) -> list:
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
    args,
) -> dict:
    return {
        "hpr_hot_streams": _build_latent_streams(
            T_cond, args.dt_phase_change, Q_cond, is_hot=True
        ),
        "hpr_cold_streams": _build_latent_streams(
            T_evap, args.dt_phase_change, Q_evap, is_hot=False
        ),
    }


def get_ambient_air_stream(
    Q_amb_hot: float = 0.0,
    Q_amb_cold: float = 0.0,
    args: HPRTargetInputs = None,
) -> StreamCollection:
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
    return (
        work
        + (Q_ext_heat * heat_to_power_ratio)
        + (Q_ext_cold * cold_to_power_ratio)
        + penalty
    ) / Q_hpr_target


#######################################################################################################
# Helper Functions
#######################################################################################################


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = create_problem_table_with_t_int(
        streams=hot_streams + cold_streams,
        is_shifted=False,
    )
    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value],
            hot_streams,
            cold_streams,
            is_shifted=False,
        )
    )
    return pt.col[PT.T.value], pt.col[PT.H_HOT_UT.value], pt.col[PT.H_COLD_UT.value]
