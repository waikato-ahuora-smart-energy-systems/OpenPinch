"""Shared helpers for heat pump and refrigeration targeting."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from CoolProp.CoolProp import PropsSI

from ....classes.stream_collection import StreamCollection
from ....lib.config import tol
from ....lib.enums import PT
from ....lib.schemas.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPRParsedState,
    HPRThermoArtifacts,
    SimulatedHPRAnnualizedCostAccounting,
)
from ....utils.blackbox_minimisers import multiminima
from ....utils.costing import (
    compute_annual_capital_cost,
    compute_annual_energy_cost,
    compute_capital_cost,
)
from ...common.miscellaneous import (
    g_ineq_penalty,
)
from ...common.problem_table_analysis import (
    get_process_heat_cascade,
    get_utility_heat_cascade,
)
from ._shared.plotting import plot_multi_hp_profiles_from_results
from ._shared.streams import (
    get_ambient_air_stream,
    get_carnot_hpr_cycle_streams,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
)

__all__ = [
    "PropsSI",
    "solve_hpr_placement",
    "compute_entropic_mean_temperature",
    "calc_carnot_heat_pump_cop",
    "calc_carnot_heat_engine_eta",
    "get_carnot_hpr_cycle_streams",
    "get_Q_vals_at_T_hpr_from_bckgrd_profile",
    "validate_vapour_hp_refrigerant_ls",
    "calc_hpr_obj",
    "calc_simulated_hpr_annualized_costs",
    "evaluate_carnot_hpr_result",
    "evaluate_vapour_hpr_result",
    "plot_multi_hp_profiles_from_results",
    "get_process_heat_cascade",
    "get_utility_heat_cascade",
    "g_ineq_penalty",
    "tol",
]


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
    x0_arr = _verify_x0_ls(x0_ls)
    try:
        local_minima_x, local_minima_f = multiminima(
            func=f_obj,
            func_kwargs=args,
            x0_ls=x0_arr,
            bounds=bnds,
            optimiser_handle=args.bb_minimiser,
            opt_kwargs={"n_runs": max(1, int(args.max_multi_start))},
        )
    except Exception as exc:
        if x0_arr is None or x0_arr.size == 0:
            raise ValueError(
                "Heat pump and refrigeration targeting "
                f"({args.hpr_type}) failed during optimisation and has no "
                "initial candidate fallback."
            ) from exc
        local_minima_x, local_minima_f = np.asarray([]), np.asarray([])
    candidate_x, candidate_f = _merge_candidate_points(
        local_minima_x=local_minima_x,
        local_minima_f=local_minima_f,
        x0_arr=x0_arr,
        f_obj=f_obj,
        args=args,
    )
    if candidate_x.size == 0:
        raise ValueError(
            "Heat pump and refrigeration targeting "
            f"({args.hpr_type}) failed to return any local minima."
        )

    for candidate_idx in np.argsort(candidate_f):
        result = _evaluate_hpr_candidate(f_obj, candidate_x[candidate_idx], args)
        if result.success and np.isfinite(float(result.obj)):
            return result.with_updates(
                success=True,
                amb_streams=get_ambient_air_stream(
                    result.Q_amb_hot, result.Q_amb_cold, args
                ),
            )

    raise ValueError(
        "Heat pump and refrigeration targeting "
        f"({args.hpr_type}) failed to return an optimal result."
    )


def _merge_candidate_points(
    local_minima_x: list | np.ndarray,
    local_minima_f: list | np.ndarray,
    x0_arr: np.ndarray | None,
    f_obj: Callable,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_blocks = []
    objective_blocks = []
    local_minima_arr = np.asarray(local_minima_x, dtype=float)
    if local_minima_arr.size:
        if local_minima_arr.ndim == 1:
            local_minima_arr = local_minima_arr.reshape(1, -1)
        candidate_blocks.append(local_minima_arr)
        objective_blocks.append(np.asarray(local_minima_f, dtype=float).reshape(-1))
    if x0_arr is not None and x0_arr.size:
        x0_block = np.asarray(x0_arr, dtype=float).reshape(x0_arr.shape[0], -1)
        candidate_blocks.append(x0_block)
        objective_blocks.append(
            np.asarray(
                [_score_hpr_candidate_objective(f_obj, x0, args) for x0 in x0_block],
                dtype=float,
            )
        )
    if not candidate_blocks:
        return np.asarray([]), np.asarray([])
    n_cols = (
        np.asarray(x0_arr, dtype=float).reshape(x0_arr.shape[0], -1).shape[1]
        if x0_arr is not None and x0_arr.size
        else candidate_blocks[0].shape[1]
    )
    filtered_blocks = []
    filtered_objectives = []
    for block, objectives in zip(candidate_blocks, objective_blocks):
        if block.shape[1] != n_cols:
            continue
        filtered_blocks.append(block)
        filtered_objectives.append(objectives)
    if not filtered_blocks:
        return np.asarray([]), np.asarray([])

    candidate_x = np.vstack(filtered_blocks)
    candidate_f = np.concatenate(filtered_objectives)
    _, unique_idx = np.unique(candidate_x, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return candidate_x[unique_idx], candidate_f[unique_idx]


def _score_hpr_candidate_objective(
    f_obj: Callable,
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> float:
    try:
        result = f_obj(x, args, debug=False)
        obj = float(result["obj"])
    except Exception:
        return 1e30
    return obj if np.isfinite(obj) else 1e30


def _evaluate_hpr_candidate(
    f_obj: Callable,
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    result = f_obj(x, args, debug=args.debug)
    if not isinstance(result, HPRBackendResult):
        raise TypeError(
            "Heat pump and refrigeration objective functions must return "
            "HPRBackendResult."
        )
    return result


def _build_hpr_accounting(
    *,
    work: float,
    Q_ext_heat: float,
    Q_ext_cold: float,
    args: HeatPumpTargetInputs,
    penalty_terms: np.ndarray | None = None,
    penalise_external_cold_when_refrigerating: bool = False,
) -> tuple[float, float, float, float]:
    """Standardize external-utility, penalty, and objective semantics."""
    positive_penalty_terms = (
        np.maximum(penalty_terms, 0.0) if penalty_terms is not None else np.array([])
    )
    penalty = (
        float(
            g_ineq_penalty(
                positive_penalty_terms,
                eta=args.eta_penalty,
                rho=args.rho_penalty,
                form="square",
            )
        )
        if positive_penalty_terms.size
        else 0.0
    )
    if penalise_external_cold_when_refrigerating and not getattr(
        args, "is_heat_pumping", True
    ):
        penalty += float(
            g_ineq_penalty(g=Q_ext_cold, rho=args.rho_penalty, form="square")
        )

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


def _cycle_penalty(
    *,
    args: HeatPumpTargetInputs,
    cycle_penalty_terms: Any = None,
) -> float:
    cycle_terms = np.maximum(
        np.asarray(
            [] if cycle_penalty_terms is None else cycle_penalty_terms, dtype=float
        ).reshape(-1),
        0.0,
    )
    if not cycle_terms.size:
        return 0.0
    return float(
        g_ineq_penalty(
            cycle_terms,
            eta=float(getattr(args, "eta_penalty", 0.01)),
            rho=float(getattr(args, "rho_penalty", 10.0)),
            form="square",
        )
    )


def calc_simulated_hpr_annualized_costs(
    *,
    work: float,
    work_arr: float | list | np.ndarray | None,
    Q_ext_heat: float,
    Q_ext_cold: float,
    hpr_streams: StreamCollection,
    hx_units: int,
    penalty_power_equivalent: float,
    args: HeatPumpTargetInputs,
) -> SimulatedHPRAnnualizedCostAccounting:
    """Return unit-aware annualized cost accounting for simulated HPR candidates."""
    annual_hours = max(float(args.annual_op_time), 0.0)
    ele_price = max(float(args.ele_price), 0.0)
    heat_price = ele_price * max(float(args.heat_to_power_ratio), 0.0)
    cold_price = ele_price * max(float(args.cold_to_power_ratio), 0.0)

    operating_cost = (
        compute_annual_energy_cost(work, ele_price, annual_hours)
        + compute_annual_energy_cost(Q_ext_heat, heat_price, annual_hours)
        + compute_annual_energy_cost(Q_ext_cold, cold_price, annual_hours)
    ).to("$/y")

    work_values = np.asarray([] if work_arr is None else work_arr, dtype=float).reshape(
        -1
    )
    compressor_capital = compute_capital_cost(
        work,
        max(int(np.count_nonzero(work_values > tol)), 1),
        args.hpr_comp_fixed_cost,
        args.hpr_comp_variable_cost,
        args.hpr_comp_cost_exp,
    )

    hpr_hot_streams = hpr_streams.get_hot_utility_streams()
    hpr_cold_streams = hpr_streams.get_cold_utility_streams()
    hx_duty = sum(
        max(float(streams.sum_stream_attribute("heat_flow", idx=args.idx)), 0.0)
        for streams in (hpr_hot_streams, hpr_cold_streams)
        if len(streams) > 0
    )
    hx_capital = compute_capital_cost(
        hx_duty,
        hx_units,
        args.hpr_hx_fixed_cost,
        args.hpr_hx_variable_cost,
        args.hpr_hx_cost_exp,
    )

    capital_cost = (compressor_capital + hx_capital).to("$")
    annualized_capital = compute_annual_capital_cost(
        capital_cost,
        args.discount_rate,
        args.serv_life,
    )
    total_annualized = (operating_cost + annualized_capital).to("$/y")
    feasibility_penalty = compute_annual_energy_cost(
        penalty_power_equivalent,
        ele_price,
        annual_hours,
    )

    return SimulatedHPRAnnualizedCostAccounting(
        hpr_operating_cost=operating_cost,
        hpr_capital_cost=capital_cost,
        hpr_annualized_capital_cost=annualized_capital,
        hpr_total_annualized_cost=total_annualized,
        hpr_compressor_capital_cost=compressor_capital,
        hpr_heat_exchanger_capital_cost=hx_capital,
        feasibility_penalty=feasibility_penalty,
    )


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
    penalty_terms: np.ndarray | None = None,
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

    hot_streams = (
        hpr_streams.get_hot_utility_streams()
        + args.bckgrd_hot_streams
        + get_ambient_air_stream(Q_amb_hot=state.Q_amb_hot, args=args)
    )
    cold_streams = (
        hpr_streams.get_cold_utility_streams()
        + args.bckgrd_cold_streams
        + get_ambient_air_stream(Q_amb_cold=state.Q_amb_cold, args=args)
    )
    pt = get_process_heat_cascade(
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        is_shifted=True,
        idx=args.idx,
    )
    Q_ext_heat = float(pt[PT.H_NET][0])
    Q_ext_cold = float(pt[PT.H_NET][-1])
    hx_units = len(hot_streams) + len(cold_streams)
    penalty_power_equivalent = _cycle_penalty(
        args=args,
        cycle_penalty_terms=penalty_terms,
    )
    cost_accounting = calc_simulated_hpr_annualized_costs(
        work=float(work),
        work_arr=work_arr,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        hpr_streams=hpr_streams,
        hx_units=hx_units,
        penalty_power_equivalent=penalty_power_equivalent,
        args=args,
    )
    penalty = float(cost_accounting.feasibility_penalty.to("$/y").value)
    obj = float(cost_accounting.hpr_total_annualized_cost.to("$/y").value) + penalty

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
                f"Obj {obj:.5f} $/y = "
                f"{cost_accounting.hpr_total_annualized_cost.value:.5f} $/y + "
                f"{penalty:.5f} $/y"
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
        hpr_operating_cost=cost_accounting.hpr_operating_cost,
        hpr_capital_cost=cost_accounting.hpr_capital_cost,
        hpr_annualized_capital_cost=cost_accounting.hpr_annualized_capital_cost,
        hpr_total_annualized_cost=cost_accounting.hpr_total_annualized_cost,
        hpr_compressor_capital_cost=cost_accounting.hpr_compressor_capital_cost,
        hpr_heat_exchanger_capital_cost=(
            cost_accounting.hpr_heat_exchanger_capital_cost
        ),
        feasibility_penalty=penalty,
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
