"""Shared helpers for heat pump and refrigeration targeting."""

from __future__ import annotations

from typing import Any

import CoolProp.CoolProp as _coolprop
import numpy as np

import OpenPinch.analysis.targeting.cascade as _cascade

from ....analysis.economics import (
    compute_annual_capital_cost,
    compute_annual_energy_cost,
    compute_capital_cost,
)
from ....analysis.numerics import g_ineq_penalty as _g_ineq_penalty
from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPRParsedState,
    HPRThermoArtifacts,
    SimulatedHPRAnnualizedCostAccounting,
)
from ....domain.configuration import tol as _tol
from ....domain.enums import PenaltyForm, ProblemTableLabel
from ....domain.stream_collection import StreamCollection
from ..optimisation_adapter import build_hpr_accounting as _build_hpr_accounting
from ._shared import plotting as _plotting
from ._shared import streams as _streams
from ._shared.ambient_preallocation import (
    preallocate_direct_ambient_duties as _preallocate_direct_ambient_duties,
)

__all__ = [
    "calc_simulated_hpr_annualized_costs",
    "calc_carnot_heat_engine_eta",
    "calc_carnot_heat_pump_cop",
    "compute_entropic_mean_temperature",
    "evaluate_carnot_hpr_result",
    "evaluate_vapour_hpr_result",
    "validate_vapour_hp_refrigerant_ls",
]


def _cycle_penalty(
    *,
    args: HeatPumpTargetInputs,
    cycle_penalty_terms: list[float] | None = None,
) -> float:
    if cycle_penalty_terms is None:
        cycle_terms = np.array([])
    else:
        cycle_terms = np.maximum(np.asarray(cycle_penalty_terms, dtype=float), 0.0)
    if not cycle_terms.size:
        return 0.0
    return float(
        _g_ineq_penalty(
            cycle_terms,
            eta=float(getattr(args, "eta_penalty", 0.01)),
            rho=float(getattr(args, "rho_penalty", 10.0)),
            form=PenaltyForm.SQUARE,
        )
    )


def calc_simulated_hpr_annualized_costs(
    *,
    work: float,
    work_arr: np.ndarray | None,
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

    work_values = np.array([]) if work_arr is None else work_arr
    compressor_capital = compute_capital_cost(
        work,
        max(int(np.count_nonzero(work_values > _tol)), 1),
        args.hpr_comp_fixed_cost,
        args.hpr_comp_variable_cost,
        args.hpr_comp_cost_exp,
    )

    hpr_hot_streams = hpr_streams.get_hot_utility_streams()
    hpr_cold_streams = hpr_streams.get_cold_utility_streams()
    hx_duty = sum(
        max(float(streams.sum_stream_attribute("heat_flow", idx=args.period_idx)), 0.0)
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
        Q_ext_heat=np.abs(H_cold_with_amb[0]) - Q_cond_total.sum(),
        Q_ext_cold=np.abs(H_hot_with_amb[-1]) - Q_evap_total.sum(),
        args=args,
        penalty_terms=penalty_terms,
        penalise_external_cold_when_refrigerating=True,
    )
    hpr_streams = _streams.get_carnot_hpr_cycle_streams(
        state.T_cond,
        Q_cond_total,
        state.T_evap,
        Q_evap_total,
        args,
    )
    debug_figure = None
    if debug:
        debug_figure = _plotting.plot_multi_hp_profiles_from_results(
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
            period_idx=args.period_idx,
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
    work_arr: np.ndarray,
    Q_heat: np.ndarray,
    Q_cool: np.ndarray,
    cop_h: float,
    hpr_streams: StreamCollection,
    model: Any = None,
    penalty_terms: list[float] | None = None,
    dT_subcool: np.ndarray | None = None,
    dT_superheat: np.ndarray | None = None,
    debug: bool = False,
) -> HPRBackendResult:
    """Shared simulated-vapour accounting, plotting, and result assembly."""
    ambient_prealloc = _preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=state.Q_amb_hot,
        Q_amb_cold=state.Q_amb_cold,
    )
    H_hot_with_amb = ambient_prealloc.H_hot_with_residual_ambient(args)
    H_cold_with_amb = ambient_prealloc.H_cold_with_residual_ambient(args)

    cond_hot_streams = hpr_streams.get_hot_utility_streams()
    cond_cold_streams = (
        ambient_prealloc.bckgrd_cold_streams
        + _streams.get_ambient_air_stream(
            Q_amb_cold=ambient_prealloc.Q_amb_cold_residual,
            args=args,
        )
    )
    if len(cond_hot_streams) or len(cond_cold_streams):
        pt_cond = _cascade.get_process_heat_cascade(
            hot_streams=cond_hot_streams,
            cold_streams=cond_cold_streams,
            is_shifted=True,
            period_idx=args.period_idx,
        )
        Q_ext_heat = float(pt_cond[ProblemTableLabel.H_NET][0])
        cond_wrong_side = float(pt_cond[ProblemTableLabel.H_NET][-1])
    else:
        Q_ext_heat = 0.0
        cond_wrong_side = 0.0
    evap_hot_streams = (
        ambient_prealloc.bckgrd_hot_streams
        + _streams.get_ambient_air_stream(
            Q_amb_hot=ambient_prealloc.Q_amb_hot_residual,
            args=args,
        )
    )
    evap_cold_streams = hpr_streams.get_cold_utility_streams()
    if len(evap_hot_streams) or len(evap_cold_streams):
        pt_evap = _cascade.get_process_heat_cascade(
            hot_streams=evap_hot_streams,
            cold_streams=evap_cold_streams,
            is_shifted=True,
            period_idx=args.period_idx,
        )
        Q_ext_cold = float(pt_evap[ProblemTableLabel.H_NET][-1])
        evap_wrong_side = float(pt_evap[ProblemTableLabel.H_NET][0])
    else:
        Q_ext_cold = 0.0
        evap_wrong_side = 0.0
    hx_units = (
        len(cond_hot_streams)
        + len(cond_cold_streams)
        + len(evap_hot_streams)
        + len(evap_cold_streams)
    )
    all_penalty_terms = [*(penalty_terms or []), cond_wrong_side, evap_wrong_side]
    penalty_power_equivalent = _cycle_penalty(
        args=args,
        cycle_penalty_terms=all_penalty_terms,
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
        debug_figure = _plotting.plot_multi_hp_profiles_from_results(
            ambient_prealloc.T_hot_residual,
            H_hot_with_amb,
            ambient_prealloc.T_cold_residual,
            H_cold_with_amb,
            hpr_streams.get_hot_utility_streams(),
            hpr_streams.get_cold_utility_streams(),
            title=(
                f"Obj {obj:.5f} $/y = "
                f"{cost_accounting.hpr_total_annualized_cost.value:.5f} $/y + "
                f"{penalty:.5f} $/y"
            ),
            period_idx=args.period_idx,
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
    if T_arr.var() < _tol:
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
                    (
                        (ref, _coolprop.PropsSI("Tcrit", ref))
                        for ref in args.refrigerant_ls
                    ),
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
