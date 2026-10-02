"""Shared helpers for heat pump and refrigeration targeting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import CoolProp.CoolProp as _coolprop
import numpy as np

import OpenPinch.analysis.targeting.cascade as _cascade

from ....analysis.economics import (
    compute_annual_capital_cost,
    compute_annual_energy_cost,
)
from ....analysis.numerics import g_ineq_penalty as _g_ineq_penalty
from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPREvaluationMode,
    HPRParsedState,
    HPRThermoArtifacts,
    SimulatedHPRAnnualizedCostAccounting,
)
from ....domain.configuration import tol as _tol
from ....domain.enums import PenaltyForm, ProblemTableLabel
from ....domain.stream_collection import StreamCollection
from ....domain.value import Value
from ..optimisation_adapter import (
    build_hpr_accounting as _build_hpr_accounting,
)
from ..optimisation_adapter import (
    normalise_hpr_penalty_terms as _normalise_hpr_penalty_terms,
)
from ._shared import plotting as _plotting
from ._shared import streams as _streams
from ._shared.air import (
    CascadeWithAir,
    cascade_with_air,
)
from ._shared.air import (
    ambient_sink_temperature as _ambient_sink_temperature,
)
from ._shared.air import (
    ambient_source_temperature as _ambient_source_temperature,
)
from ._shared.air import (
    cooling_water_sink_temperature as _cooling_water_sink_temperature,
)

__all__ = [
    "HPRCostUnit",
    "SUBCRITICAL_CONDENSING_MARGIN_K",
    "calc_hpr_capital_cost",
    "calc_hpr_machine_capital_costs",
    "NEGLIGIBLE_USEFUL_DUTY_FRACTION",
    "cap_stage_condensing_temperatures",
    "is_negligible_useful_duty",
    "condensing_temperature_search_range",
    "calc_simulated_hpr_annualized_costs",
    "hpr_penalty_cost_scale",
    "refrigeration_allowance",
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
    """Return the dimensionless penalty rho * sum((g / Q_hpr_target)^2).

    Violations are relative to the targeted duty, so the penalty is the same
    for a 10 kW and a 10 MW problem with the same relative shortfall.
    """
    cycle_terms = np.maximum(
        np.asarray(_normalise_hpr_penalty_terms(cycle_penalty_terms), dtype=float),
        0.0,
    )
    if not cycle_terms.size:
        return 0.0
    return float(
        _g_ineq_penalty(
            cycle_terms / _penalty_duty_scale(args),
            eta=float(getattr(args, "eta_penalty", 0.01)),
            rho=float(getattr(args, "rho_penalty", 10.0)),
            form=PenaltyForm.SQUARE,
        )
    )


def _penalty_duty_scale(args: HeatPumpTargetInputs) -> float:
    """Return the duty (kW) that violations are measured against."""
    return max(abs(float(getattr(args, "Q_hpr_target", 0.0) or 0.0)), 1.0)


def hpr_penalty_cost_scale(args: HeatPumpTargetInputs) -> float:
    """Return the annual cost ($/y) that prices the feasibility penalty.

    This is what serving ``Q_hpr_target`` costs with no heat pump: the default
    hot utility (heat pumping) or default refrigeration, operating cost plus
    annualised capital. The capital part keeps the penalty in force when the
    electricity price or operating hours are zero. It is never below 1 $/y.
    """
    duty = _penalty_duty_scale(args)
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    ratio = (
        float(getattr(args, "heat_to_power_ratio", 0.0))
        if is_heat_pumping
        else float(getattr(args, "refrigeration_to_power_ratio", 0.0))
    )
    operating = compute_annual_energy_cost(
        duty,
        max(float(getattr(args, "ele_price", 0.0)), 0.0) * max(ratio, 0.0),
        max(float(getattr(args, "annual_op_time", 0.0)), 0.0),
    )
    unit_capital = float(
        getattr(
            args,
            "hot_utility_capital_cost" if is_heat_pumping else
            "refrigeration_capital_cost",
            0.0,
        )
        or 0.0
    )
    capital = compute_annual_capital_cost(
        Value(duty * max(unit_capital, 0.0), "$"),
        getattr(args, "discount_rate", 0.05),
        getattr(args, "serv_life", 20.0),
    )
    total = float(operating.to("$/y").value) + float(capital.to("$/y").value)
    return max(total, 1.0)


@dataclass(frozen=True)
class HPRCostUnit:
    """One heat-pump machine for capital costing.

    ``Q_cap`` is its heating-side capacity in kW: all the heat it rejects
    outside itself (condensing, desuperheating and subcooling), whether to
    the process or to air. ``T_hot_max`` is the highest temperature at which
    it delivers heat, in degC. ``n_closed`` and ``n_mvr`` count its closed
    refrigerant and open MVR compression stages.
    """

    Q_cap: float
    T_hot_max: float
    n_closed: int = 1
    n_mvr: int = 0


def calc_hpr_capital_cost(
    units: Sequence[HPRCostUnit],
    args: HeatPumpTargetInputs,
) -> Value:
    """Return the installed capital cost of the heat-pump machines.

    This is the sum of :func:`calc_hpr_machine_capital_costs`.
    """
    return Value(sum(calc_hpr_machine_capital_costs(units, args)), "$")


def calc_hpr_machine_capital_costs(
    units: Sequence[HPRCostUnit],
    args: HeatPumpTargetInputs,
) -> tuple[float, ...]:
    """Return the installed capital cost of each heat-pump machine, in $.

    Each machine costs::

        F_inst * C_eq * (Q_cap / 1 MW)^exp
            * (fixed_share + stage_share * (n_closed + n_mvr)) * f_T

    with ``f_T = 1 + temp_factor * max(0, T_hot_max - temp_base) / 100 K``.
    The machines are costed separately, and a machine with no capacity costs
    nothing.
    """
    costs = []
    for unit in units:
        Q_mw = max(float(unit.Q_cap), 0.0) / 1000.0
        if Q_mw <= 0.0:
            costs.append(0.0)
            continue
        stage_factor = float(args.hpr_cost_fixed_share) + float(
            args.hpr_cost_stage_share
        ) * (int(unit.n_closed) + int(unit.n_mvr))
        f_T = (
            1.0
            + float(args.hpr_cost_temp_factor)
            * max(0.0, float(unit.T_hot_max) - float(args.hpr_cost_temp_base))
            / 100.0
        )
        costs.append(
            float(args.hpr_installation_factor)
            * float(args.hpr_equipment_cost)
            * Q_mw ** float(args.hpr_cost_exp)
            * stage_factor
            * f_T
        )
    return tuple(costs)


def _single_cost_unit(
    state: HPRParsedState,
    hpr_streams: StreamCollection,
    work_arr: np.ndarray | None,
    args: HeatPumpTargetInputs,
) -> list[HPRCostUnit]:
    """Treat all the stages as one machine (a cascade).

    Each closed stage has its own compressor, so the stage count comes from
    the solved stage works: a cascade with ``n_cond`` condenser and
    ``n_evap`` evaporator levels has ``n_cond + n_evap - 1`` stages, more
    than its ``n_cond`` condensing temperatures.
    """
    hot = hpr_streams.get_hot_utility_streams()
    Q_cap = (
        max(float(hot.sum_stream_attribute("heat_flow", idx=args.period_idx)), 0.0)
        if len(hot)
        else 0.0
    )
    T_cond = np.asarray(state.T_cond, dtype=float).ravel()
    n_stages = (
        np.asarray(work_arr, dtype=float).size if work_arr is not None else 0
    ) or T_cond.size
    return [
        HPRCostUnit(
            Q_cap=Q_cap,
            T_hot_max=float(T_cond.max()) if T_cond.size else 0.0,
            n_closed=max(int(n_stages), 1),
        )
    ]


def calc_simulated_hpr_annualized_costs(
    *,
    work: float,
    Q_ext_heat: float,
    Q_cooling_water: float,
    Q_refrigeration: float,
    cost_units: Sequence[HPRCostUnit],
    penalty_weight: float,
    args: HeatPumpTargetInputs,
) -> SimulatedHPRAnnualizedCostAccounting:
    """Return unit-aware annualized cost accounting for simulated HPR candidates.

    The heat pump's installed capital is annualised unless
    ``args.hpr_capital_recovery`` is off. The capital of the default hot
    utility and refrigeration capacity the design still needs is annualised
    too unless ``args.utility_capital_recovery`` is off, so a heat pump that
    displaces utility capacity is credited for it.
    """
    annual_hours = max(float(args.annual_op_time), 0.0)
    ele_price = max(float(args.ele_price), 0.0)
    heat_price = ele_price * max(float(args.heat_to_power_ratio), 0.0)
    cw_price = ele_price * max(float(args.cooling_water_to_power_ratio), 0.0)
    ref_price = ele_price * max(float(args.refrigeration_to_power_ratio), 0.0)

    operating_cost = (
        compute_annual_energy_cost(work, ele_price, annual_hours)
        + compute_annual_energy_cost(Q_ext_heat, heat_price, annual_hours)
        + compute_annual_energy_cost(Q_cooling_water, cw_price, annual_hours)
        + compute_annual_energy_cost(Q_refrigeration, ref_price, annual_hours)
    ).to("$/y")

    machine_capital = calc_hpr_machine_capital_costs(cost_units, args)
    machine_annualized_capital = tuple(
        float(
            compute_annual_capital_cost(
                Value(capital, "$"), args.discount_rate, args.serv_life
            )
            .to("$/y")
            .value
        )
        if getattr(args, "hpr_capital_recovery", True)
        else 0.0
        for capital in machine_capital
    )
    capital_cost = Value(sum(machine_capital), "$")
    annualized_capital = Value(sum(machine_annualized_capital), "$/y")
    # The hot utility and refrigeration plant are separate assets, each sized
    # for its own peak, so they are annualised and reported separately.
    hot_utility_annualized_capital = _annualized_utility_capital(
        Q_ext_heat, getattr(args, "hot_utility_capital_cost", 0.0), args
    )
    refrigeration_annualized_capital = _annualized_utility_capital(
        Q_refrigeration, getattr(args, "refrigeration_capital_cost", 0.0), args
    )
    utility_annualized_capital = (
        hot_utility_annualized_capital + refrigeration_annualized_capital
    ).to("$/y")
    total_annualized = (
        operating_cost + annualized_capital + utility_annualized_capital
    ).to("$/y")
    # The dimensionless penalty is priced at the no-heat-pump annual cost of
    # the targeted service, so it is in $/y on the same scale as real costs.
    feasibility_penalty = Value(
        float(penalty_weight) * hpr_penalty_cost_scale(args), "$/y"
    )

    return SimulatedHPRAnnualizedCostAccounting(
        hpr_operating_cost=operating_cost,
        hpr_capital_cost=capital_cost,
        hpr_annualized_capital_cost=annualized_capital,
        hpr_hot_utility_annualized_capital_cost=hot_utility_annualized_capital,
        hpr_refrigeration_annualized_capital_cost=refrigeration_annualized_capital,
        hpr_utility_annualized_capital_cost=utility_annualized_capital,
        hpr_total_annualized_cost=total_annualized,
        feasibility_penalty=feasibility_penalty,
        hpr_machine_capital_costs=machine_capital,
        hpr_machine_annualized_capital_costs=machine_annualized_capital,
    )


def _annualized_utility_capital(
    duty: float,
    unit_capital_cost: float,
    args: HeatPumpTargetInputs,
) -> Value:
    """Annualise the installed capital of one default-utility capacity."""
    if not getattr(args, "utility_capital_recovery", True):
        return Value(0.0, "$/y")
    capital = Value(
        max(float(duty), 0.0) * max(float(unit_capital_cost), 0.0),
        "$",
    )
    return compute_annual_capital_cost(capital, args.discount_rate, args.serv_life).to(
        "$/y"
    )


def refrigeration_allowance(args: HeatPumpTargetInputs) -> float:
    """Return the default refrigeration the untargeted cooling needs on its own.

    With a load fraction below 1, refrigeration targets the coldest part of the
    cooling. The warmer rest may itself need some default refrigeration (when
    it is below the cooling-water level); that is not a shortfall of the
    design, so it is not penalised. Computed once and stored on ``args``.
    """
    if getattr(args, "is_heat_pumping", True):
        return 0.0
    cached = getattr(args, "refrigeration_allowance", None)
    if cached is not None:
        return float(cached)
    streams = getattr(args, "untargeted_cooling_streams", None)
    allowance = 0.0
    if streams is not None and len(streams):
        allowance = float(
            _cascade_air_duties(
                streams,
                StreamCollection(),
                args,
                _ambient_source_temperature(args),
                _ambient_sink_temperature(args),
                _cooling_water_sink_temperature(args),
            ).Q_ext_bottom
        )
    try:
        args.refrigeration_allowance = allowance
    except AttributeError, TypeError, ValueError:
        pass
    return allowance


def _cascade_air_duties(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    args: HeatPumpTargetInputs,
    T_air_source: float,
    T_air_sink: float,
    T_cooling_water: float | None = None,
) -> CascadeWithAir:
    """Cascade the streams and let air, then any cooling water, cover what they can."""
    if not len(hot_streams) and not len(cold_streams):
        return CascadeWithAir(0.0, 0.0, 0.0, 0.0)
    pt = _cascade.get_process_heat_cascade(
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        is_shifted=True,
        period_idx=args.period_idx,
    )
    return cascade_with_air(
        np.asarray(pt[ProblemTableLabel.T], dtype=float),
        np.asarray(pt[ProblemTableLabel.H_NET], dtype=float),
        T_air_source=T_air_source,
        T_air_sink=T_air_sink,
        T_cooling_water=T_cooling_water,
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
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Shared Carnot-family accounting, plotting, and result assembly."""
    hpr_streams = _streams.get_carnot_hpr_cycle_streams(
        state.T_cond,
        Q_cond_total,
        state.T_evap,
        Q_evap_total,
        args,
    )
    T_air_source = _ambient_source_temperature(args)
    T_air_sink = _ambient_sink_temperature(args)
    cond = _cascade_air_duties(
        hpr_streams.get_hot_utility_streams(),
        args.bckgrd_cold_streams,
        args,
        T_air_source,
        T_air_sink,
    )
    evap = _cascade_air_duties(
        args.bckgrd_hot_streams,
        hpr_streams.get_cold_utility_streams(),
        args,
        T_air_source,
        T_air_sink,
        _cooling_water_sink_temperature(args),
    )
    Q_ext_heat, Q_ext_cold, penalty, obj = _build_hpr_accounting(
        work=float(w_net),
        Q_ext_heat=cond.Q_ext_top,
        Q_ext_cold=evap.Q_ext_bottom,
        Q_cooling_water=evap.Q_cooling_water,
        args=args,
        penalty_terms=[
            *_normalise_hpr_penalty_terms(penalty_terms),
            cond.Q_ext_bottom,
            evap.Q_ext_top,
        ],
        penalise_external_cold_when_refrigerating=True,
        refrigeration_allowance=refrigeration_allowance(args),
    )
    debug_figure = None
    if debug and artifact_mode is HPREvaluationMode.FINAL:
        debug_figure = _plotting.plot_multi_hp_profiles_from_results(
            args.T_hot,
            args.H_hot,
            args.T_cold,
            args.H_cold,
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
        feasibility_penalty=penalty,
        utility_tot=float(w_net + Q_ext_heat + Q_ext_cold + evap.Q_cooling_water),
        w_net=float(w_net),
        w_hpr=w_hpr,
        w_he=w_he,
        heat_recovery=heat_recovery,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold + evap.Q_cooling_water,
        Q_cooling_water=evap.Q_cooling_water,
        Q_refrigeration=Q_ext_cold,
        Q_amb_hot=cond.Q_air_source + evap.Q_air_source,
        Q_amb_cold=cond.Q_air_sink + evap.Q_air_sink,
        cop_h=cop_h,
        eta_he=eta_he,
        T_cond=state.T_cond,
        T_evap=state.T_evap,
        Q_cond=Q_cond_total if Q_cond is None else Q_cond,
        Q_evap=Q_evap_total if Q_evap is None else Q_evap,
        Q_cond_he=Q_cond_he,
        Q_evap_he=Q_evap_he,
        artifacts=(
            HPRThermoArtifacts(hpr_streams=hpr_streams, debug_figure=debug_figure)
            if artifact_mode is HPREvaluationMode.FINAL
            else None
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
    cost_units: Sequence[HPRCostUnit] | None = None,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Shared simulated-vapour accounting, plotting, and result assembly.

    ``cost_units`` lists the machines to cost; by default all the stages are
    one machine.
    """
    # Ambient air is a free utility in each cascade: it supplies heat below its
    # source temperature and takes heat above its sink temperature, only as
    # far as the cascade needs it.
    T_air_source = _ambient_source_temperature(args)
    T_air_sink = _ambient_sink_temperature(args)

    cond_hot_streams = hpr_streams.get_hot_utility_streams()
    cond_cold_streams = args.bckgrd_cold_streams
    cond = _cascade_air_duties(
        cond_hot_streams, cond_cold_streams, args, T_air_source, T_air_sink
    )
    Q_ext_heat = cond.Q_ext_top
    cond_wrong_side = cond.Q_ext_bottom

    evap_hot_streams = args.bckgrd_hot_streams
    evap_cold_streams = hpr_streams.get_cold_utility_streams()
    evap = _cascade_air_duties(
        evap_hot_streams,
        evap_cold_streams,
        args,
        T_air_source,
        T_air_sink,
        _cooling_water_sink_temperature(args),
    )
    Q_cooling_water = evap.Q_cooling_water
    Q_refrigeration = evap.Q_ext_bottom
    Q_ext_cold = Q_cooling_water + Q_refrigeration
    evap_wrong_side = evap.Q_ext_top
    Q_amb_hot = cond.Q_air_source + evap.Q_air_source
    Q_amb_cold = cond.Q_air_sink + evap.Q_air_sink
    if cost_units is None:
        cost_units = _single_cost_unit(state, hpr_streams, work_arr, args)
    all_penalty_terms = [
        *_normalise_hpr_penalty_terms(penalty_terms),
        cond_wrong_side,
        evap_wrong_side,
    ]
    if not getattr(args, "is_heat_pumping", True):
        # A refrigerator must serve the targeted coldest cooling. Cooling water
        # and air take the warmer rest for free; only default refrigeration
        # beyond what that rest needs on its own is unserved, so a zero-duty
        # refrigerator is never a valid design.
        all_penalty_terms.append(
            max(Q_refrigeration - refrigeration_allowance(args), 0.0)
        )
    penalty_weight = _cycle_penalty(
        args=args,
        cycle_penalty_terms=all_penalty_terms,
    )
    cost_accounting = calc_simulated_hpr_annualized_costs(
        work=float(work),
        Q_ext_heat=Q_ext_heat,
        Q_cooling_water=Q_cooling_water,
        Q_refrigeration=Q_refrigeration,
        cost_units=cost_units,
        penalty_weight=penalty_weight,
        args=args,
    )
    penalty = float(cost_accounting.feasibility_penalty.to("$/y").value)
    obj = float(cost_accounting.hpr_total_annualized_cost.to("$/y").value) + penalty

    debug_figure = None
    if debug and artifact_mode is HPREvaluationMode.FINAL:
        debug_figure = _plotting.plot_multi_hp_profiles_from_results(
            args.T_hot,
            args.H_hot,
            args.T_cold,
            args.H_cold,
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
        Q_cooling_water=Q_cooling_water,
        Q_refrigeration=Q_refrigeration,
        hpr_operating_cost=cost_accounting.hpr_operating_cost,
        hpr_capital_cost=cost_accounting.hpr_capital_cost,
        hpr_annualized_capital_cost=cost_accounting.hpr_annualized_capital_cost,
        hpr_machine_capital_costs=cost_accounting.hpr_machine_capital_costs,
        hpr_machine_annualized_capital_costs=(
            cost_accounting.hpr_machine_annualized_capital_costs
        ),
        hpr_hot_utility_annualized_capital_cost=(
            cost_accounting.hpr_hot_utility_annualized_capital_cost
        ),
        hpr_refrigeration_annualized_capital_cost=(
            cost_accounting.hpr_refrigeration_annualized_capital_cost
        ),
        hpr_utility_annualized_capital_cost=(
            cost_accounting.hpr_utility_annualized_capital_cost
        ),
        hpr_total_annualized_cost=cost_accounting.hpr_total_annualized_cost,
        feasibility_penalty=penalty,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        cop_h=float(cop_h),
        T_cond=state.T_cond,
        T_evap=state.T_evap,
        dT_subcool=dT_subcool,
        dT_superheat=dT_superheat,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        artifacts=(
            HPRThermoArtifacts(
                hpr_streams=hpr_streams,
                model=model,
                debug_figure=debug_figure,
            )
            if artifact_mode is HPREvaluationMode.FINAL
            else None
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


# Condensing temperatures within this margin of a refrigerant's critical
# temperature have no usable saturation state (the cycle's pressure lookup
# switches to the critical isochore there), so the search stays below it.
SUBCRITICAL_CONDENSING_MARGIN_K = 2.0


def condensing_temperature_search_range(
    args: HeatPumpTargetInputs,
) -> tuple[float, float]:
    """Return the ``(hottest, coldest)`` condensing temperatures to search, in degC.

    The range spans the heat-sink profile (``args.T_cold``) but is capped just
    below the highest critical temperature among the candidate refrigerants:
    a subcritical vapour-compression condenser cannot operate above it, and
    every candidate there fails in CoolProp. Without the cap the search can
    spend its whole budget in that infeasible region. The range is returned
    unchanged for the TESPy backend (which solves its own cycle states), when
    the refrigerants' critical points are unknown, or when the cap would leave
    nothing to search.
    """
    t_hot, t_cold = float(args.T_cold[0]), float(args.T_cold[-1])
    if getattr(args, "simulation_backend", "coolprop") == "tespy":
        return t_hot, t_cold
    try:
        refrigerants = list(getattr(args, "refrigerant_ls", None) or ["water"])
        t_crit = max(float(_coolprop.PropsSI("Tcrit", ref)) for ref in refrigerants)
    except ValueError, TypeError:
        return t_hot, t_cold
    ceiling = t_crit - 273.15 - SUBCRITICAL_CONDENSING_MARGIN_K
    if not np.isfinite(ceiling) or ceiling <= t_cold or ceiling >= t_hot:
        return t_hot, t_cold
    return ceiling, t_cold


# A design whose useful duty is below this fraction of Q_hpr_target is treated
# as no heat pump, so targeting reports "no beneficial heat pump" instead of
# returning a vanishing design.
NEGLIGIBLE_USEFUL_DUTY_FRACTION = 1e-3


def is_negligible_useful_duty(duty: float, args: HeatPumpTargetInputs) -> bool:
    """Return whether ``duty`` is too small to count as a heat pump."""
    target = abs(float(getattr(args, "Q_hpr_target", 0.0) or 0.0))
    return float(duty) <= NEGLIGIBLE_USEFUL_DUTY_FRACTION * target


def cap_stage_condensing_temperatures(
    T_cond: np.ndarray,
    args: HeatPumpTargetInputs,
) -> np.ndarray:
    """Cap each stage's condensing temperature below its own refrigerant's Tcrit.

    ``condensing_temperature_search_range`` caps the whole search at the
    highest critical temperature among the refrigerants. A stage with a
    lower-Tcrit refrigerant (R134a beside water, say) could still be decoded
    above its own critical point, where every candidate fails. Stage ``i`` uses
    ``args.refrigerant_ls[i]``; the result keeps the stages in descending order.
    """
    T_cond = np.asarray(T_cond, dtype=float)
    if getattr(args, "simulation_backend", "coolprop") == "tespy":
        return T_cond
    refrigerants = list(getattr(args, "refrigerant_ls", None) or [])
    if not refrigerants:
        return T_cond
    ceilings = np.full(T_cond.shape, np.inf)
    for index in range(T_cond.size):
        refrigerant = refrigerants[min(index, len(refrigerants) - 1)]
        try:
            t_crit = float(_coolprop.PropsSI("Tcrit", refrigerant)) - 273.15
        except ValueError, TypeError:
            continue
        ceilings[index] = t_crit - SUBCRITICAL_CONDENSING_MARGIN_K
    capped = np.minimum(T_cond, ceilings)
    return np.sort(capped)[::-1]


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
