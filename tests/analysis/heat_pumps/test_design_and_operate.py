"""A shared multiperiod heat pump designed once and operated in each period.

Design-and-operate (review finding 19, decided 2026-11-28): an outer search
sizes the installed machine and each period runs it at its cheapest
operation within those sizes. It starts from each period solved alone with
the machine each component's peak, so it is never dearer than that.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps._multiperiod.design_and_operate as dao
import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
from OpenPinch.analysis.heat_pumps._multiperiod.design_and_operate import (
    design_and_operate,
)
from OpenPinch.analysis.heat_pumps.common import preprocessing as hp_pre
from OpenPinch.analysis.heat_pumps.common.capital import (
    installed_capital,
    installed_machine,
)
from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    evaluate_hpr_candidate,
    run_hpr_candidate_search,
)
from OpenPinch.contracts.hpr import (
    ZERO_USEFUL_DUTY_REASON,
    HPRBackendResult,
    HPREvaluationMode,
    HPRPeriodCase,
    HPRSearchBudget,
    MultiPeriodHPRTargetInputs,
)
from OpenPinch.domain.configuration import Configuration

_SCALES = {"base": 1.0, "peak": 1.6}


def _period(scale: float, evaluations: int, **options):
    args = hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0 * scale,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=scale * np.array([0.0, -50.0, -100.0]),
        H_cold=scale * np.array([100.0, 50.0, 0.0]),
        config=Configuration(options={"HPR_N_COND": 1, "HPR_N_EVAP": 1, **options}),
    )
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    # One evaluation takes about half a second under Pyodide, so each test
    # searches no more than it needs.
    args.search_budget = HPRSearchBudget(maximum_evaluations=evaluations)
    return args


def _inputs(
    evaluations: int,
    scales: dict[str, float] = _SCALES,
    weights: dict[str, float] | None = None,
    **options,
) -> MultiPeriodHPRTargetInputs:
    cases = [
        HPRPeriodCase(
            period_id=name,
            period_idx=idx,
            weight=0.5 if weights is None else weights[name],
            args=_period(scale, evaluations, **options),
        )
        for idx, (name, scale) in enumerate(scales.items())
    ]
    selected = cases[0].args
    return MultiPeriodHPRTargetInputs(
        search_budget=selected.search_budget,
        period_cases=cases,
        selected_period_id="base",
        selected_period_idx=0,
        hpr_type=selected.hpr_type,
        max_multi_start=1,
        bb_minimiser=selected.bb_minimiser,
    )


def _periods_alone_total(args: MultiPeriodHPRTargetInputs, x0, bounds) -> float:
    """Each period solved alone, with the machine each component's peak."""
    results = []
    for case in args.period_cases:
        best = min(
            run_hpr_candidate_search(
                objective=hp_cascade._compute_cascade_hp_system_obj,
                initial_points=x0,
                bounds=bounds,
                args=case.args,
            ),
            key=lambda candidate: candidate.objective,
        )
        results.append(
            evaluate_hpr_candidate(
                objective=hp_cascade._compute_cascade_hp_system_obj,
                point=best.point,
                args=case.args,
            )
        )
    operating = sum(
        case.weight * float(result.hpr_operating_cost.value)
        for case, result in zip(args.period_cases, results, strict=True)
    )
    (machine,) = (
        installed_machine(periods)
        for periods in zip(*(result.hpr_machine_components for result in results))
    )
    ratio = (
        results[0].hpr_machine_annualized_capital_costs[0]
        / results[0].hpr_machine_capital_costs[0]
    )
    return operating + ratio * installed_capital(machine)


def test_a_designed_and_operated_machine_is_never_dearer_than_periods_alone():
    args = _inputs(evaluations=6)
    x0, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)

    result = design_and_operate(
        f_obj=hp_cascade._compute_cascade_hp_system_obj,
        x0_ls=x0,
        bnds=bounds,
        args=args,
    )

    weighted = result.weighted_output
    bound = _periods_alone_total(args, x0, bounds)
    assert float(weighted.hpr_total_annualized_cost.value) <= bound * (1.0 + 1e-6)


def test_each_period_runs_the_installed_machine():
    # The base period's own machine is smaller than the peak's; run on the
    # installed machine it uses the spare area and costs less to run than
    # its own design. Running its own design instead, the two costs are
    # equal (4,805 $/y), which is what this test rules out. The saving needs
    # a search of 60 evaluations to find; at 30 or fewer it does not.
    args = _inputs(evaluations=60)
    x0, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)

    result = design_and_operate(
        f_obj=hp_cascade._compute_cascade_hp_system_obj,
        x0_ls=x0,
        bnds=bounds,
        args=args,
    )

    base = args.period_cases[0].args
    own = min(
        run_hpr_candidate_search(
            objective=hp_cascade._compute_cascade_hp_system_obj,
            initial_points=x0,
            bounds=bounds,
            args=base,
        ),
        key=lambda candidate: candidate.objective,
    )
    own = evaluate_hpr_candidate(
        objective=hp_cascade._compute_cascade_hp_system_obj, point=own.point, args=base
    )
    ran = result.period_outputs["base"]
    assert ran.hpr_operating_cost.value < own.hpr_operating_cost.value


def test_a_period_the_installed_machine_cannot_serve_runs_without_it():
    # A light period (5 % of the base load) cannot run the base's machine at
    # 0.9 of its volume flow (HPR_TURNDOWN_DUTY), so the machine is off
    # there and the period is priced on its utilities. It raised for want of
    # a cost breakdown, as the reviewer's three-period case did.
    args = _inputs(
        evaluations=6, scales={"base": 1.0, "light": 0.05}, HPR_TURNDOWN_DUTY=0.9
    )
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)

    result = design_and_operate(
        f_obj=hp_cascade._compute_cascade_hp_system_obj,
        x0_ls=None,
        bnds=bounds,
        args=args,
    )

    light = result.period_outputs["light"]
    assert light.w_net == 0.0
    assert light.hpr_operating_cost.value > 0.0


def _no_heat_pump_at_peak(
    x, args, debug=False, artifact_mode=HPREvaluationMode.FINAL
) -> HPRBackendResult:
    """The cycle objective, but the peak period's final evaluations deliver
    no useful duty, as when its own best design is no heat pump."""
    if args.Q_hpr_target > 60.0 and artifact_mode is HPREvaluationMode.FINAL:
        return HPRBackendResult.failure(reason=ZERO_USEFUL_DUTY_REASON)
    return hp_cascade._compute_cascade_hp_system_obj(
        x, args, debug=debug, artifact_mode=artifact_mode
    )


def test_a_period_whose_own_best_is_no_heat_pump_does_not_size_the_machine():
    # It raised sizing the machine from a result with no components (Codex
    # review on #568); the machine is sized from the periods that run one.
    args = _inputs(evaluations=6)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)

    result = design_and_operate(
        f_obj=_no_heat_pump_at_peak, x0_ls=None, bnds=bounds, args=args
    )

    assert result.success
    assert result.period_outputs["peak"].w_net == 0.0


def test_the_year_pays_for_the_machine_installed(monkeypatch):
    # Every period using half of each component: the year still pays for the
    # whole machine installed, not for the components the periods used
    # (69,423 against 113,132 $; Codex review on #568).
    args = _inputs(evaluations=6)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)
    seen = {}
    operate = dao.operate_installed_machine

    def half_used(machines, **kwargs):
        seen["machines"] = machines
        result = operate(machines, **kwargs)
        return result.with_updates(
            hpr_machine_components=tuple(_halved(m) for m in machines)
        )

    monkeypatch.setattr(dao, "operate_installed_machine", half_used)

    result = dao.design_and_operate(
        f_obj=hp_cascade._compute_cascade_hp_system_obj,
        x0_ls=None,
        bnds=bounds,
        args=args,
    )

    installed = sum(installed_capital(machine) for machine in seen["machines"])
    assert result.weighted_output.hpr_capital_cost.value == pytest.approx(installed)


def test_a_machine_that_costs_the_year_more_than_none_is_no_heat_pump():
    # The base period, 2 % of the year, pays for the machine on its own; the
    # light period cannot run it (HPR_TURNDOWN_DUTY 0.9). The year costs
    # about 10,000 $/y with it and about 3,000 without, and the machine was
    # returned (Codex review on #568).
    args = _inputs(
        evaluations=6,
        scales={"base": 1.0, "light": 0.05},
        weights={"base": 0.02, "light": 0.98},
        HPR_TURNDOWN_DUTY=0.9,
    )
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)

    result = design_and_operate(
        f_obj=hp_cascade._compute_cascade_hp_system_obj,
        x0_ls=None,
        bnds=bounds,
        args=args,
    )

    assert result.failure_reason == ZERO_USEFUL_DUTY_REASON


def test_the_design_vector_is_the_installed_sizes():
    # A shared target reports its design vector (notebook 10 checks it); a
    # designed and operated machine's design is its installed sizes, per
    # compressor its volume flow and shaft power, then its exchanger areas.
    args = _inputs(evaluations=6)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args.period_cases[0].args)

    result = design_and_operate(
        f_obj=hp_cascade._compute_cascade_hp_system_obj,
        x0_ls=None,
        bnds=bounds,
        args=args,
    )

    (machine,) = result.weighted_output.hpr_machine_components
    (compressor,) = machine.compressors
    np.testing.assert_allclose(
        result.design_vector,
        [compressor.volume_flow, compressor.shaft_power, *machine.exchanger_areas],
    )


def _halved(machine):
    return dataclasses.replace(
        machine,
        compressors=tuple(
            dataclasses.replace(
                c, volume_flow=c.volume_flow / 2, shaft_power=c.shaft_power / 2
            )
            for c in machine.compressors
        ),
        exchanger_areas=tuple(area / 2 for area in machine.exchanger_areas),
    )
