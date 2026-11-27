"""A shared multiperiod heat pump designed once and operated in each period.

Design-and-operate (review finding 19, decided 2026-11-28): an outer search
sizes the installed machine and each period runs it at its cheapest
operation within those sizes. It starts from each period solved alone with
the machine each component's peak, so it is never dearer than that.
"""

from __future__ import annotations

import numpy as np

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
    HPRPeriodCase,
    HPRSearchBudget,
    MultiPeriodHPRTargetInputs,
)
from OpenPinch.domain.configuration import Configuration

_SCALES = {"base": 1.0, "peak": 1.6}


def _period(scale: float):
    args = hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0 * scale,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=scale * np.array([0.0, -50.0, -100.0]),
        H_cold=scale * np.array([100.0, 50.0, 0.0]),
        config=Configuration(options={"HPR_N_COND": 1, "HPR_N_EVAP": 1}),
    )
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    args.search_budget = HPRSearchBudget(maximum_evaluations=60)
    return args


def _inputs() -> MultiPeriodHPRTargetInputs:
    cases = [
        HPRPeriodCase(period_id=name, period_idx=idx, weight=0.5, args=_period(scale))
        for idx, (name, scale) in enumerate(_SCALES.items())
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
    args = _inputs()
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
