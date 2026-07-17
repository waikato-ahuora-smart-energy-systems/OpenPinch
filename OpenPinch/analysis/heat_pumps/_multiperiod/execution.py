"""Execute one shared HPR design across all prepared periods."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetOutputs,
    HPRBackendResult,
    MultiPeriodHPRTargetInputs,
)
from ..optimisation_adapter import (
    evaluate_hpr_candidate,
    run_hpr_candidate_search,
    translate_hpr_output,
    translate_hpr_result,
)
from .aggregation import evaluate_multiperiod_candidate, selected_period_case
from .preparation import period_case_by_id
from .setup import get_multiperiod_hpr_optimisation_setup
from .state import _PreparedHPRPeriodCase


def get_multiperiod_hpr_targets(
    *,
    period_cases: list[_PreparedHPRPeriodCase],
    selected_period_id: str,
    selected_period_idx: int,
) -> HeatPumpTargetOutputs:
    """Solve a shared HPR design and project it into the output contract."""
    solver_cases = [case.solver_case for case in period_cases]
    selected_case = period_case_by_id(period_cases, selected_period_id).solver_case
    initial_points, bounds, period_objective = get_multiperiod_hpr_optimisation_setup(
        solver_cases,
        selected_case=selected_case,
    )
    args = MultiPeriodHPRTargetInputs(
        period_cases=solver_cases,
        selected_period_id=selected_period_id,
        selected_period_idx=selected_period_idx,
        hpr_type=selected_case.args.hpr_type,
        max_multi_start=selected_case.args.max_multi_start,
        bb_minimiser=selected_case.args.bb_minimiser,
        debug=selected_case.args.debug,
    )
    result = solve_hpr_multiperiod_placement(
        f_obj=period_objective,
        x0_ls=initial_points,
        bnds=bounds,
        args=args,
    )
    return translate_hpr_output(result)


def solve_hpr_multiperiod_placement(
    f_obj: Callable,
    x0_ls: list | np.ndarray | float | None,
    bnds: list,
    args: MultiPeriodHPRTargetInputs,
) -> HPRBackendResult:
    """Optimise one shared HPR vector against every prepared period case."""
    objective = partial(
        evaluate_multiperiod_candidate,
        period_objective=f_obj,
    )
    candidates = run_hpr_candidate_search(
        objective=objective,
        initial_points=x0_ls,
        bounds=bnds,
        args=args,
    )
    if not candidates:
        raise ValueError(
            "Multi-period heat pump and refrigeration targeting "
            f"({args.hpr_type}) failed to return any local minima."
        )

    selected_case = selected_period_case(args)
    for candidate in candidates:
        result = evaluate_hpr_candidate(
            objective=objective,
            point=candidate.point,
            args=args,
        )
        if result.success and np.isfinite(float(result.obj)):
            return translate_hpr_result(result, ambient_args=selected_case.args)

    raise ValueError(
        "Multi-period heat pump and refrigeration targeting "
        f"({args.hpr_type}) failed to return an optimal result."
    )


__all__ = ["get_multiperiod_hpr_targets", "solve_hpr_multiperiod_placement"]
