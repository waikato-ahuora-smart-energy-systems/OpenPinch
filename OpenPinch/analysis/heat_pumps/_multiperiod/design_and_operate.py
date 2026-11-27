"""Design a shared heat pump once and operate it in each period.

Design-and-operate for shared multiperiod designs (review finding 19,
decided 2026-11-28). The search starts from each period solved alone, with
the machine each component's peak, which every period's own operation fits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    MultiPeriodHPRTargetInputs,
)
from ..optimisation_adapter import (
    aggregate_hpr_period_results,
    evaluate_hpr_candidate,
    run_hpr_candidate_search,
)
from .aggregation import _with_installed_capital, selected_period_case

__all__ = ["design_and_operate"]


def design_and_operate(
    *,
    f_obj: Callable,
    x0_ls: Sequence[float] | np.ndarray | None,
    bnds: Sequence[Sequence[float]],
    args: MultiPeriodHPRTargetInputs,
) -> HPRBackendResult:
    """Return the selected period's result of the designed and operated machine.

    ``period_outputs`` hold each period's operation and ``weighted_output``
    the year's figures with the installed machine's capital.
    """
    cases = args.period_cases
    weights = np.asarray([case.weight for case in cases], dtype=float)
    outputs = {
        str(case.period_id): _solve_alone(f_obj, x0_ls, bnds, case.args)
        for case in cases
    }
    weighted, objective = aggregate_hpr_period_results(
        outputs, weights, idle_weight=args.idle_period_weight
    )
    selected = _with_installed_capital(
        outputs[str(selected_period_case(args).period_id)], weighted
    )
    return selected.with_updates(
        obj=objective,
        period_outputs=outputs,
        weighted_output=weighted,
        period_ids=[str(case.period_id) for case in cases],
        period_weights=[float(case.weight) for case in cases],
    )


def _solve_alone(f_obj, x0_ls, bnds, args: HeatPumpTargetInputs) -> HPRBackendResult:
    """One period's own best design."""
    best = min(
        run_hpr_candidate_search(
            objective=f_obj, initial_points=x0_ls, bounds=bnds, args=args
        ),
        key=lambda candidate: candidate.objective,
    )
    return evaluate_hpr_candidate(objective=f_obj, point=best.point, args=args)
