"""Design a shared heat pump once and operate it in each period.

Design-and-operate for shared multiperiod designs (review finding 19,
decided 2026-11-28). The installed machine starts as each period solved
alone, each component at its peak, which every period's own operation
fits. Each period then runs that machine at its cheapest operation within
its sizes (``operate_installed_machine``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ....contracts.hpr import (
    ZERO_USEFUL_DUTY_REASON,
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPREvaluationMode,
    MultiPeriodHPRTargetInputs,
)
from ..common.capital import installed_machine
from ..optimisation_adapter import (
    aggregate_hpr_period_results,
    evaluate_hpr_candidate,
    run_hpr_candidate_search,
)
from .aggregation import _as_idle_period, _with_installed_capital, selected_period_case
from .operation import operate_installed_machine

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
    alone = [_solve_alone(f_obj, x0_ls, bnds, case.args) for case in cases]
    machines = tuple(
        installed_machine(periods)
        for periods in zip(*(result.hpr_machine_components for _, result in alone))
    )
    points = [point for point, _ in alone]
    operated = [
        operate_installed_machine(
            machines,
            objective=f_obj,
            initial_points=_starts(index, points, x0_ls),
            bounds=bnds,
            args=case.args,
        )
        for index, case in enumerate(cases)
    ]
    if not any(operation.success for operation in operated):
        # A machine no period can run is no heat pump.
        return HPRBackendResult.failure(reason=ZERO_USEFUL_DUTY_REASON)
    outputs = {
        str(case.period_id): _or_idle(operation, f_obj, bnds, case.args)
        for case, operation in zip(cases, operated, strict=True)
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


def _or_idle(
    operated: HPRBackendResult, f_obj, bnds, args: HeatPumpTargetInputs
) -> HPRBackendResult:
    """A period's operation, or the machine off there when none fits.

    The machine off is the cycle at its lower bounds, which carry no duty,
    priced as an idle period: its utilities serve the whole load.
    """
    if operated.success:
        return operated
    idle = evaluate_hpr_candidate(
        objective=f_obj,
        point=np.asarray(bnds, dtype=float)[:, 0],
        args=args,
        artifact_mode=HPREvaluationMode.SEARCH,
    )
    return _as_idle_period(idle)


def _solve_alone(
    f_obj, x0_ls, bnds, args: HeatPumpTargetInputs
) -> tuple[np.ndarray, HPRBackendResult]:
    """One period's own best design and its point."""
    best = min(
        run_hpr_candidate_search(
            objective=f_obj, initial_points=x0_ls, bounds=bnds, args=args
        ),
        key=lambda candidate: candidate.objective,
    )
    point = np.asarray(best.point, dtype=float)
    return point, evaluate_hpr_candidate(objective=f_obj, point=point, args=args)


def _starts(index: int, points: Sequence[np.ndarray], x0_ls) -> np.ndarray:
    """A period's operation starts from its own design, then the other
    periods' designs (each fits the installed machine), then the warm
    starts."""
    rows = [np.atleast_2d(points[index])] + [
        np.atleast_2d(point) for other, point in enumerate(points) if other != index
    ]
    if x0_ls is not None:
        rows.append(np.atleast_2d(np.asarray(x0_ls, dtype=float)))
    return np.vstack(rows)
