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
from ....domain.value import Value
from ..common.capital import MachineComponents, installed_capital, installed_machine
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
    idle_point: Callable | None = None,
) -> HPRBackendResult:
    """Return the selected period's result of the designed and operated machine.

    ``period_outputs`` hold each period's operation and ``weighted_output``
    the year's figures with the installed machine's capital. ``idle_point``
    (``x, args -> x``) turns a period's own design into the machine off;
    without it the machine off is the cycle at its lower bounds.
    """
    cases = args.period_cases
    weights = np.asarray([case.weight for case in cases], dtype=float)
    alone = [_solve_alone(f_obj, x0_ls, bnds, case.args) for case in cases]
    # Periods whose own best is no heat pump size nothing.
    sized = [
        result.hpr_machine_components
        for _, result in alone
        if result.success and result.hpr_machine_components
    ]
    if not sized:
        return HPRBackendResult.failure(reason=ZERO_USEFUL_DUTY_REASON)
    machines = tuple(installed_machine(periods) for periods in zip(*sized))
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
    off = [
        _off_point(point, bnds, case.args, idle_point)
        for point, case in zip(points, cases, strict=True)
    ]
    outputs = {
        str(case.period_id): _or_idle(operation, f_obj, point, case.args)
        for case, operation, point in zip(cases, operated, off, strict=True)
    }
    weighted, objective = aggregate_hpr_period_results(
        outputs, weights, idle_weight=args.idle_period_weight
    )
    weighted, objective = _priced_on(
        machines, weighted, objective, [result for _, result in alone]
    )
    _, without = aggregate_hpr_period_results(
        {
            str(case.period_id): _idle(f_obj, point, case.args)
            for case, point in zip(cases, off, strict=True)
        },
        weights,
        idle_weight=args.idle_period_weight,
    )
    if objective >= without:
        # The year costs no less with the machine than without one.
        return HPRBackendResult.failure(reason=ZERO_USEFUL_DUTY_REASON)
    selected = _with_installed_capital(
        outputs[str(selected_period_case(args).period_id)], weighted
    )
    return selected.with_updates(
        obj=objective,
        period_outputs=outputs,
        weighted_output=weighted,
        design_vector=np.asarray(
            [
                size
                for machine in machines
                for size in (
                    *(
                        value
                        for compressor in machine.compressors
                        for value in (compressor.volume_flow, compressor.shaft_power)
                    ),
                    *machine.exchanger_areas,
                )
            ],
            dtype=float,
        ),
        period_ids=[str(case.period_id) for case in cases],
        period_weights=[float(case.weight) for case in cases],
    )


def _priced_on(
    machines: Sequence[MachineComponents],
    weighted: HPRBackendResult,
    objective: float,
    alone: Sequence[HPRBackendResult],
) -> tuple[HPRBackendResult, float]:
    """The year's figures with the capital of the machines installed.

    The aggregation sizes each component for the most any period used; the
    year pays for what is installed, which can be more. Each machine is
    annualised at its periods' own ratio of annualised to installed capital.
    """
    capital = [installed_capital(machine) for machine in machines]
    annualized = [
        cost * _annualising_ratio(alone, index) for index, cost in enumerate(capital)
    ]
    before = weighted.hpr_annualized_capital_cost
    added = sum(annualized) - (0.0 if before is None else float(before.to("$/y").value))
    total = weighted.hpr_total_annualized_cost
    return (
        weighted.with_updates(
            hpr_machine_components=tuple(machines),
            hpr_machine_capital_costs=tuple(capital),
            hpr_machine_annualized_capital_costs=tuple(annualized),
            hpr_capital_cost=Value(sum(capital), "$"),
            hpr_annualized_capital_cost=Value(sum(annualized), "$/y"),
            hpr_total_annualized_cost=(
                None if total is None else total + Value(added, "$/y")
            ),
            obj=objective + added,
        ),
        objective + added,
    )


def _annualising_ratio(results: Sequence[HPRBackendResult], index: int) -> float:
    """A machine's annualised over installed capital, from its periods."""
    return next(
        (
            result.hpr_machine_annualized_capital_costs[index]
            / result.hpr_machine_capital_costs[index]
            for result in results
            if result.hpr_machine_capital_costs
            and result.hpr_machine_annualized_capital_costs
            and result.hpr_machine_capital_costs[index] > 0.0
        ),
        0.0,
    )


def _off_point(
    own: np.ndarray, bnds, args: HeatPumpTargetInputs, idle_point: Callable | None
) -> np.ndarray:
    """The point at which a period's machine is off: its own design with no
    duty (``idle_point``), else the cycle at its lower bounds."""
    if idle_point is not None:
        return idle_point(own, args)
    return np.asarray(bnds, dtype=float)[:, 0]


def _or_idle(
    operated: HPRBackendResult, f_obj, off: np.ndarray, args: HeatPumpTargetInputs
) -> HPRBackendResult:
    """A period's operation, or the machine off there when none fits."""
    if operated.success:
        return operated
    return _idle(f_obj, off, args)


def _idle(f_obj, off: np.ndarray, args: HeatPumpTargetInputs) -> HPRBackendResult:
    """A period with the machine off (the cycle at ``off``, which carries no
    duty), priced as an idle period on its utilities."""
    idle = evaluate_hpr_candidate(
        objective=f_obj,
        point=off,
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
