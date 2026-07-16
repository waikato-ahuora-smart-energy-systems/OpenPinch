"""Private multiperiod execution helpers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

from ....lib.schemas.io import TargetOutput


def solve_default_target_for_period(
    problem_inputs: dict[str, Any],
    project_name: str,
    period_id: str,
) -> dict[str, Any]:
    """Solve one isolated default target run for an executor worker."""
    from ...pinch_problem import PinchProblem

    problem = PinchProblem(source=problem_inputs, project_name=project_name)
    result = problem.target(period_id=period_id)
    return result.model_dump(mode="python")


def solve_periods_parallel(
    *,
    problem_inputs: dict[str, Any],
    project_name: str,
    period_ids: list[str],
    backend: str,
    max_workers: int | None,
) -> dict[str, TargetOutput]:
    """Execute isolated period solves and key results by requested identity."""
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    results: dict[str, TargetOutput] = {}
    with executor_cls(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                solve_default_target_for_period,
                problem_inputs,
                project_name,
                period_id,
            ): period_id
            for period_id in period_ids
        }
        for future in as_completed(futures):
            period_id = futures[future]
            results[period_id] = TargetOutput.model_validate(future.result())
    return results
