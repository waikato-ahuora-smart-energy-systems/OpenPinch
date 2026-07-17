"""Private multiperiod execution helpers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from ....contracts.output import TargetOutput

if TYPE_CHECKING:
    from ...problem import PinchProblem


def target_all_periods(
    problem: "PinchProblem",
    *,
    parallel: bool | str,
    max_workers: int | None,
    preserve_cached_results: bool,
) -> dict[str, TargetOutput]:
    """Run the default target once per period while restoring parent state."""
    period_ids = list(problem.period_ids)
    if not period_ids:
        raise ValueError("This problem has no canonical period_ids to target.")

    previous_results = problem._results
    previous_recording_state = problem._suspend_target_run_recording
    try:
        problem._suspend_target_run_recording = True
        if parallel in (False, None):
            results = {
                period_id: problem._solve_target_for_period(period_id)
                for period_id in period_ids
            }
        else:
            results = target_all_periods_parallel(
                problem,
                period_ids=period_ids,
                backend="thread" if parallel == "thread" else "process",
                max_workers=max_workers,
            )
        return order_period_results(
            period_ids=period_ids,
            results_by_requested_period=results,
        )
    finally:
        problem._suspend_target_run_recording = previous_recording_state
        if preserve_cached_results:
            problem._results = previous_results


def solve_target_for_period(
    problem: "PinchProblem",
    period_id: str,
) -> TargetOutput:
    """Solve and detach one period result from its parent cache."""
    result = problem.target(period_id=period_id)
    return TargetOutput.model_validate(result.model_dump(mode="python"))


def period_result_key(
    result: TargetOutput,
    *,
    requested_period_id: str,
) -> str:
    """Resolve the canonical identity for one period result."""
    return (
        str(result.period_id) if result.period_id is not None else requested_period_id
    )


def order_period_results(
    *,
    period_ids: list[str],
    results_by_requested_period: dict[str, TargetOutput],
) -> dict[str, TargetOutput]:
    """Return period results in canonical input order."""
    return {
        period_result_key(
            results_by_requested_period[requested_period_id],
            requested_period_id=requested_period_id,
        ): results_by_requested_period[requested_period_id]
        for requested_period_id in period_ids
    }


def target_all_periods_parallel(
    problem: "PinchProblem",
    *,
    period_ids: list[str],
    backend: str,
    max_workers: int | None,
) -> dict[str, TargetOutput]:
    """Run isolated period solves through the selected executor backend."""
    return solve_periods_parallel(
        problem_inputs=problem.canonical_problem_json(),
        project_name=problem.project_name,
        period_ids=period_ids,
        backend=backend,
        max_workers=max_workers,
    )


def solve_default_target_for_period(
    problem_inputs: dict[str, Any],
    project_name: str,
    period_id: str,
) -> dict[str, Any]:
    """Solve one isolated default target run for an executor worker."""
    from ...problem import PinchProblem

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
