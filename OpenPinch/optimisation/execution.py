"""Execution primitives shared by scalar optimisation backends."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from multiprocessing.reduction import ForkingPickler
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Raised while a pool starts or tasks are submitted: the objective has not run.
_POOL_STARTUP_ERRORS = (OSError, RuntimeError)


@runtime_checkable
class TaskScopedObjective(Protocol):
    """Objective whose mutable state is scoped to one optimiser task.

    For a batch of ``count`` tasks the optimiser calls ``for_task(index,
    count)`` in the parent, runs task ``index`` with that copy (serially or in
    a worker process), and finally passes every copy's ``task_report()`` to
    ``merge_task_reports`` in task order. Serial and pooled runs therefore see
    identical objective state.
    """

    def __call__(self, x, *args) -> float: ...

    def for_task(self, index: int, count: int) -> TaskScopedObjective: ...

    def task_report(self) -> object: ...

    def merge_task_reports(self, reports: Sequence[object]) -> None: ...


def _is_picklable(obj: object) -> bool:
    """Return whether ``obj`` can be sent to a process-pool worker."""
    try:
        ForkingPickler.dumps(obj)
    except Exception:
        # Whatever pickling raises, the object cannot reach a worker.
        return False
    return True


def _map_in_process_pool(
    fn: Callable,
    items: Sequence,
    *,
    max_workers: int,
    pool_executor_cls=ProcessPoolExecutor,
    broken_pool_exc=BrokenProcessPool,
) -> list:
    """Return ``[fn(item) for item in items]``, in a process pool when possible.

    The pool is skipped when one worker suffices or ``fn`` cannot be pickled,
    and abandoned for a serial rerun when it cannot start or breaks. An
    exception raised by ``fn`` itself propagates unchanged.
    """
    items = list(items)
    workers = min(int(max_workers), len(items))
    if workers <= 1:
        return [fn(item) for item in items]
    # Items of one batch share their types, so the first stands for the rest.
    if not _is_picklable((fn, items[0])):
        logger.debug("Optimisation task is not picklable; running serially.")
        return [fn(item) for item in items]

    started = False
    try:
        with pool_executor_cls(max_workers=workers) as pool:
            pending = pool.map(fn, items)
            started = True
            return list(pending)
    except broken_pool_exc as exc:
        reason = exc
    except _POOL_STARTUP_ERRORS as exc:
        if started:
            raise
        reason = exc
    logger.warning("Process pool unavailable (%s); running tasks serially.", reason)
    return [fn(item) for item in items]


def _call_with_task_objective(task: Callable, item_and_func: tuple) -> object:
    item, func = item_and_func
    return task(item, func=func)


def _call_with_scoped_objective(task: Callable, item_and_func: tuple) -> tuple:
    item, func = item_and_func
    return task(item, func=func), func.task_report()


def _run_objective_tasks(
    task: Callable,
    items: Iterable,
    *,
    func: Callable,
    max_workers: int,
    pool_executor_cls=ProcessPoolExecutor,
    broken_pool_exc=BrokenProcessPool,
) -> list:
    """Return ``task(item, func=...)`` for each item, scoping ``func`` per task."""
    items = list(items)
    if not isinstance(func, TaskScopedObjective):
        return _map_in_process_pool(
            partial(_call_with_task_objective, task),
            [(item, func) for item in items],
            max_workers=max_workers,
            pool_executor_cls=pool_executor_cls,
            broken_pool_exc=broken_pool_exc,
        )

    count = len(items)
    outputs = _map_in_process_pool(
        partial(_call_with_scoped_objective, task),
        [(item, func.for_task(index, count)) for index, item in enumerate(items)],
        max_workers=max_workers,
        pool_executor_cls=pool_executor_cls,
        broken_pool_exc=broken_pool_exc,
    )
    func.merge_task_reports([report for _, report in outputs])
    return [result for result, _ in outputs]


def _collect_candidates_in_parallel(
    run_fn: Callable[..., tuple[list[np.ndarray], list[float]]],
    n_runs: int,
    pool_executor_cls=ProcessPoolExecutor,
    broken_pool_exc=BrokenProcessPool,
    func: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute backend runs in parallel, falling back for process limitations.

    With ``func`` given, each run is called as ``run_fn(run, func=...)`` so a
    :class:`TaskScopedObjective` gets one scoped copy per run.
    """
    all_x: list[np.ndarray] = []
    all_f: list[float] = []
    run_count = max(1, int(n_runs))

    if func is None:
        run_results = _map_in_process_pool(
            run_fn,
            range(run_count),
            max_workers=run_count,
            pool_executor_cls=pool_executor_cls,
            broken_pool_exc=broken_pool_exc,
        )
    else:
        run_results = _run_objective_tasks(
            run_fn,
            range(run_count),
            func=func,
            max_workers=run_count,
            pool_executor_cls=pool_executor_cls,
            broken_pool_exc=broken_pool_exc,
        )

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


__all__ = ["TaskScopedObjective", "_collect_candidates_in_parallel"]
