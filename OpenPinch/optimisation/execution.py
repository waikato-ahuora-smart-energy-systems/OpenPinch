"""Execution primitives shared by scalar optimisation backends."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable

import numpy as np


def _collect_candidates_in_parallel(
    run_fn: Callable[[int], tuple[list[np.ndarray], list[float]]],
    n_runs: int,
    pool_executor_cls=ProcessPoolExecutor,
    broken_pool_exc=BrokenProcessPool,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute backend runs in parallel, falling back for process limitations."""
    all_x: list[np.ndarray] = []
    all_f: list[float] = []
    run_count = max(1, int(n_runs))

    if run_count == 1:
        run_results = map(run_fn, range(run_count))
    else:
        try:
            with pool_executor_cls(max_workers=run_count) as pool:
                run_results = list(pool.map(run_fn, range(run_count)))
        except (PermissionError, OSError, RuntimeError, broken_pool_exc):
            run_results = map(run_fn, range(run_count))

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


__all__ = ["_collect_candidates_in_parallel"]
