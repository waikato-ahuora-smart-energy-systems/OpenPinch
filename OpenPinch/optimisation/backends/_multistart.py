"""Multistart plumbing shared by every backend.

Each backend supplies one ``run_single(run, *, func, bounds, x0_ls, args,
**options)`` function. :func:`run_multistart` runs it ``n_runs`` times in
parallel, then clusters and polishes the pooled candidates.
"""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Optional

import numpy as np

from ..candidates import (
    _cluster_candidates,
    _polish_candidates,
    _postprocess_candidates,
)
from ..execution import _collect_candidates_in_parallel

RunSingle = Callable[..., tuple[list[np.ndarray], list[float]]]


def as_seed_array(x0_ls) -> Optional[np.ndarray]:
    """Return ``x0_ls`` as a 2D float array (one start per row), or ``None``."""
    if x0_ls is None:
        return None
    x0_arr = np.asarray(x0_ls, dtype=float)
    if x0_arr.ndim == 1:
        x0_arr = x0_arr.reshape(1, -1)
    return x0_arr


def collect_candidates(
    run_single: RunSingle,
    *,
    func: Callable,
    bounds,
    x0_ls,
    args: tuple,
    n_runs: int,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``run_single`` for runs ``0..n_runs-1`` and pool their candidates.

    Run ``k`` starts from row ``k % len(x0_ls)`` of the seed starts.
    """
    run_fn = partial(
        run_single,
        func=func,
        bounds=bounds,
        x0_ls=as_seed_array(x0_ls),
        args=args,
        **options,
    )
    return _collect_candidates_in_parallel(
        run_fn=run_fn,
        n_runs=n_runs,
        pool_executor_cls=ProcessPoolExecutor,
        broken_pool_exc=BrokenProcessPool,
    )


def run_multistart(
    run_single: RunSingle,
    *,
    func: Callable,
    bounds,
    x0_ls,
    args: tuple,
    constraints,
    n_runs: int,
    cluster_tol: float,
    max_minima: int,
    local_method: str,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deduplicated, polished local minima from ``n_runs`` backend runs.

    ``options`` are passed unchanged to every ``run_single`` call.
    """
    bounds = np.asarray(bounds, dtype=float)
    all_x, all_f = collect_candidates(
        run_single,
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        n_runs=n_runs,
        **options,
    )
    return _postprocess_candidates(
        func=func,
        args=args,
        bounds=bounds,
        constraints=constraints,
        all_x=all_x,
        all_f=all_f,
        cluster_tol=cluster_tol,
        max_minima=max_minima,
        local_method=local_method,
        cluster_fn=_cluster_candidates,
        polish_fn=_polish_candidates,
    )


__all__ = ["as_seed_array", "collect_candidates", "run_multistart"]
