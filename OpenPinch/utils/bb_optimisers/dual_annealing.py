"""Dual-annealing multi-start backend."""

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Callable

import numpy as np
from scipy.optimize import dual_annealing

from .common import (
    _cluster_candidates,
    _collect_candidates_in_parallel,
    _polish_candidates,
    _postprocess_candidates,
)


def _get_da_multiminima_in_parallel(
    func,
    bounds,
    x0_ls=None,
    args=(),
    constraints=(),
    n_runs=os.cpu_count(),
    maxiter=300,
    seed=0,
    initial_temp=5230.0,
    restart_temp_ratio=2e-5,
    visit=2.62,
    accept=-5.0,
    maxfun=1_000_000,
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
):
    """Return deduplicated local minima from multi-start dual annealing."""
    bounds = np.asarray(bounds, dtype=float)
    all_x, all_f = _collect_da_candidates(
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        n_runs=n_runs,
        maxiter=maxiter,
        seed=seed,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
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


def _collect_da_candidates(
    func: Callable,
    bounds: tuple,
    x0_ls: np.ndarray,
    args: dict,
    n_runs: int,
    maxiter: int,
    seed: int,
    initial_temp: float,
    restart_temp_ratio: float,
    visit: float,
    accept: float,
    maxfun: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect candidate minima from multiple dual-annealing runs."""
    run_fn = partial(
        _run_da_single,
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        maxiter=maxiter,
        seed=seed,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
    )
    return _collect_candidates_in_parallel(
        run_fn=run_fn,
        n_runs=n_runs,
        pool_executor_cls=ProcessPoolExecutor,
        broken_pool_exc=BrokenProcessPool,
    )


def _run_da_single(
    run: int,
    func: Callable,
    bounds: tuple,
    x0_ls: np.ndarray,
    args: dict,
    maxiter: int,
    seed: int,
    initial_temp: float,
    restart_temp_ratio: float,
    visit: float,
    accept: float,
    maxfun: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Execute one dual-annealing run and record callback minima."""
    run_minima_x = []
    run_minima_f = []
    x0 = x0_ls[run % np.shape(x0_ls)[0]] if x0_ls is not None else None

    def callback(x, f, context):
        run_minima_x.append(np.array(x, dtype=float))
        run_minima_f.append(float(f))
        return False

    res = dual_annealing(
        func=func,
        x0=x0,
        bounds=bounds,
        args=args,
        maxiter=maxiter,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
        seed=seed + run,
        no_local_search=True,
        callback=callback,
    )

    run_minima_x.append(np.array(res.x, dtype=float))
    run_minima_f.append(float(res.fun))
    return run_minima_x, run_minima_f
