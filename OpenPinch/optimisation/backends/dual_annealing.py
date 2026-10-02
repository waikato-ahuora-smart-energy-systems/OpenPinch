"""Dual-annealing multi-start backend."""

import os
from typing import Callable

import numpy as np
from scipy.optimize import dual_annealing

from ._multistart import run_multistart


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
    return run_multistart(
        _run_da_single,
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        constraints=constraints,
        n_runs=n_runs,
        cluster_tol=cluster_tol,
        max_minima=max_minima,
        local_method=local_method,
        maxiter=maxiter,
        seed=seed,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
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
    """Execute one dual-annealing run and record callback minima.

    Coordinates with fixed bounds (lower == upper) are held at that value and
    only the free ones are annealed: scipy rejects zero-width bounds.
    """
    run_minima_x = []
    run_minima_f = []
    x0 = x0_ls[run % np.shape(x0_ls)[0]] if x0_ls is not None else None

    bounds_array = np.asarray(bounds, dtype=float)
    fixed = bounds_array[:, 0] >= bounds_array[:, 1]
    template = bounds_array[:, 0].copy()
    free = ~fixed

    def expand(x_free) -> np.ndarray:
        full = template.copy()
        full[free] = np.asarray(x_free, dtype=float)
        return full

    if not free.any():
        value = float(func(template.copy(), *tuple(args or ())))
        return [template.copy()], [value]

    def free_func(x_free, *func_args):
        return func(expand(x_free), *func_args)

    def callback(x, f, context):
        run_minima_x.append(expand(x))
        run_minima_f.append(float(f))
        return False

    res = dual_annealing(
        func=free_func,
        x0=None if x0 is None else np.asarray(x0, dtype=float)[free],
        bounds=bounds_array[free],
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

    run_minima_x.append(expand(res.x))
    run_minima_f.append(float(res.fun))
    return run_minima_x, run_minima_f
