"""Shared scaffolding for unit-box surrogate backends (BO and RBF)."""

from typing import Callable, Optional

import numpy as np

from ..candidates import _evaluate_scalar_objective

ProposeFn = Callable[
    [int, np.ndarray, np.ndarray, np.random.Generator, Optional[np.ndarray], int],
    np.ndarray,
]


def as_seed_array(x0_ls) -> Optional[np.ndarray]:
    """Return ``x0_ls`` as a 2D float array (one start per row), or ``None``."""
    if x0_ls is None:
        return None
    x0_arr = np.asarray(x0_ls, dtype=float)
    if x0_arr.ndim == 1:
        x0_arr = x0_arr.reshape(1, -1)
    return x0_arr


def run_surrogate_loop(
    run: int,
    *,
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    maxiter: int,
    seed: int,
    maxfevals: int,
    n_init: Optional[int],
    propose: ProposeFn,
    evaluate: Callable = _evaluate_scalar_objective,
) -> tuple[list[np.ndarray], list[float]]:
    """Execute one surrogate-driven run in the unit box and record improving incumbents.

    ``propose(iter_idx, X_arr, y_arr, rng, best_u, n_dim)`` returns the next
    unit-box query point for iteration ``iter_idx`` given all observations so far.
    ``evaluate(func, x, args)`` scores one point in the original coordinates.
    """
    rng = np.random.default_rng(seed + run)
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    n_dim = lb.size

    if np.allclose(ub, lb):
        x_fixed = np.array(lb, dtype=float)
        f_fixed = evaluate(func, x_fixed, args)
        return [x_fixed], [f_fixed]

    span = np.where(ub > lb, ub - lb, 1.0)

    def to_unit(x):
        return np.clip((x - lb) / span, 0.0, 1.0)

    def from_unit(u):
        return np.clip(lb + u * span, lb, ub)

    x_seed = None
    if x0_ls is not None and np.shape(x0_ls)[0] > 0:
        x_seed = np.clip(np.array(x0_ls[run % np.shape(x0_ls)[0]], dtype=float), lb, ub)

    n_init_eff = int(n_init) if n_init is not None else max(6, 2 * n_dim + 2)
    n_init_eff = max(1, n_init_eff)
    maxfevals = int(maxfevals) if maxfevals is not None else int(maxiter) + n_init_eff

    X_unit = []
    y = []
    run_minima_x = []
    run_minima_f = []
    best_f = np.inf
    best_u = None
    eval_count = 0

    def evaluate_and_store(u):
        nonlocal best_f, best_u, eval_count
        x = from_unit(u)
        f = evaluate(func, x, args)
        eval_count += 1
        X_unit.append(np.array(u, dtype=float))
        y.append(float(f))
        if f < best_f:
            best_f = float(f)
            best_u = np.array(u, dtype=float)
            run_minima_x.append(np.array(x, dtype=float))
            run_minima_f.append(float(f))

    if x_seed is not None and eval_count < maxfevals:
        evaluate_and_store(to_unit(x_seed))

    while len(X_unit) < n_init_eff and eval_count < maxfevals:
        u = rng.uniform(0.0, 1.0, size=n_dim)
        evaluate_and_store(u)

    for iter_idx in range(int(maxiter)):
        if eval_count >= maxfevals or len(X_unit) == 0:
            break

        X_arr = np.asarray(X_unit, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        u_next = propose(iter_idx, X_arr, y_arr, rng, best_u, n_dim)
        evaluate_and_store(u_next)

    if not run_minima_x and len(X_unit) > 0:
        idx = int(np.argmin(y))
        x_best = from_unit(np.asarray(X_unit[idx], dtype=float))
        f_best = float(y[idx])
        run_minima_x.append(np.array(x_best, dtype=float))
        run_minima_f.append(float(f_best))
    elif run_minima_x:
        run_minima_x.append(np.array(run_minima_x[-1], dtype=float))
        run_minima_f.append(float(run_minima_f[-1]))

    return run_minima_x, run_minima_f
