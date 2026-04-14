"""RBF-surrogate multi-start backend."""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import RBFInterpolator

from .common import (
    _cluster_candidates,
    _collect_candidates_in_parallel,
    _evaluate_scalar_objective,
    _polish_candidates,
    _postprocess_candidates,
)


def _get_rbf_surrogate_multiminima_in_parallel(
    func,
    bounds,
    x0_ls=None,
    args=(),
    constraints=(),
    n_runs=6,
    maxiter=300,
    seed=0,
    maxfun=1_000_000,
    maxfevals=None,
    n_init=None,
    n_candidates=2048,
    kernel="cubic",
    epsilon=1.0,
    smoothing=1e-8,
    degree=1,
    distance_tol=1e-6,
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
):
    """Return deduplicated local minima from multi-start RBF-surrogate search."""
    bounds = np.asarray(bounds, dtype=float)
    effective_maxfevals = maxfevals if maxfevals is not None else maxfun

    all_x, all_f = _collect_rbf_surrogate_candidates(
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        n_runs=n_runs,
        maxiter=maxiter,
        seed=seed,
        maxfevals=effective_maxfevals,
        n_init=n_init,
        n_candidates=n_candidates,
        kernel=kernel,
        epsilon=epsilon,
        smoothing=smoothing,
        degree=degree,
        distance_tol=distance_tol,
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


def _collect_rbf_surrogate_candidates(
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    n_runs: int,
    maxiter: int,
    seed: int,
    maxfevals: int,
    n_init: Optional[int],
    n_candidates: int,
    kernel: str,
    epsilon: float,
    smoothing: float,
    degree: int,
    distance_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect candidate minima from multiple RBF-surrogate runs."""
    x0_arr = None
    if x0_ls is not None:
        x0_arr = np.asarray(x0_ls, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)

    run_fn = partial(
        _run_rbf_surrogate_single,
        func=func,
        bounds=bounds,
        x0_ls=x0_arr,
        args=args,
        maxiter=maxiter,
        seed=seed,
        maxfevals=maxfevals,
        n_init=n_init,
        n_candidates=n_candidates,
        kernel=kernel,
        epsilon=epsilon,
        smoothing=smoothing,
        degree=degree,
        distance_tol=distance_tol,
    )
    return _collect_candidates_in_parallel(
        run_fn=run_fn,
        n_runs=n_runs,
        pool_executor_cls=ProcessPoolExecutor,
        broken_pool_exc=BrokenProcessPool,
    )


def _run_rbf_surrogate_single(
    run: int,
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    maxiter: int,
    seed: int,
    maxfevals: int,
    n_init: Optional[int],
    n_candidates: int,
    kernel: str,
    epsilon: float,
    smoothing: float,
    degree: int,
    distance_tol: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Execute one RBF-surrogate run and record improving incumbents."""
    rng = np.random.default_rng(seed + run)
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    n_dim = lb.size

    if np.allclose(ub, lb):
        x_fixed = np.array(lb, dtype=float)
        f_fixed = _evaluate_scalar_objective(func, x_fixed, args)
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
    n_candidates = max(128, int(n_candidates))
    merit_weights = (0.25, 0.45, 0.65, 0.85, 0.95)

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
        f = _evaluate_scalar_objective(func, x, args)
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
        model = _fit_rbf_surrogate_model(
            X=X_arr,
            y=y_arr,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
            degree=degree,
        )
        merit_weight = merit_weights[iter_idx % len(merit_weights)]
        u_next = _propose_rbf_surrogate_candidate(
            model=model,
            X_obs=X_arr,
            rng=rng,
            n_candidates=n_candidates,
            n_dim=n_dim,
            best_u=best_u,
            merit_weight=merit_weight,
            distance_tol=distance_tol,
        )
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


def _fit_rbf_surrogate_model(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    epsilon: float,
    smoothing: float,
    degree: int,
):
    """Fit an RBF interpolant surrogate, returning ``None`` when ill-conditioned."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] < 2:
        return None
    try:
        return RBFInterpolator(
            y=X,
            d=y,
            kernel=kernel,
            epsilon=float(epsilon),
            smoothing=float(smoothing),
            degree=int(degree),
        )
    except Exception:
        return None


def _propose_rbf_surrogate_candidate(
    model,
    X_obs: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int,
    n_dim: int,
    best_u: Optional[np.ndarray],
    merit_weight: float,
    distance_tol: float,
) -> np.ndarray:
    """Select next point from a merit function over random and local candidates."""
    U = rng.uniform(0.0, 1.0, size=(n_candidates, n_dim))

    if best_u is not None:
        n_local = max(32, n_candidates // 8)
        U_local = np.clip(
            best_u + rng.normal(0.0, 0.08, size=(n_local, n_dim)),
            0.0,
            1.0,
        )
        U = np.vstack([U, U_local, best_u.reshape(1, -1)])

    dists = np.linalg.norm(U[:, np.newaxis, :] - X_obs[np.newaxis, :, :], axis=2)
    d_min = np.min(dists, axis=1)

    if model is None:
        idx_far = int(np.argmax(d_min))
        return np.array(U[idx_far], dtype=float)

    y_hat = np.asarray(model(U), dtype=float).reshape(-1)
    y_min = float(np.min(y_hat))
    y_span = float(np.max(y_hat) - y_min)
    y_norm = (y_hat - y_min) / (y_span + 1e-12)

    d_max = float(np.max(d_min))
    d_norm = d_min / (d_max + 1e-12)
    score = merit_weight * y_norm + (1.0 - merit_weight) * (1.0 - d_norm)
    idx = int(np.argmin(score))
    u_next = np.array(U[idx], dtype=float)

    if d_min[idx] < distance_tol:
        idx_far = int(np.argmax(d_min))
        u_next = np.array(U[idx_far], dtype=float)

    return u_next
