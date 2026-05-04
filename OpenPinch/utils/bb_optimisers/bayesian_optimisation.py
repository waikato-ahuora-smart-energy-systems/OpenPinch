"""Bayesian-optimisation multi-start backend."""

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Callable, Optional

import numpy as np
from scipy.special import erf

from .common import (
    _cluster_candidates,
    _collect_candidates_in_parallel,
    _evaluate_scalar_objective,
    _polish_candidates,
    _postprocess_candidates,
)


def _get_bo_multiminima_in_parallel(
    func,
    bounds,
    x0_ls=None,
    args=(),
    constraints=(),
    n_runs=os.cpu_count(),
    maxiter=300,
    seed=0,
    maxfun=1_000_000,
    maxfevals=None,
    n_init=None,
    acq_candidates=2048,
    lengthscale=None,
    noise=1e-8,
    xi=1e-3,
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
):
    """Return deduplicated local minima from multi-start Bayesian optimisation."""
    bounds = np.asarray(bounds, dtype=float)
    effective_maxfevals = maxfevals if maxfevals is not None else maxfun

    all_x, all_f = _collect_bo_candidates(
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        n_runs=n_runs,
        maxiter=maxiter,
        seed=seed,
        maxfevals=effective_maxfevals,
        n_init=n_init,
        acq_candidates=acq_candidates,
        lengthscale=lengthscale,
        noise=noise,
        xi=xi,
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


def _collect_bo_candidates(
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    n_runs: int,
    maxiter: int,
    seed: int,
    maxfevals: int,
    n_init: Optional[int],
    acq_candidates: int,
    lengthscale: Optional[float],
    noise: float,
    xi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect candidate minima from multiple Bayesian-optimisation runs."""
    x0_arr = None
    if x0_ls is not None:
        x0_arr = np.asarray(x0_ls, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)

    run_fn = partial(
        _run_bo_single,
        func=func,
        bounds=bounds,
        x0_ls=x0_arr,
        args=args,
        maxiter=maxiter,
        seed=seed,
        maxfevals=maxfevals,
        n_init=n_init,
        acq_candidates=acq_candidates,
        lengthscale=lengthscale,
        noise=noise,
        xi=xi,
    )
    return _collect_candidates_in_parallel(
        run_fn=run_fn,
        n_runs=n_runs,
        pool_executor_cls=ProcessPoolExecutor,
        broken_pool_exc=BrokenProcessPool,
    )


def _run_bo_single(
    run: int,
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    maxiter: int,
    seed: int,
    maxfevals: int,
    n_init: Optional[int],
    acq_candidates: int,
    lengthscale: Optional[float],
    noise: float,
    xi: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Execute one BO run and record improving incumbent candidates."""
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

    for _ in range(int(maxiter)):
        if eval_count >= maxfevals or len(X_unit) == 0:
            break

        X_arr = np.asarray(X_unit, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        try:
            model = _fit_bo_gp_model(
                X=X_arr,
                y=y_arr,
                lengthscale=lengthscale,
                noise=noise,
            )
            u_next = _propose_bo_candidate(
                model=model,
                best_f=float(np.min(y_arr)),
                rng=rng,
                acq_candidates=acq_candidates,
                n_dim=n_dim,
                xi=xi,
                best_u=best_u,
            )
        except np.linalg.LinAlgError, ValueError, FloatingPointError:
            u_next = rng.uniform(0.0, 1.0, size=n_dim)

        if X_arr.size > 0:
            d_min = np.min(np.linalg.norm(X_arr - u_next, axis=1))
            if d_min < 1e-6:
                u_next = np.clip(
                    u_next + rng.normal(0.0, 0.03, size=n_dim),
                    0.0,
                    1.0,
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


def _fit_bo_gp_model(
    X: np.ndarray,
    y: np.ndarray,
    lengthscale: Optional[float],
    noise: float,
) -> dict:
    """Fit a simple isotropic RBF Gaussian-process model."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not np.all(finite):
        X = X[finite]
        y = y[finite]
    if X.shape[0] == 0:
        raise ValueError("No finite BO observations available for GP fitting.")

    n_samples, n_dim = X.shape

    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    signal = max(float(np.std(y_centered)), 1e-6)

    if lengthscale is not None:
        ell = max(float(lengthscale), 1e-6)
    elif n_samples > 1:
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        d_nonzero = dists[dists > 0.0]
        if d_nonzero.size > 0:
            ell = float(np.clip(np.median(d_nonzero), 0.05, 1.5))
        else:
            ell = 0.35
    else:
        ell = 0.35 * np.sqrt(max(n_dim, 1))

    noise_var = max(float(noise), 1e-10)
    K_base = _rbf_kernel(X, X, ell, signal)
    jitter = 1e-8
    L = None
    for _ in range(12):
        try:
            K = K_base + (noise_var + jitter) * np.eye(n_samples)
            L = np.linalg.cholesky(K)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if L is None:
        return {
            "X": X,
            "alpha": np.zeros(n_samples, dtype=float),
            "L": None,
            "y_mean": y_mean,
            "lengthscale": ell,
            "signal": signal,
        }

    try:
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    except np.linalg.LinAlgError:
        alpha = np.zeros(n_samples, dtype=float)
    return {
        "X": X,
        "alpha": alpha,
        "L": L,
        "y_mean": y_mean,
        "lengthscale": ell,
        "signal": signal,
    }


def _propose_bo_candidate(
    model: dict,
    best_f: float,
    rng: np.random.Generator,
    acq_candidates: int,
    n_dim: int,
    xi: float,
    best_u: Optional[np.ndarray],
) -> np.ndarray:
    """Select next BO query point by maximizing expected improvement."""
    n_rand = max(128, int(acq_candidates))
    U = rng.uniform(0.0, 1.0, size=(n_rand, n_dim))

    if best_u is not None:
        n_local = max(32, n_rand // 8)
        U_local = np.clip(
            best_u + rng.normal(0.0, 0.08, size=(n_local, n_dim)),
            0.0,
            1.0,
        )
        U = np.vstack([U, U_local, best_u.reshape(1, -1)])

    mu, var = _predict_bo_gp(model, U)
    ei = _expected_improvement(
        mu=mu,
        var=var,
        best_f=best_f,
        xi=xi,
    )
    idx = int(np.argmax(ei))
    return np.array(U[idx], dtype=float)


def _predict_bo_gp(model: dict, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict posterior mean and variance for a fitted BO GP model."""
    X_train = model["X"]
    alpha = model["alpha"]
    L = model["L"]
    y_mean = model["y_mean"]
    ell = model["lengthscale"]
    signal = model["signal"]
    n_query = X_query.shape[0]
    if L is None:
        mu = np.full(n_query, y_mean, dtype=float)
        var = np.full(n_query, max(signal**2, 1e-12), dtype=float)
        return mu, var

    K_star = _rbf_kernel(X_train, X_query, ell, signal)
    mu = y_mean + K_star.T @ alpha

    try:
        v = np.linalg.solve(L, K_star)
    except np.linalg.LinAlgError:
        mu = np.full(n_query, y_mean, dtype=float)
        var = np.full(n_query, max(signal**2, 1e-12), dtype=float)
        return mu, var
    var = signal**2 - np.sum(v * v, axis=0)
    var = np.maximum(var, 1e-16)
    return mu, var


def _expected_improvement(
    mu: np.ndarray,
    var: np.ndarray,
    best_f: float,
    xi: float,
) -> np.ndarray:
    """Expected-improvement acquisition for minimization."""
    sigma = np.sqrt(np.maximum(var, 1e-16))
    improvement = best_f - mu - float(xi)
    z = np.zeros_like(improvement)
    valid = sigma > 1e-14
    z[valid] = improvement[valid] / sigma[valid]
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    ei = improvement * cdf + sigma * pdf
    ei[~valid] = 0.0
    return np.maximum(ei, 0.0)


def _rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    lengthscale: float,
    signal: float,
) -> np.ndarray:
    """Isotropic RBF kernel."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    return (signal**2) * np.exp(-0.5 * sq_dist / (lengthscale**2))
