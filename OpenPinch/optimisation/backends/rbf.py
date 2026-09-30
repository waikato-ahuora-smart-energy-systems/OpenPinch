"""RBF-surrogate multi-start backend."""

import os
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import RBFInterpolator

from ..candidates import _evaluate_scalar_objective
from ._multistart import run_multistart
from ._surrogate import candidate_pool, run_surrogate_loop


def _get_rbf_surrogate_multiminima_in_parallel(
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
    effective_maxfevals = maxfevals if maxfevals is not None else maxfun
    return run_multistart(
        _run_rbf_surrogate_single,
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
        maxfevals=effective_maxfevals,
        n_init=n_init,
        n_candidates=n_candidates,
        kernel=kernel,
        epsilon=epsilon,
        smoothing=smoothing,
        degree=degree,
        distance_tol=distance_tol,
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
    merit_weights = (0.25, 0.45, 0.65, 0.85, 0.95)

    def propose(iter_idx, X_arr, y_arr, rng, best_u, n_dim):
        model = _fit_rbf_surrogate_model(
            X=X_arr,
            y=y_arr,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
            degree=degree,
        )
        merit_weight = merit_weights[iter_idx % len(merit_weights)]
        return _propose_rbf_surrogate_candidate(
            model=model,
            X_obs=X_arr,
            rng=rng,
            n_candidates=max(128, int(n_candidates)),
            n_dim=n_dim,
            best_u=best_u,
            merit_weight=merit_weight,
            distance_tol=distance_tol,
        )

    return run_surrogate_loop(
        run,
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        maxiter=maxiter,
        seed=seed,
        maxfevals=maxfevals,
        n_init=n_init,
        propose=propose,
        evaluate=_evaluate_scalar_objective,
    )


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
    U = candidate_pool(rng, n_candidates, n_dim, best_u)

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
