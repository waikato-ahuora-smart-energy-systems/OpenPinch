"""Shared helpers for black-box optimisation backends."""

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize


def _evaluate_scalar_objective(
    func: Callable,
    x: np.ndarray,
    args: tuple,
) -> float:
    """Evaluate ``func`` and coerce the result to float."""
    if args:
        return float(func(x, *args))
    return float(func(x))


def _objective_from_result_dict(x, func, func_kwargs):
    """Adapt a mapping-returning objective to a scalar objective."""
    return func(x, func_kwargs)["obj"]


def _set_objective_func(func, func_kwargs):
    """Create a scalar objective wrapper for SciPy optimizers."""
    return partial(_objective_from_result_dict, func=func, func_kwargs=func_kwargs)


def _polish_single_candidate(
    idx: int,
    all_x: np.ndarray,
    func: Callable,
    args,
    local_method: str,
    bounds_arg: Optional[np.ndarray],
    constraints,
) -> tuple[np.ndarray, float]:
    """Run local optimization from one candidate point."""
    x0 = all_x[idx]
    res_loc = minimize(
        fun=func,
        x0=x0,
        args=args,
        method=local_method,
        bounds=bounds_arg,
        constraints=constraints,
    )
    return np.array(res_loc.x, dtype=float), float(res_loc.fun)


def _cluster_candidates(
    xs: np.ndarray,
    fs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol_norm: float,
) -> list:
    """Cluster candidate points in normalized space and keep best per cluster."""
    if np.all(ub != lb):
        xs_norm = (xs - lb) / (ub - lb)
    else:
        xs_norm = (xs - lb) / (ub - lb + 1e-3)

    centers_norm = []
    best_idx_per_cluster = []
    best_f_per_cluster = []

    for idx in range(len(xs_norm)):
        x_norm = xs_norm[idx]
        if not centers_norm:
            centers_norm.append(x_norm.copy())
            best_idx_per_cluster.append(idx)
            best_f_per_cluster.append(fs[idx])
            continue

        dists = np.linalg.norm(np.asarray(centers_norm) - x_norm, axis=1)
        cluster_idx = int(np.argmin(dists))

        if dists[cluster_idx] <= tol_norm:
            if fs[idx] < best_f_per_cluster[cluster_idx]:
                best_f_per_cluster[cluster_idx] = fs[idx]
                best_idx_per_cluster[cluster_idx] = idx
        else:
            centers_norm.append(x_norm.copy())
            best_idx_per_cluster.append(idx)
            best_f_per_cluster.append(fs[idx])

    return [i for _, i in sorted(zip(best_f_per_cluster, best_idx_per_cluster))]


def _polish_candidates(
    func: Callable,
    args: dict,
    all_x: np.ndarray,
    basin_reps_idx: list,
    local_method: str,
    bounds: np.ndarray,
    constraints,
) -> tuple[np.ndarray, np.ndarray]:
    """Polish clustered basin representatives in parallel."""
    if not basin_reps_idx:
        return np.asarray([]), np.asarray([])

    bounds_arg = (
        bounds if local_method.upper() in ("SLSQP", "L-BFGS-B", "TNC") else None
    )
    worker_fn = partial(
        _polish_single_candidate,
        all_x=all_x,
        func=func,
        args=args,
        local_method=local_method,
        bounds_arg=bounds_arg,
        constraints=constraints,
    )

    n_tasks = len(basin_reps_idx)
    n_workers = min(n_tasks, max(1, os.cpu_count()))

    if n_workers == 1:
        results = list(map(worker_fn, basin_reps_idx))
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(worker_fn, basin_reps_idx))
        except PermissionError, OSError, BrokenProcessPool:
            results = list(map(worker_fn, basin_reps_idx))

    polished_x = np.asarray([x for x, _ in results], dtype=float)
    polished_f = np.asarray([f for _, f in results], dtype=float)
    return polished_x, polished_f


def _collect_candidates_in_parallel(
    run_fn: Callable[[int], tuple[list[np.ndarray], list[float]]],
    n_runs: int,
    pool_executor_cls=ProcessPoolExecutor,
    broken_pool_exc=BrokenProcessPool,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute per-run candidate collectors in parallel with serial fallback."""
    all_x = []
    all_f = []
    n_jobs_eff = max(1, int(n_runs))

    if n_jobs_eff == 1:
        run_results = map(run_fn, range(n_runs))
    else:
        try:
            with pool_executor_cls(max_workers=n_jobs_eff) as pool:
                run_results = list(pool.map(run_fn, range(n_runs)))
        except PermissionError, OSError, broken_pool_exc:
            run_results = map(run_fn, range(n_runs))

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


def _postprocess_candidates(
    func: Callable,
    args: tuple,
    bounds: np.ndarray,
    constraints,
    all_x: np.ndarray,
    all_f: np.ndarray,
    cluster_tol: float,
    max_minima: Optional[int],
    local_method: str,
    cluster_fn: Callable = _cluster_candidates,
    polish_fn: Callable = _polish_candidates,
) -> np.ndarray:
    """Apply shared sort/cluster/polish/deduplicate pipeline to candidates."""
    if all_x.size == 0:
        return np.asarray([])

    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]

    order = np.argsort(all_f)
    all_x = all_x[order]
    all_f = all_f[order]

    basin_reps_idx = cluster_fn(
        xs=all_x,
        fs=all_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    if max_minima is not None:
        basin_reps_idx = basin_reps_idx[:max_minima]

    polished_x, polished_f = polish_fn(
        func=func,
        args=args,
        all_x=all_x,
        basin_reps_idx=basin_reps_idx,
        local_method=local_method,
        bounds=bounds,
        constraints=constraints,
    )
    if polished_x.size == 0:
        return polished_x

    local_minima_idx = cluster_fn(
        xs=polished_x,
        fs=polished_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    return np.asarray(polished_x[local_minima_idx])


def _filter_opt_kwargs(opt_kwargs: dict, allowed_keys: frozenset) -> dict:
    """Return only optimizer keyword arguments supported by this module."""
    return {k: v for k, v in opt_kwargs.items() if k in allowed_keys}
