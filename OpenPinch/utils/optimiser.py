"""Optimise a given function using a parallelised dual annealing approach."""

import os
import numpy as np
from typing import Callable, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import (
    dual_annealing,
    minimize,
)
_DA_ALLOWED_KEYS = frozenset(
    {
        "args", "constraints", "n_runs", "maxiter", "seed",
        "initial_temp", "restart_temp_ratio", "visit", 
        "accept", "maxfun", "cluster_tol", "max_minima",
    }
)

__all__ = ["dual_annealing_multiminima"]


#######################################################################################################
# Public API
#######################################################################################################

def dual_annealing_multiminima(
    func: Callable,
    x0: Optional[tuple] = None,
    func_kwargs: Optional[dict] = {},
    bounds: Optional[tuple] = None,
    opt_kwargs: Optional[dict] = {},
) -> list:
    """
    Multi-start dual annealing that returns verified local optima by:
      - collecting candidate minima via callback across multiple runs,
      - clustering them in normalized decision space,
      - polishing each cluster representative via constrained local optimization,
      - deduplicating polished minima.

    Parameters
    ----------
    func : callable
        Objective ``f(x, *args) -> scalar``, with ``x`` as a 1D array-like.
    x0 : tuple
        Tuple of initial variable values
    func_kwargs : tuple
        Extra arguments passed to ``func`` as ``func(x, *args)``.
    bounds : sequence of (lb, ub)
        Bounds [(lb1, ub1), ..., (lbn, ubn)].
    opt_kwargs : dict
        Optional keywords {
            constraints : dict or sequence of dict, optional
                SciPy-style constraints for `minimize` (e.g. for SLSQP or trust-constr).
                They are applied only in the polishing step, not in dual_annealing itself.
            n_runs : int
                Number of independent dual_annealing runs with different seeds.
            maxiter : int
                dual_annealing global search iterations per run.
            seed : int
                Base seed; run r uses seed + r.
            initial_temp, restart_temp_ratio, visit, accept, maxfun :
                Standard dual_annealing parameters.
            cluster_tol : float
                Distance tolerance in *normalized* decision space (0-1 per dim)
                for basin clustering.
            max_minima : int or None
                Maximum number of clustered minima to polish and return. If None,
                all clustered representatives are polished.
            local_method : str
                Local method for polishing (e.g. 'SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr').
        }

    Returns
    -------
    local_minima : list of ndarray
        [ndarray, ...] verified and deduplicated minima.
    """
    objective = _set_objective_func(func=func, func_kwargs=func_kwargs)
    da_kwargs = _filter_opt_kwargs(opt_kwargs, _DA_ALLOWED_KEYS)
    local_minima_x = _get_da_multiminima_in_parallel(
        func=objective,
        x0=x0,
        bounds=bounds,
        **da_kwargs,
    )        
    return local_minima_x


#######################################################################################################
# Helper functions
#######################################################################################################


def _get_da_multiminima_in_parallel(
    func,
    bounds,
    x0=None,
    args=(),
    constraints=(),
    n_runs=6,
    maxiter=300,
    seed=150,
    initial_temp=5230.0,
    restart_temp_ratio=2e-5,
    visit=2.62,
    accept=-5.0,
    maxfun=1_000_000,
    # clustering parameters
    cluster_tol=0.02,      # normalized distance for *first* clustering
    max_minima=4,       # maximum number of minima to polish/return
    local_method="SLSQP",  # default local method
):

    # --------- 1. Run Dual Annealing to gather potential minima ----------
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    all_x, all_f = _collect_da_candidates(
        func=func,
        bounds=bounds,
        x0=x0,
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
    order = np.argsort(all_f)
    all_x = all_x[order]
    all_f = all_f[order]
    # --------- 2. First-stage clustering (basin representatives) ----------
    basin_reps_idx = _cluster_candidates(
        xs=all_x,
        fs=all_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    if max_minima is not None:
        basin_reps_idx = basin_reps_idx[:max_minima]
    # --------- 3. Polishing: constrained local optimization from each rep ----------
    polished_x, polished_f = _polish_candidates(
        func=func,
        args=args,
        all_x=all_x,
        basin_reps_idx=basin_reps_idx,
        local_method=local_method,
        bounds=bounds,
        constraints=constraints,
    )
    # --------- 4. Deduplicate polished minima ----------
    local_minima_idx = _cluster_candidates(
        xs=polished_x,
        fs=polished_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    local_minima_fun = polished_f[local_minima_idx]
    local_minima_x = polished_x[local_minima_idx]
    return np.asarray(local_minima_fun), np.asarray(local_minima_x)


def _collect_da_candidates(
    func,
    bounds,
    x0,
    args,
    n_runs: int,
    maxiter: int,
    seed: int,
    initial_temp: float,
    restart_temp_ratio: float,
    visit: float,
    accept: float,
    maxfun: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_x = []
    all_f = []
    n_jobs_eff = max(1, int(n_runs))
    run_fn = partial(
        _run_da_single,
        func=func,
        bounds=bounds,
        x0=x0,
        args=args,
        maxiter=maxiter,
        seed=seed,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
    )

    if n_jobs_eff == 1:
        run_results = map(run_fn, range(n_runs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
            run_results = list(pool.map(run_fn, range(n_runs)))        
        # try:
        #     with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
        #         run_results = list(pool.map(run_fn, range(n_runs)))
        # except (pickle.PicklingError, AttributeError, TypeError):
        #     # Fall back to threads when objective context is not picklable.
        #     with ThreadPoolExecutor(max_workers=n_jobs_eff) as pool:
        #         run_results = list(pool.map(run_fn, range(n_runs)))

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


def _run_da_single(
    run: int,
    func,
    bounds,
    x0,
    args,
    maxiter: int,
    seed: int,
    initial_temp: float,
    restart_temp_ratio: float,
    visit: float,
    accept: float,
    maxfun: int,
) -> tuple[list[np.ndarray], list[float]]:
    run_minima_x = []
    run_minima_f = []

    def callback(x, f, context):
        # context: 0 = annealing, 1 = end of local search, 2 = end of run
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

    # Ensure final best of this run is included
    run_minima_x.append(np.array(res.x, dtype=float))
    run_minima_f.append(float(res.fun))
    return run_minima_x, run_minima_f


def _objective_from_result_dict(x, func, func_kwargs):
    return func(x, func_kwargs)["obj"]


def _set_objective_func(func, func_kwargs):
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
    """
    Greedy clustering in normalized decision space based on x-similarity.
    For each cluster, keep the index with the minimum f value.
    """
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
    if not basin_reps_idx:
        return np.asarray([]), np.asarray([])

    bounds_arg = bounds if local_method.upper() in ("SLSQP", "L-BFGS-B", "TNC") else None
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
    n_workers = min(n_tasks, max(1, os.cpu_count() or 1))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker_fn, basin_reps_idx))

    polished_x = np.asarray([x for x, _ in results], dtype=float)
    polished_f = np.asarray([f for _, f in results], dtype=float)
    return polished_x, polished_f


def _filter_opt_kwargs(opt_kwargs: dict, allowed_keys: frozenset) -> dict:
    return {k: v for k, v in opt_kwargs.items() if k in allowed_keys}
