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
    x0_ls: Optional[tuple] = None,
    func_kwargs: Optional[dict] = {},
    bounds: Optional[tuple] = None,
    opt_kwargs: Optional[dict] = {},
) -> list:
    """Return deduplicated local minima from multi-start dual annealing.

    The routine performs four stages:

    1. collect candidate minima from multiple dual-annealing runs,
    2. cluster candidates in normalized decision space,
    3. polish one representative per cluster with local optimization,
    4. re-cluster polished solutions to remove duplicates.

    Parameters
    ----------
    func : callable
        Objective called as ``func(x, func_kwargs)`` and expected to return a
        mapping containing key ``"obj"``.
    x0_ls : tuple, optional
        Optional collection of initial decision vectors. When provided, run
        ``r`` starts from ``x0_ls[r % len(x0_ls)]``.
    func_kwargs : dict, optional
        Extra data passed to ``func`` via the objective wrapper.
    bounds : tuple, optional
        Bounds ``((lb1, ub1), ..., (lbn, ubn))`` for the decision vector.
    opt_kwargs : dict
        Optional controls. Supported keys are:
        ``constraints``, ``n_runs``, ``maxiter``, ``seed``, ``initial_temp``,
        ``restart_temp_ratio``, ``visit``, ``accept``, ``maxfun``,
        ``cluster_tol``, and ``max_minima``.

    Returns
    -------
    np.ndarray
        Decision vectors for polished, deduplicated local minima, ordered by
        objective value.
    """
    objective = _set_objective_func(func=func, func_kwargs=func_kwargs)
    da_kwargs = _filter_opt_kwargs(opt_kwargs, _DA_ALLOWED_KEYS)
    local_minima_x = _get_da_multiminima_in_parallel(
        func=objective,
        x0_ls=x0_ls,
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
    x0_ls=None,
    args=(),
    constraints=(),
    n_runs=6,
    maxiter=300,
    seed=0,
    initial_temp=5230.0,
    restart_temp_ratio=2e-5,
    visit=2.62,
    accept=-5.0,
    maxfun=1_000_000,
    # clustering parameters
    cluster_tol=0.01,      # normalized distance for *first* clustering
    max_minima=4,       # maximum number of minima to polish/return
    local_method="SLSQP",  # default local method
):
    """Run candidate collection, clustering, polishing, and deduplication.

    Parameters
    ----------
    func : callable
        Scalar objective function compatible with SciPy optimizers.
    bounds : array-like
        Per-variable lower/upper bounds.
    x0_ls : array-like, optional
        Optional initial vectors cycled across runs.
    args : tuple, default=()
        Positional arguments passed to ``func`` by SciPy.
    constraints : sequence, default=()
        SciPy-compatible constraints for the local polishing step.
    n_runs : int, default=6
        Number of independent dual-annealing runs.
    maxiter : int, default=300
        Maximum dual-annealing iterations per run.
    seed : int, default=0
        Base random seed. Run ``r`` uses ``seed + r``.
    initial_temp, restart_temp_ratio, visit, accept, maxfun
        Standard dual-annealing controls.
    cluster_tol : float, default=0.01
        Euclidean clustering tolerance in normalized decision space.
    max_minima : int, default=4
        Maximum number of basin representatives selected for polishing.
    local_method : str, default="SLSQP"
        Local optimizer used for candidate polishing.

    Returns
    -------
    np.ndarray
        Polished, deduplicated local-minimum decision vectors.
    """

    # --------- 1. Run Dual Annealing to gather potential minima ----------
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
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
    return np.asarray(
        polished_x[local_minima_idx]
    )


def _collect_da_candidates(
    func: Callable,
    bounds: tuple,
    x0_ls:np.ndarray,
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
    """Collect candidate minima from multiple dual-annealing runs.

    Parameters
    ----------
    func : Callable
        Scalar objective function.
    bounds : tuple
        Per-variable lower/upper bounds.
    x0_ls : np.ndarray
        Optional initial vectors used to seed each run.
    args : dict
        Extra arguments forwarded to SciPy objective calls.
    n_runs : int
        Number of independent runs.
    maxiter : int
        Maximum dual-annealing iterations per run.
    seed : int
        Base random seed.
    initial_temp, restart_temp_ratio, visit, accept, maxfun
        Standard dual-annealing controls.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Candidate decision vectors and their objective values.
    """
    all_x = []
    all_f = []
    n_jobs_eff = max(1, int(n_runs))
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

    if n_jobs_eff == 1:
        run_results = map(run_fn, range(n_runs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
            run_results = list(pool.map(run_fn, range(n_runs)))        

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


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

    Parameters
    ----------
    run : int
        Zero-based run index.
    func : Callable
        Scalar objective function.
    bounds : tuple
        Per-variable lower/upper bounds.
    x0_ls : np.ndarray
        Optional initial vectors cycled by run index.
    args : dict
        Extra arguments forwarded to SciPy objective calls.
    maxiter : int
        Maximum dual-annealing iterations.
    seed : int
        Base seed; this run uses ``seed + run``.
    initial_temp, restart_temp_ratio, visit, accept, maxfun
        Standard dual-annealing controls.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        Candidate vectors and objective values gathered during the run.
    """
    run_minima_x = []
    run_minima_f = []
    x0 = x0_ls[run % np.shape(x0_ls)[0]] if x0_ls is not None else None
    
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
    """Adapt a mapping-returning objective to a scalar objective.

    Parameters
    ----------
    x : np.ndarray
        Decision vector.
    func : Callable
        Objective returning a dict-like result containing ``"obj"``.
    func_kwargs : dict
        Extra objective inputs.

    Returns
    -------
    float
        Scalar objective value ``func(...)[\"obj\"]``.
    """
    return func(x, func_kwargs)["obj"]


def _set_objective_func(func, func_kwargs):
    """Create a scalar objective wrapper for SciPy optimizers.

    Parameters
    ----------
    func : Callable
        Objective returning a mapping with key ``"obj"``.
    func_kwargs : dict
        Extra data passed to ``func``.

    Returns
    -------
    Callable
        Callable of signature ``f(x) -> float``.
    """
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
    """Run local optimization from one candidate point.

    Parameters
    ----------
    idx : int
        Index of the candidate in ``all_x``.
    all_x : np.ndarray
        Candidate decision vectors.
    func : Callable
        Scalar objective function.
    args : Any
        Positional arguments forwarded to ``minimize``.
    local_method : str
        Local optimizer name (for example ``"SLSQP"``).
    bounds_arg : np.ndarray or None
        Bounds used by compatible local methods.
    constraints : Any
        Constraints passed to ``minimize``.

    Returns
    -------
    tuple[np.ndarray, float]
        Polished decision vector and objective value.
    """
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
    """Cluster candidate points in normalized space and keep best per cluster.

    Parameters
    ----------
    xs : np.ndarray
        Candidate decision vectors.
    fs : np.ndarray
        Objective values associated with ``xs``.
    lb, ub : np.ndarray
        Lower and upper variable bounds used for normalization.
    tol_norm : float
        Maximum normalized Euclidean distance for cluster membership.

    Returns
    -------
    list
        Indices of selected representatives, sorted by objective value.
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
    """Polish clustered basin representatives in parallel.

    Parameters
    ----------
    func : Callable
        Scalar objective function.
    args : dict
        Positional arguments forwarded to the objective.
    all_x : np.ndarray
        Candidate vectors.
    basin_reps_idx : list
        Indices of representative candidates to polish.
    local_method : str
        Local optimizer name.
    bounds : np.ndarray
        Bounds for local methods that accept them.
    constraints : Any
        Constraints passed to local optimization.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Polished vectors and objective values.
    """
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
    """Return only optimizer keyword arguments supported by this module.

    Parameters
    ----------
    opt_kwargs : dict
        User-supplied optimizer keyword arguments.
    allowed_keys : frozenset
        Supported option names.

    Returns
    -------
    dict
        Filtered keyword dictionary with unsupported keys removed.
    """
    return {k: v for k, v in opt_kwargs.items() if k in allowed_keys}
