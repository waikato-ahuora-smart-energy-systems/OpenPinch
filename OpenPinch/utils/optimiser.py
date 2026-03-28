"""Target heat pump integration for given heating or cooler profiles."""

import numpy as np
from typing import Callable, Optional
from scipy.optimize import (
    differential_evolution,
    dual_annealing,
    minimize,
)
_DE_ALLOWED_KEYS = frozenset(
    {
        "strategy", "maxiter", "popsize", "tol", "mutation",
        "recombination", "seed", "callback", "disp", "polish",
        "init", "atol", "updating", "workers", "constraints",
        "x0", "integrality", "vectorized",
    }
)
_DA_ALLOWED_KEYS = frozenset(
    {
        "args", "constraints", "n_runs", "maxiter", "seed",
        "initial_temp", "restart_temp_ratio", "visit", 
        "accept", "maxfun", "cluster_tol", "x_tol_norm",
        "f_tol_rel", "max_minima",
    }
)

__all__ = ["custom_optimiser", "dual_annealing_multiminima"]


#######################################################################################################
# Public API
#######################################################################################################

def dual_annealing_multiminima(
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
    # clustering + verification parameters
    cluster_tol=0.05,      # normalized distance for *first* clustering
    x_tol_norm=0.01,       # normalized distance to consider two minima identical
    f_tol_rel=0.01,        # relative f difference tolerance
    max_minima=5,       # maximum number of minima to polish/return
    local_method="SLSQP",  # default local method
):
    """
    Multi-start dual_annealing that returns *verified* local optima by:
      - collecting candidate minima via callback across multiple runs,
      - clustering them in normalized decision space,
      - polishing each cluster representative via constrained local optimization,
      - deduplicating polished minima in (x, f) space.

    Parameters
    ----------
    func : callable
        Objective f(x, *args) -> scalar, with x a 1D array-like.
    bounds : sequence of (lb, ub)
        Bounds [(lb1, ub1), ..., (lbn, ubn)].
    args : tuple
        Extra arguments passed to func as func(x, *args).
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
        for first-stage basin clustering.
    x_tol_norm : float
        Normalized distance threshold to treat two polished minima as identical.
    f_tol_rel : float
        Relative objective difference threshold to treat minima as identical.
    max_minima : int or None
        Maximum number of clustered minima to polish and return. If None,
        all clustered representatives are polished.
    local_method : str
        Local method for polishing (e.g. 'SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr').

    Returns
    -------
    local_minima : list of ndarray
        [ndarray, ...] verified and deduplicated minima.
    """
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
    local_minima_fun, local_minima_x = _dedup_minima(
        polished_x,
        polished_f,
        lb,
        ub,
        x_tol_norm=x_tol_norm,
        f_tol_rel=f_tol_rel,
    )

    return np.asarray(local_minima_fun), np.asarray(local_minima_x)


def custom_optimiser(
    func: Callable,
    x0: Optional[tuple] = None,
    func_kwargs: Optional[dict] = {},
    bounds: Optional[tuple] = None,
    opt_kwargs: Optional[dict] = {},
) -> list:
    """
    """
    da_kwargs = _filter_opt_kwargs(opt_kwargs, _DA_ALLOWED_KEYS)
    local_minima_fun, local_minima_x = dual_annealing_multiminima(
        func=lambda x: func(x, func_kwargs)["obj"],
        x0=x0,
        bounds=bounds,
        **da_kwargs,
    )
    valid_minima = local_minima_x[local_minima_fun > 0]
    if valid_minima.size == 0:
        de_kwargs = _filter_opt_kwargs(opt_kwargs, _DE_ALLOWED_KEYS)
        res = differential_evolution(
            func=lambda x: func(x, func_kwargs)["obj"],
            bounds=bounds,
            **de_kwargs,
        )
        valid_minima = np.asarray([res.x])
        
    return valid_minima


#######################################################################################################
# Helper functions
#######################################################################################################


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

    for run in range(n_runs):
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

        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


def _cluster_candidates(
    xs: np.ndarray,
    fs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol_norm: float,
) -> list:
    """
    Greedy clustering in normalized decision space.
    Accepts the best point first, then any point farther than tol_norm
    from all existing centers (in normalized coordinates).
    """
    idx_sorted = np.argsort(fs)
    xs_norm = (xs - lb) / (ub - lb)

    centers_norm = []
    centers_idx = []

    for idx in idx_sorted:
        x_norm = xs_norm[idx]
        if not centers_norm:
            centers_norm.append(x_norm.copy())
            centers_idx.append(idx)
            continue

        dists = np.linalg.norm(np.asarray(centers_norm) - x_norm, axis=1)
        if np.all(dists > tol_norm):
            centers_norm.append(x_norm.copy())
            centers_idx.append(idx)

    return centers_idx


def _polish_candidates(
    func,
    args,
    all_x: np.ndarray,
    basin_reps_idx: list,
    local_method: str,
    bounds: np.ndarray,
    constraints,
) -> tuple[np.ndarray, np.ndarray]:
    polished_x = []
    polished_f = []

    for idx in basin_reps_idx:
        x0 = all_x[idx]
        res_loc = minimize(
            lambda x: func(x, *args),
            x0,
            method=local_method,
            bounds=bounds if local_method.upper() in ("SLSQP", "L-BFGS-B", "TNC") else None,
            constraints=constraints,
        )
        polished_x.append(np.array(res_loc.x, dtype=float))
        polished_f.append(float(res_loc.fun))

    return np.asarray(polished_x), np.asarray(polished_f)


def _dedup_minima(
    xs: np.ndarray,
    fs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x_tol_norm: float,
    f_tol_rel: float,
) -> tuple[list, list]:
    """
    Treat minima as identical if:
        - normalized distance <= x_tol_norm, AND
        - |f1 - f2| <= f_tol_rel * max(1, |f1|, |f2|)
    We keep the best (lowest f) representative.
    """
    idx_sorted = np.argsort(fs)
    xs_norm = (xs - lb) / (ub - lb)

    reps = []
    reps_vals = []

    for idx in idx_sorted:
        x_norm = xs_norm[idx]
        f = fs[idx]

        if not reps:
            reps.append(xs[idx].copy())
            reps_vals.append(f)
            continue

        is_new = True
        for r, fr in zip(reps, reps_vals):
            r_norm = (r - lb) / (ub - lb)
            dx = np.linalg.norm(x_norm - r_norm)
            df = abs(f - fr)
            scale = max(1.0, abs(f), abs(fr))
            if dx <= x_tol_norm and df <= f_tol_rel * scale:
                is_new = False
                break

        if is_new:
            reps.append(xs[idx].copy())
            reps_vals.append(f)

    return [float(fr) for fr in reps_vals], [r.tolist() for r in reps]


def _filter_opt_kwargs(opt_kwargs: dict, allowed_keys: frozenset) -> dict:
    return {k: v for k, v in opt_kwargs.items() if k in allowed_keys}
