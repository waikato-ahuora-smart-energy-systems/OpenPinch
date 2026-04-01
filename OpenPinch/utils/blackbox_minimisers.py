"""Black-box optimisation helpers for dual annealing, CMA-ES, BO, and RBF multistarts."""

import os
import numpy as np
from typing import Callable, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from scipy.interpolate import RBFInterpolator
from scipy.optimize import dual_annealing, minimize
from scipy.special import erf
from ..lib.enums import BB_Minimiser
_DA_ALLOWED_KEYS = frozenset(
    {
        "args", "constraints", "n_runs", "maxiter", "seed",
        "initial_temp", "restart_temp_ratio", "visit", 
        "accept", "maxfun", "cluster_tol", "max_minima",
    }
)
_CMA_ALLOWED_KEYS = frozenset(
    {
        "args", "constraints", "n_runs", "maxiter", "seed",
        "maxfun", "maxfevals", "popsize", "sigma0",
        "cluster_tol", "max_minima", "local_method",
        "tolx", "tolfun",
    }
)
_BO_ALLOWED_KEYS = frozenset(
    {
        "args", "constraints", "n_runs", "maxiter", "seed",
        "maxfun", "maxfevals", "cluster_tol", "max_minima",
        "local_method", "n_init", "acq_candidates",
        "lengthscale", "noise", "xi",
    }
)
_RBF_ALLOWED_KEYS = frozenset(
    {
        "args", "constraints", "n_runs", "maxiter", "seed",
        "maxfun", "maxfevals", "cluster_tol", "max_minima",
        "local_method", "n_init", "n_candidates", "kernel",
        "epsilon", "smoothing", "degree", "distance_tol",
    }
)

__all__ = [
    "multiminima",
]


#######################################################################################################
# Public API
#######################################################################################################


_DEFAULT_MULTIMINIMA_HANDLE = BB_Minimiser.DA
_MULTIMINIMA_ALLOWED_KEYS = {
    BB_Minimiser.DA: _DA_ALLOWED_KEYS,
    BB_Minimiser.CMAES: _CMA_ALLOWED_KEYS,
    BB_Minimiser.BO: _BO_ALLOWED_KEYS,
    BB_Minimiser.RBF: _RBF_ALLOWED_KEYS,
}
_MULTIMINIMA_HANDLE_ALIASES = {method.value: method for method in BB_Minimiser}
_MULTIMINIMA_HANDLE_ALIASES.update(
    {
        "da": BB_Minimiser.DA,
        "dualannealing": BB_Minimiser.DA,
        "dual-annealing": BB_Minimiser.DA,
        "dual_annealing_multiminima": BB_Minimiser.DA,
        "cma": BB_Minimiser.CMAES,
        "cma-es": BB_Minimiser.CMAES,
        "cma_es": BB_Minimiser.CMAES,
        "cmaes_multiminima": BB_Minimiser.CMAES,
        "bayes": BB_Minimiser.BO,
        "bayesian_optimization": BB_Minimiser.BO,
        "bayesian_optimisation": BB_Minimiser.BO,
        "bo_multiminima": BB_Minimiser.BO,
        "rbf": BB_Minimiser.RBF,
        "rbf-surrogate": BB_Minimiser.RBF,
        "rbf_surrogate_multiminima": BB_Minimiser.RBF,
    }
)


def _get_supported_multiminima_handles() -> tuple[BB_Minimiser, ...]:
    """Return canonical optimizer handles supported by :func:`multiminima`."""
    return tuple(_MULTIMINIMA_ALLOWED_KEYS.keys())


def _get_multiminima_handler(
    handle: Optional[str | BB_Minimiser] = _DEFAULT_MULTIMINIMA_HANDLE,
) -> Callable:
    """Return the backend runner corresponding to ``handle``."""
    canonical_handle = _normalise_multiminima_handle(handle)
    if canonical_handle is BB_Minimiser.DA:
        return _get_da_multiminima_in_parallel
    if canonical_handle is BB_Minimiser.CMAES:
        return _get_cma_multiminima_in_parallel
    if canonical_handle is BB_Minimiser.BO:
        return _get_bo_multiminima_in_parallel
    if canonical_handle is BB_Minimiser.RBF:
        return _get_rbf_surrogate_multiminima_in_parallel

    supported = ", ".join(method.value for method in _get_supported_multiminima_handles())
    raise ValueError(
        f"Unsupported optimiser handle {canonical_handle!r}. Supported handles: {supported}."
    )


def multiminima(
    func: Callable,
    x0_ls: Optional[tuple] = None,
    func_kwargs: Optional[dict] = {},
    bounds: Optional[tuple] = None,
    opt_kwargs: Optional[dict] = {},
    optimiser_handle: Optional[str | BB_Minimiser] = _DEFAULT_MULTIMINIMA_HANDLE,
) -> list:
    """Run a selected multi-start optimizer by handle.

    Parameters
    ----------
    func, x0_ls, func_kwargs, bounds, opt_kwargs
        Passed directly to the selected optimizer function.
    optimiser_handle : str or BB_Minimiser, optional
        Canonical or alias handle for the optimization method.

    Returns
    -------
    np.ndarray
        Decision vectors for polished, deduplicated local minima.
    """
    func_kwargs = {} if func_kwargs is None else func_kwargs
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs
    objective = _set_objective_func(func=func, func_kwargs=func_kwargs)
    canonical_handle = _normalise_multiminima_handle(optimiser_handle)
    handler = _get_multiminima_handler(handle=canonical_handle)
    allowed_keys = _MULTIMINIMA_ALLOWED_KEYS[canonical_handle]
    backend_kwargs = _filter_opt_kwargs(opt_kwargs, allowed_keys)
    return handler(
        func=objective,
        x0_ls=x0_ls,
        bounds=bounds,
        **backend_kwargs,
    )


#######################################################################################################
# Helper functions
#######################################################################################################


def _normalise_multiminima_handle(
    handle: Optional[str | BB_Minimiser],
) -> BB_Minimiser:
    """Resolve a canonical optimizer handle from a canonical name or alias."""
    if handle is None:
        return _DEFAULT_MULTIMINIMA_HANDLE
    if isinstance(handle, BB_Minimiser):
        return handle
    if not isinstance(handle, str):
        raise TypeError(
            "Optimizer handle must be a string or BB_Minimiser; "
            f"got {type(handle).__name__}."
        )

    normalised = handle.strip().lower().replace(" ", "_")
    canonical = _MULTIMINIMA_HANDLE_ALIASES.get(normalised)
    if canonical is None:
        supported = ", ".join(method.value for method in _get_supported_multiminima_handles())
        raise ValueError(
            f"Unsupported optimiser handle {handle!r}. Supported handles: {supported}."
        )
    return canonical


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
    """Return deduplicated local minima from multi-start dual annealing.

    The routine performs four stages:

    1. collect candidate minima from multiple dual-annealing runs,
    2. cluster candidates in normalized decision space,
    3. polish one representative per cluster with local optimization,
    4. re-cluster polished solutions to remove duplicates.

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


def _get_cma_multiminima_in_parallel(
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
    popsize=None,
    sigma0=None,
    # clustering parameters
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
    tolx=1e-8,
    tolfun=1e-10,
):
    """Return deduplicated local minima from multi-start CMA-ES.

    Parameters
    ----------
    func : callable
        Scalar objective function.
    bounds : array-like
        Per-variable lower/upper bounds.
    x0_ls : array-like, optional
        Optional initial vectors cycled across runs.
    args : tuple, default=()
        Positional arguments passed to ``func``.
    constraints : sequence, default=()
        SciPy-compatible constraints for the local polishing step.
    n_runs : int, default=6
        Number of independent CMA-ES runs.
    maxiter : int, default=300
        Maximum generations per run.
    seed : int, default=0
        Base random seed. Run ``r`` uses ``seed + r``.
    maxfun, maxfevals : int, optional
        Maximum objective evaluations per run. ``maxfevals`` takes precedence.
    popsize : int, optional
        CMA-ES population size. When ``None`` a dimension-based default is used.
    sigma0 : float, optional
        Initial global step-size.
    cluster_tol : float, default=0.01
        Euclidean clustering tolerance in normalized decision space.
    max_minima : int, default=4
        Maximum number of basin representatives selected for polishing.
    local_method : str, default="SLSQP"
        Local optimizer used for candidate polishing.
    tolx : float, default=1e-8
        Absolute step-size termination threshold.
    tolfun : float, default=1e-10
        Improvement threshold used for stagnation detection.

    Returns
    -------
    np.ndarray
        Polished, deduplicated local-minimum decision vectors.
    """
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    effective_maxfevals = maxfevals if maxfevals is not None else maxfun

    all_x, all_f = _collect_cma_candidates(
        func=func,
        bounds=bounds,
        x0_ls=x0_ls,
        args=args,
        n_runs=n_runs,
        maxiter=maxiter,
        seed=seed,
        maxfevals=effective_maxfevals,
        popsize=popsize,
        sigma0=sigma0,
        tolx=tolx,
        tolfun=tolfun,
    )
    if all_x.size == 0:
        return np.asarray([])

    order = np.argsort(all_f)
    all_x = all_x[order]
    all_f = all_f[order]

    basin_reps_idx = _cluster_candidates(
        xs=all_x,
        fs=all_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    if max_minima is not None:
        basin_reps_idx = basin_reps_idx[:max_minima]

    polished_x, polished_f = _polish_candidates(
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

    local_minima_idx = _cluster_candidates(
        xs=polished_x,
        fs=polished_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    return np.asarray(polished_x[local_minima_idx])


def _collect_cma_candidates(
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    n_runs: int,
    maxiter: int,
    seed: int,
    maxfevals: int,
    popsize: Optional[int],
    sigma0: Optional[float],
    tolx: float,
    tolfun: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect candidate minima from multiple CMA-ES runs."""
    all_x = []
    all_f = []

    x0_arr = None
    if x0_ls is not None:
        x0_arr = np.asarray(x0_ls, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)

    n_jobs_eff = max(1, int(n_runs))
    run_fn = partial(
        _run_cma_single,
        func=func,
        bounds=bounds,
        x0_ls=x0_arr,
        args=args,
        maxiter=maxiter,
        seed=seed,
        maxfevals=maxfevals,
        popsize=popsize,
        sigma0=sigma0,
        tolx=tolx,
        tolfun=tolfun,
    )

    if n_jobs_eff == 1:
        run_results = map(run_fn, range(n_runs))
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
                run_results = list(pool.map(run_fn, range(n_runs)))
        except (PermissionError, OSError, BrokenProcessPool):
            run_results = map(run_fn, range(n_runs))

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


def _run_cma_single(
    run: int,
    func: Callable,
    bounds: np.ndarray,
    x0_ls: Optional[np.ndarray],
    args: tuple,
    maxiter: int,
    seed: int,
    maxfevals: int,
    popsize: Optional[int],
    sigma0: Optional[float],
    tolx: float,
    tolfun: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Execute one CMA-ES run and record improving generation-best candidates."""
    rng = np.random.default_rng(seed + run)
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    n_dim = lb.size

    if x0_ls is not None and np.shape(x0_ls)[0] > 0:
        mean = np.array(x0_ls[run % np.shape(x0_ls)[0]], dtype=float)
    else:
        mean = rng.uniform(lb, ub)
    mean = np.clip(mean, lb, ub)

    if np.allclose(ub, lb):
        f_fixed = _evaluate_scalar_objective(func, mean, args)
        return [mean], [f_fixed]

    lambda_pop = int(popsize) if popsize is not None else (4 + int(3 * np.log(n_dim + 1)))
    lambda_pop = max(4, lambda_pop)
    mu = lambda_pop // 2
    base_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    base_weights = base_weights / np.sum(base_weights)

    step0 = _default_cma_sigma(bounds) if sigma0 is None else float(sigma0)
    sigma = max(step0, 1e-12)

    C = np.eye(n_dim)
    B = np.eye(n_dim)
    D = np.ones(n_dim)
    pc = np.zeros(n_dim)
    ps = np.zeros(n_dim)
    chi_n = np.sqrt(n_dim) * (1.0 - 1.0 / (4.0 * n_dim) + 1.0 / (21.0 * n_dim * n_dim))

    run_minima_x = []
    run_minima_f = []
    best_f = np.inf
    eval_count = 0
    stall_generations = max(20, 5 * n_dim)
    stall_counter = 0

    for generation in range(int(maxiter)):
        remaining = None if maxfevals is None else int(maxfevals) - eval_count
        if remaining is not None and remaining <= 0:
            break

        n_eval = lambda_pop if remaining is None else min(lambda_pop, remaining)
        if n_eval < 2:
            break

        arz = rng.standard_normal((n_dim, n_eval))
        ary = B @ (D[:, np.newaxis] * arz)
        arx = np.clip(mean[:, np.newaxis] + sigma * ary, lb[:, np.newaxis], ub[:, np.newaxis])

        fitness = np.empty(n_eval, dtype=float)
        for i in range(n_eval):
            fitness[i] = _evaluate_scalar_objective(func, arx[:, i], args)
        eval_count += n_eval

        order = np.argsort(fitness)
        gen_best_idx = int(order[0])
        gen_best_f = float(fitness[gen_best_idx])
        gen_best_x = np.array(arx[:, gen_best_idx], dtype=float)
        if gen_best_f + tolfun < best_f:
            best_f = gen_best_f
            stall_counter = 0
            run_minima_x.append(gen_best_x)
            run_minima_f.append(gen_best_f)
        else:
            stall_counter += 1

        n_select = min(mu, n_eval)
        weights = np.array(base_weights[:n_select], dtype=float)
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4.0 + mueff / n_dim) / (n_dim + 4.0 + 2.0 * mueff / n_dim)
        cs = (mueff + 2.0) / (n_dim + mueff + 5.0)
        c1 = 2.0 / ((n_dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n_dim + 1.0)) - 1.0) + cs

        old_mean = mean.copy()
        x_sel = arx[:, order[:n_select]]
        y_sel = (x_sel - old_mean[:, np.newaxis]) / sigma
        y_w = y_sel @ weights
        mean = np.clip(old_mean + sigma * y_w, lb, ub)

        invsqrtC_y = B @ ((B.T @ y_w) / D)
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * invsqrtC_y
        norm_ps = np.linalg.norm(ps)
        hsig_cond = norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (generation + 1))) / chi_n
        hsig = 1.0 if hsig_cond < (1.4 + 2.0 / (n_dim + 1.0)) else 0.0

        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w
        rank_mu = (y_sel * weights[np.newaxis, :]) @ y_sel.T

        C = (
            (1.0 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
            + cmu * rank_mu
        )
        C = 0.5 * (C + C.T)

        sigma *= np.exp((cs / damps) * (norm_ps / chi_n - 1.0))

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        D = np.sqrt(eigvals)
        B = eigvecs

        if sigma * np.max(D) < tolx:
            break
        if stall_counter >= stall_generations:
            break

    if not run_minima_x:
        f_mean = _evaluate_scalar_objective(func, mean, args)
        run_minima_x.append(np.array(mean, dtype=float))
        run_minima_f.append(float(f_mean))
    else:
        run_minima_x.append(np.array(run_minima_x[-1], dtype=float))
        run_minima_f.append(float(run_minima_f[-1]))

    return run_minima_x, run_minima_f


def _default_cma_sigma(bounds: np.ndarray) -> float:
    """Return a dimension-scaled default initial CMA-ES step size."""
    widths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    finite_widths = widths[np.isfinite(widths) & (widths > 0)]
    if finite_widths.size == 0:
        return 1.0
    return float(0.3 * np.mean(finite_widths))


def _evaluate_scalar_objective(
    func: Callable,
    x: np.ndarray,
    args: tuple,
) -> float:
    """Evaluate ``func`` and coerce the result to float."""
    if args:
        return float(func(x, *args))
    return float(func(x))


def _get_bo_multiminima_in_parallel(
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
    # BO controls
    n_init=None,
    acq_candidates=2048,
    lengthscale=None,
    noise=1e-8,
    xi=1e-3,
    # clustering parameters
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
):
    """Return deduplicated local minima from multi-start Bayesian optimisation.

    Parameters
    ----------
    func : callable
        Scalar objective function.
    bounds : array-like
        Per-variable lower/upper bounds.
    x0_ls : array-like, optional
        Optional initial vectors cycled across runs.
    args : tuple, default=()
        Positional arguments passed to ``func``.
    constraints : sequence, default=()
        SciPy-compatible constraints for the local polishing step.
    n_runs : int, default=6
        Number of independent BO runs.
    maxiter : int, default=300
        Maximum BO acquisition iterations per run.
    seed : int, default=0
        Base random seed. Run ``r`` uses ``seed + r``.
    maxfun, maxfevals : int, optional
        Maximum objective evaluations per run. ``maxfevals`` takes precedence.
    n_init : int, optional
        Number of initial random (or seeded) points before BO updates.
    acq_candidates : int, default=2048
        Number of random candidates used when maximizing the acquisition.
    lengthscale : float, optional
        Fixed GP lengthscale. When ``None`` it is inferred from data.
    noise : float, default=1e-8
        Observation noise added to the GP kernel diagonal.
    xi : float, default=1e-3
        Exploration parameter for expected improvement.
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
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
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
    if all_x.size == 0:
        return np.asarray([])

    order = np.argsort(all_f)
    all_x = all_x[order]
    all_f = all_f[order]

    basin_reps_idx = _cluster_candidates(
        xs=all_x,
        fs=all_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    if max_minima is not None:
        basin_reps_idx = basin_reps_idx[:max_minima]

    polished_x, polished_f = _polish_candidates(
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

    local_minima_idx = _cluster_candidates(
        xs=polished_x,
        fs=polished_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    return np.asarray(polished_x[local_minima_idx])


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
    all_x = []
    all_f = []

    x0_arr = None
    if x0_ls is not None:
        x0_arr = np.asarray(x0_ls, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)

    n_jobs_eff = max(1, int(n_runs))
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

    if n_jobs_eff == 1:
        run_results = map(run_fn, range(n_runs))
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
                run_results = list(pool.map(run_fn, range(n_runs)))
        except (PermissionError, OSError, BrokenProcessPool):
            run_results = map(run_fn, range(n_runs))

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


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
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # If GP algebra fails (ill-conditioned covariance), continue with a random step.
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
    # RBF controls
    n_init=None,
    n_candidates=2048,
    kernel="cubic",
    epsilon=1.0,
    smoothing=1e-8,
    degree=1,
    distance_tol=1e-6,
    # clustering parameters
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
):
    """Return deduplicated local minima from multi-start RBF-surrogate search.

    Parameters
    ----------
    func : callable
        Scalar objective function.
    bounds : array-like
        Per-variable lower/upper bounds.
    x0_ls : array-like, optional
        Optional initial vectors cycled across runs.
    args : tuple, default=()
        Positional arguments passed to ``func``.
    constraints : sequence, default=()
        SciPy-compatible constraints for the local polishing step.
    n_runs : int, default=6
        Number of independent RBF-surrogate runs.
    maxiter : int, default=300
        Maximum surrogate-guided iterations per run.
    seed : int, default=0
        Base random seed. Run ``r`` uses ``seed + r``.
    maxfun, maxfevals : int, optional
        Maximum objective evaluations per run. ``maxfevals`` takes precedence.
    n_init : int, optional
        Number of initial random (or seeded) points before surrogate updates.
    n_candidates : int, default=2048
        Number of random candidates scored by the merit function each step.
    kernel : str, default="cubic"
        Kernel passed to ``scipy.interpolate.RBFInterpolator``.
    epsilon : float, default=1.0
        Shape parameter passed to ``RBFInterpolator``.
    smoothing : float, default=1e-8
        Smoothing parameter passed to ``RBFInterpolator``.
    degree : int, default=1
        Polynomial degree passed to ``RBFInterpolator``.
    distance_tol : float, default=1e-6
        Minimum normalized separation target for new candidates.
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
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
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
    if all_x.size == 0:
        return np.asarray([])

    order = np.argsort(all_f)
    all_x = all_x[order]
    all_f = all_f[order]

    basin_reps_idx = _cluster_candidates(
        xs=all_x,
        fs=all_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    if max_minima is not None:
        basin_reps_idx = basin_reps_idx[:max_minima]

    polished_x, polished_f = _polish_candidates(
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

    local_minima_idx = _cluster_candidates(
        xs=polished_x,
        fs=polished_f,
        lb=lb,
        ub=ub,
        tol_norm=cluster_tol,
    )
    return np.asarray(polished_x[local_minima_idx])


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
    all_x = []
    all_f = []

    x0_arr = None
    if x0_ls is not None:
        x0_arr = np.asarray(x0_ls, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)

    n_jobs_eff = max(1, int(n_runs))
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

    if n_jobs_eff == 1:
        run_results = map(run_fn, range(n_runs))
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
                run_results = list(pool.map(run_fn, range(n_runs)))
        except (PermissionError, OSError, BrokenProcessPool):
            run_results = map(run_fn, range(n_runs))

    for run_minima_x, run_minima_f in run_results:
        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    return np.asarray(all_x), np.asarray(all_f)


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
        try:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
                run_results = list(pool.map(run_fn, range(n_runs)))
        except (PermissionError, OSError, BrokenProcessPool):
            run_results = map(run_fn, range(n_runs))

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

    if n_workers == 1:
        results = list(map(worker_fn, basin_reps_idx))
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(worker_fn, basin_reps_idx))
        except (PermissionError, OSError, BrokenProcessPool):
            results = list(map(worker_fn, basin_reps_idx))

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
