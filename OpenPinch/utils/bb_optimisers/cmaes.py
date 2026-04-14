"""CMA-ES multi-start backend."""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Callable, Optional

import numpy as np

from .common import (
    _cluster_candidates,
    _collect_candidates_in_parallel,
    _evaluate_scalar_objective,
    _polish_candidates,
    _postprocess_candidates,
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
    cluster_tol=0.01,
    max_minima=4,
    local_method="SLSQP",
    tolx=1e-8,
    tolfun=1e-10,
):
    """Return deduplicated local minima from multi-start CMA-ES.

    Reference
    ----------
    This pure-Python implementation follows the standard rank-mu CMA-ES update
    equations described in:

    - Hansen, N. and Ostermeier, A. (2001), "Completely Derandomized
      Self-Adaptation in Evolution Strategies", Evolutionary Computation.
      DOI: 10.1162/106365601750190398
      Source: https://pubmed.ncbi.nlm.nih.gov/11382355/
    - Hansen, N. (2016), "The CMA Evolution Strategy: A Tutorial".
      DOI: 10.48550/arXiv.1604.00772
      Source: https://arxiv.org/abs/1604.00772
    """
    bounds = np.asarray(bounds, dtype=float)
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
    x0_arr = None
    if x0_ls is not None:
        x0_arr = np.asarray(x0_ls, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)

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
    return _collect_candidates_in_parallel(
        run_fn=run_fn,
        n_runs=n_runs,
        pool_executor_cls=ProcessPoolExecutor,
        broken_pool_exc=BrokenProcessPool,
    )


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

    lambda_pop = (
        int(popsize) if popsize is not None else (4 + int(3 * np.log(n_dim + 1)))
    )
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
        arx = np.clip(
            mean[:, np.newaxis] + sigma * ary, lb[:, np.newaxis], ub[:, np.newaxis]
        )

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
        mueff = 1.0 / np.sum(weights**2)

        cc = (4.0 + mueff / n_dim) / (n_dim + 4.0 + 2.0 * mueff / n_dim)
        cs = (mueff + 2.0) / (n_dim + mueff + 5.0)
        c1 = 2.0 / ((n_dim + 1.3) ** 2 + mueff)
        cmu = min(
            1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_dim + 2.0) ** 2 + mueff)
        )
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n_dim + 1.0)) - 1.0) + cs

        old_mean = mean.copy()
        x_sel = arx[:, order[:n_select]]
        y_sel = (x_sel - old_mean[:, np.newaxis]) / sigma
        y_w = y_sel @ weights
        mean = np.clip(old_mean + sigma * y_w, lb, ub)

        invsqrtC_y = B @ ((B.T @ y_w) / D)
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * invsqrtC_y
        norm_ps = np.linalg.norm(ps)
        hsig_cond = (
            norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (generation + 1))) / chi_n
        )
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
    widths = (
        np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    )
    finite_widths = widths[np.isfinite(widths) & (widths > 0)]
    if finite_widths.size == 0:
        return 1.0
    return float(0.3 * np.mean(finite_widths))
