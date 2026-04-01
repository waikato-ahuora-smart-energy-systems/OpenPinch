import time

import numpy as np

from OpenPinch.utils.blackbox_minimisers import multiminima


def _rugged_noisy_surface(x, _):
    """Deterministic pseudo-noisy multimodal objective."""
    x = np.asarray(x, dtype=float)
    base = 0.25 * (x[0] + 0.8) ** 2 + 0.18 * (x[1] - 0.25) ** 2
    nonconvex = 0.7 * np.sin(2.8 * x[0]) * np.cos(3.4 * x[1])
    fuzzy = 0.05 * np.sin(29 * x[0] + 17 * x[1]) + 0.03 * np.cos(
        31 * x[0] - 13 * x[1]
    )
    return {"obj": float(base + nonconvex + fuzzy)}


def _rippled_rosenbrock_surface(x, _):
    """Rosenbrock valley with additional ripples to induce local minima."""
    x = np.asarray(x, dtype=float)
    valley = (1.0 - x[0]) ** 2 + 40.0 * (x[1] - x[0] ** 2) ** 2
    ripples = 0.08 * np.sin(18 * x[0]) + 0.06 * np.cos(22 * x[1])
    return {"obj": float(valley + ripples)}


def _run_benchmark(
    optimiser_handle,
    objective,
    bounds,
    opt_kwargs,
):
    start = time.perf_counter()
    minima = multiminima(
        func=objective,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
        optimiser_handle=optimiser_handle,
    )
    elapsed = time.perf_counter() - start

    minima = np.asarray(minima, dtype=float)
    assert minima.ndim == 2
    assert minima.shape[0] >= 1
    assert minima.shape[1] == len(bounds)

    f_vals = np.asarray([objective(x, {})["obj"] for x in minima], dtype=float)
    return {
        "elapsed_s": float(elapsed),
        "best_f": float(np.min(f_vals)),
        "n_minima": int(minima.shape[0]),
    }


def test_benchmark_dual_annealing_vs_cmaes_on_rugged_surface():
    bounds = ((-2.5, 2.5), (-2.5, 2.5))
    opt_kwargs = {
        "n_runs": 4,
        "maxiter": 45,
        "maxfun": 15_000,
        "seed": 2,
        "max_minima": 4,
    }

    da = _run_benchmark(
        optimiser_handle="dual_annealing",
        objective=_rugged_noisy_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )
    cma = _run_benchmark(
        optimiser_handle="cmaes",
        objective=_rugged_noisy_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )
    bo = _run_benchmark(
        optimiser_handle="bo",
        objective=_rugged_noisy_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )
    rbf = _run_benchmark(
        optimiser_handle="rbf_surrogate",
        objective=_rugged_noisy_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )

    # Both optimizers should consistently locate the same low basin.
    assert da["best_f"] < -0.70
    assert cma["best_f"] < -0.70
    assert bo["best_f"] < -0.70
    assert rbf["best_f"] < -0.70
    assert cma["best_f"] <= da["best_f"] + 1e-3
    assert bo["best_f"] <= da["best_f"] + 1e-2
    assert rbf["best_f"] <= da["best_f"] + 0.05

    # Runtime check is intentionally loose to avoid platform-specific flakiness.
    assert cma["elapsed_s"] <= 5.0 * da["elapsed_s"] + 0.1
    assert bo["elapsed_s"] <= 200.0 * da["elapsed_s"] + 1.0
    assert rbf["elapsed_s"] <= 200.0 * da["elapsed_s"] + 1.0


def test_benchmark_dual_annealing_vs_cmaes_on_rippled_rosenbrock():
    bounds = ((-2.0, 2.0), (-1.0, 3.0))
    opt_kwargs = {
        "n_runs": 4,
        "maxiter": 45,
        "maxfun": 15_000,
        "seed": 2,
        "max_minima": 4,
    }

    da = _run_benchmark(
        optimiser_handle="dual_annealing",
        objective=_rippled_rosenbrock_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )
    cma = _run_benchmark(
        optimiser_handle="cmaes",
        objective=_rippled_rosenbrock_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )
    bo = _run_benchmark(
        optimiser_handle="bo",
        objective=_rippled_rosenbrock_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )
    rbf = _run_benchmark(
        optimiser_handle="rbf_surrogate",
        objective=_rippled_rosenbrock_surface,
        bounds=bounds,
        opt_kwargs=opt_kwargs,
    )

    # Both optimizers should reach the high-quality valley region.
    assert da["best_f"] < 0.05
    assert cma["best_f"] < 0.05
    assert bo["best_f"] < 0.05
    assert rbf["best_f"] < 0.05

    # On this surface dual annealing can be slightly better; keep CMA within tolerance.
    assert cma["best_f"] <= da["best_f"] + 0.15
    assert bo["best_f"] <= da["best_f"] + 0.15
    assert rbf["best_f"] <= da["best_f"] + 0.15
    assert cma["elapsed_s"] <= 5.0 * da["elapsed_s"] + 0.1
    assert bo["elapsed_s"] <= 200.0 * da["elapsed_s"] + 1.0
    assert rbf["elapsed_s"] <= 200.0 * da["elapsed_s"] + 1.0
