"""Regression tests for optimiser handles utility helpers."""

import numpy as np
import pytest

import OpenPinch.utils.blackbox_minimisers as optimiser_module
from OpenPinch.lib.enums import BB_Minimiser
from OpenPinch.utils.blackbox_minimisers import multiminima


def _convex_quadratic(x, _):
    """Convex quadratic objective used by the optimiser tests."""
    x = np.asarray(x, dtype=float)
    return {"obj": float((x[0] - 0.4) ** 2 + (x[1] + 0.25) ** 2)}


def _run_multiminima(handle):
    """Run multiminima for this test module."""
    return multiminima(
        func=_convex_quadratic,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        optimiser_handle=handle,
        opt_kwargs={
            "n_runs": 1,
            "maxiter": 6,
            "maxfun": 5_000,
            "max_minima": 2,
            "seed": 4,
        },
    )


def test_optimiser_public_api_exports_only_multiminima():
    assert optimiser_module.__all__ == ["multiminima"]


@pytest.mark.parametrize(
    "optimiser_handle",
    [
        "dual_annealing",
        "da",
        "dual-annealing",
        "dual_annealing_multiminima",
        "cmaes",
        "cma-es",
        "cmaes_multiminima",
        "bo",
        "bayes",
        "bo_multiminima",
        "rbf_surrogate",
        "rbf",
        "rbf_surrogate_multiminima",
    ],
)
def test_multiminima_accepts_supported_handles(optimiser_handle):
    minima = np.asarray(_run_multiminima(optimiser_handle), dtype=float)
    assert minima.ndim == 2
    assert minima.shape[0] >= 1
    assert minima.shape[1] == 2

    best_f = min(_convex_quadratic(x, {})["obj"] for x in minima)
    assert best_f < 1e-3


@pytest.mark.parametrize(
    "optimiser_handle",
    [
        BB_Minimiser.DA,
        BB_Minimiser.CMAES,
        BB_Minimiser.BO,
        BB_Minimiser.RBF,
    ],
)
def test_multiminima_accepts_enum_handles(optimiser_handle):
    minima = np.asarray(_run_multiminima(optimiser_handle), dtype=float)
    assert minima.ndim == 2
    assert minima.shape[0] >= 1
    assert minima.shape[1] == 2


@pytest.mark.parametrize(
    ("optimiser_handle", "backend_name", "allowed_key", "blocked_key"),
    [
        (
            BB_Minimiser.DA,
            "_get_da_multiminima_in_parallel",
            "initial_temp",
            "popsize",
        ),
        (
            BB_Minimiser.CMAES,
            "_get_cma_multiminima_in_parallel",
            "popsize",
            "initial_temp",
        ),
        (
            BB_Minimiser.BO,
            "_get_bo_multiminima_in_parallel",
            "acq_candidates",
            "n_candidates",
        ),
        (
            BB_Minimiser.RBF,
            "_get_rbf_surrogate_multiminima_in_parallel",
            "n_candidates",
            "acq_candidates",
        ),
    ],
)
def test_multiminima_uses_enum_to_pick_backend_and_filter_kwargs(
    monkeypatch,
    optimiser_handle,
    backend_name,
    allowed_key,
    blocked_key,
):
    captured = {}

    def _fake_backend(func, x0_ls, bounds, **kwargs):
        captured["kwargs"] = kwargs
        captured["obj_val"] = func(np.array([0.0, 0.0], dtype=float))
        return np.asarray([[0.0, 0.0]], dtype=float)

    monkeypatch.setattr(optimiser_module, backend_name, _fake_backend)

    minima = multiminima(
        func=_convex_quadratic,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        optimiser_handle=optimiser_handle,
        opt_kwargs={
            "n_runs": 1,
            "maxiter": 2,
            "maxfun": 20,
            "max_minima": 1,
            allowed_key: 123,
            blocked_key: 999,
        },
    )

    minima = np.asarray(minima, dtype=float)
    # assert np.isclose(np.array([0.4, -0.25]), minima).all()
    assert allowed_key in captured["kwargs"]
    assert blocked_key not in captured["kwargs"]


def test_multiminima_uses_dual_annealing_by_default():
    minima = np.asarray(
        multiminima(
            func=_convex_quadratic,
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            opt_kwargs={
                "n_runs": 1,
                "maxiter": 6,
                "maxfun": 5_000,
                "max_minima": 2,
                "seed": 4,
            },
        ),
        dtype=float,
    )
    assert minima.ndim == 2
    assert minima.shape[0] >= 1
    assert minima.shape[1] == 2


def test_multiminima_rejects_invalid_handle():
    with pytest.raises(ValueError, match="Unsupported optimiser handle"):
        _run_multiminima("not-a-real-method")


# ===== Merged from test_blackbox_minimisers_extra.py =====
"""Additional edge-branch coverage for blackbox minimisers."""

import numpy as np
import pytest

import OpenPinch.utils.blackbox_minimisers as bb
from OpenPinch.lib.enums import BB_Minimiser


class _FakePool:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable):
        return [fn(i) for i in iterable]


def _quadratic(x, *args):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def test_get_multiminima_handler_unreachable_guard(monkeypatch):
    monkeypatch.setattr(bb, "_normalise_multiminima_handle", lambda handle: object())
    with pytest.raises(ValueError, match="Unsupported optimiser handle"):
        bb._get_multiminima_handler("da")


def test_normalise_multiminima_handle_none_and_type_error():
    assert bb._normalise_multiminima_handle(None) is BB_Minimiser.DA
    with pytest.raises(TypeError, match="must be a string or BB_Minimiser"):
        bb._normalise_multiminima_handle(42)


def test_get_cma_multiminima_empty_collection(monkeypatch):
    monkeypatch.setattr(
        bb,
        "_collect_cma_candidates",
        lambda **kwargs: (np.asarray([]), np.asarray([])),
    )
    out = bb._get_cma_multiminima_in_parallel(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        n_runs=1,
    )
    assert out.size == 0


def test_get_cma_multiminima_empty_polish(monkeypatch):
    monkeypatch.setattr(
        bb,
        "_collect_cma_candidates",
        lambda **kwargs: (np.asarray([[0.1], [0.2]]), np.asarray([1.0, 2.0])),
    )
    monkeypatch.setattr(bb, "_cluster_candidates", lambda **kwargs: [0])
    monkeypatch.setattr(
        bb,
        "_polish_candidates",
        lambda **kwargs: (np.asarray([]), np.asarray([])),
    )
    out = bb._get_cma_multiminima_in_parallel(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        n_runs=1,
    )
    assert out.size == 0


def test_collect_cma_candidates_reshape_and_pool_path(monkeypatch):
    monkeypatch.setattr(bb, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        bb,
        "_run_cma_single",
        lambda run, **kwargs: ([np.array([float(run)])], [float(run)]),
    )
    xs, fs = bb._collect_cma_candidates(
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([0.3]),
        args=(),
        n_runs=2,
        maxiter=1,
        seed=0,
        maxfevals=5,
        popsize=None,
        sigma0=None,
        tolx=1e-8,
        tolfun=1e-10,
    )
    assert xs.shape == (2, 1)
    assert fs.shape == (2,)


def test_run_cma_single_fixed_bounds_branch():
    xs, fs = bb._run_cma_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[1.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=5,
        seed=0,
        maxfevals=5,
        popsize=None,
        sigma0=None,
        tolx=1e-8,
        tolfun=1e-10,
    )
    assert len(xs) == 1
    assert len(fs) == 1


def test_run_cma_single_breaks_on_remaining_budget_and_adds_fallback():
    xs, fs = bb._run_cma_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=5,
        seed=0,
        maxfevals=0,
        popsize=None,
        sigma0=None,
        tolx=1e-8,
        tolfun=1e-10,
    )
    assert len(xs) == 1
    assert np.isfinite(fs[0])


def test_run_cma_single_breaks_when_generation_eval_count_below_two():
    xs, fs = bb._run_cma_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=5,
        seed=0,
        maxfevals=1,
        popsize=None,
        sigma0=None,
        tolx=1e-8,
        tolfun=1e-10,
    )
    assert len(xs) == 1
    assert len(fs) == 1


def test_run_cma_single_stops_by_tolx_and_stall():
    xs_tolx, fs_tolx = bb._run_cma_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([[0.2]]),
        args=(),
        maxiter=50,
        seed=1,
        maxfevals=1000,
        popsize=4,
        sigma0=0.1,
        tolx=1e9,
        tolfun=1e-10,
    )
    assert len(xs_tolx) >= 1
    assert len(fs_tolx) >= 1

    xs_stall, fs_stall = bb._run_cma_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([[0.2]]),
        args=(),
        maxiter=25,
        seed=2,
        maxfevals=1000,
        popsize=4,
        sigma0=0.1,
        tolx=1e-12,
        tolfun=float("inf"),
    )
    assert len(xs_stall) >= 1
    assert len(fs_stall) >= 1


def test_run_cma_single_nan_objective_appends_mean_fallback(monkeypatch):
    monkeypatch.setattr(
        bb, "_evaluate_scalar_objective", lambda func, x, args: float("nan")
    )
    xs, fs = bb._run_cma_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=3,
        seed=0,
        maxfevals=50,
        popsize=4,
        sigma0=0.1,
        tolx=1e-8,
        tolfun=1e-10,
    )
    assert len(xs) == 1
    assert np.isnan(fs[0])


def test_default_cma_sigma_and_evaluate_scalar_objective_args_path():
    sigma = bb._default_cma_sigma(np.asarray([[2.0, 2.0]], dtype=float))
    assert sigma == 1.0

    out = bb._evaluate_scalar_objective(
        lambda x, a, b: float(np.sum(x) + a + b),
        x=np.asarray([1.0, 2.0]),
        args=(3.0, 4.0),
    )
    assert out == 10.0


def test_get_bo_multiminima_empty_paths(monkeypatch):
    monkeypatch.setattr(
        bb,
        "_collect_bo_candidates",
        lambda **kwargs: (np.asarray([]), np.asarray([])),
    )
    out0 = bb._get_bo_multiminima_in_parallel(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        n_runs=1,
    )
    assert out0.size == 0

    monkeypatch.setattr(
        bb,
        "_collect_bo_candidates",
        lambda **kwargs: (np.asarray([[0.2]]), np.asarray([1.0])),
    )
    monkeypatch.setattr(bb, "_cluster_candidates", lambda **kwargs: [0])
    monkeypatch.setattr(
        bb, "_polish_candidates", lambda **kwargs: (np.asarray([]), np.asarray([]))
    )
    out1 = bb._get_bo_multiminima_in_parallel(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        n_runs=1,
    )
    assert out1.size == 0


def test_collect_bo_candidates_reshape_and_pool_path(monkeypatch):
    monkeypatch.setattr(bb, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        bb,
        "_run_bo_single",
        lambda run, **kwargs: ([np.array([float(run)])], [float(run)]),
    )
    xs, fs = bb._collect_bo_candidates(
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([0.4]),
        args=(),
        n_runs=2,
        maxiter=1,
        seed=0,
        maxfevals=5,
        n_init=1,
        acq_candidates=32,
        lengthscale=None,
        noise=1e-8,
        xi=1e-3,
    )
    assert xs.shape == (2, 1)
    assert fs.shape == (2,)


def test_run_bo_single_fixed_bounds_branch():
    xs, fs = bb._run_bo_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[1.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=5,
        seed=0,
        maxfevals=5,
        n_init=1,
        acq_candidates=32,
        lengthscale=None,
        noise=1e-8,
        xi=1e-3,
    )
    assert len(xs) == 1
    assert len(fs) == 1


def test_run_bo_single_seeded_path_and_budget_break(monkeypatch):
    monkeypatch.setattr(
        bb,
        "_fit_bo_gp_model",
        lambda **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError()),
    )
    xs, fs = bb._run_bo_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([[0.3]]),
        args=(),
        maxiter=2,
        seed=0,
        maxfevals=1,
        n_init=1,
        acq_candidates=32,
        lengthscale=None,
        noise=1e-8,
        xi=1e-3,
    )
    assert len(xs) >= 1
    assert len(fs) >= 1


def test_run_bo_single_gp_failure_falls_back_to_random_step(monkeypatch):
    monkeypatch.setattr(
        bb,
        "_fit_bo_gp_model",
        lambda **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError()),
    )
    xs, fs = bb._run_bo_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=1,
        seed=0,
        maxfevals=2,
        n_init=1,
        acq_candidates=16,
        lengthscale=None,
        noise=1e-8,
        xi=1e-3,
    )
    assert len(xs) >= 1
    assert len(fs) >= 1


def test_run_bo_single_nan_objective_fallback_minimum(monkeypatch):
    monkeypatch.setattr(
        bb, "_evaluate_scalar_objective", lambda func, x, args: float("nan")
    )
    xs, fs = bb._run_bo_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=0,
        seed=0,
        maxfevals=3,
        n_init=2,
        acq_candidates=16,
        lengthscale=None,
        noise=1e-8,
        xi=1e-3,
    )
    assert len(xs) == 1
    assert np.isnan(fs[0])


def test_fit_bo_gp_model_finite_filter_and_no_observations_error():
    X = np.asarray([[0.0], [1.0]], dtype=float)
    y = np.asarray([np.nan, np.nan], dtype=float)
    with pytest.raises(ValueError, match="No finite BO observations"):
        bb._fit_bo_gp_model(X, y, lengthscale=None, noise=1e-8)


def test_fit_bo_gp_model_lengthscale_and_duplicate_points_branches():
    X_dup = np.asarray([[0.5], [0.5]], dtype=float)
    y_dup = np.asarray([1.0, 1.0], dtype=float)
    model_dup = bb._fit_bo_gp_model(X_dup, y_dup, lengthscale=None, noise=1e-8)
    assert model_dup["lengthscale"] > 0

    X_one = np.asarray([[0.2]], dtype=float)
    y_one = np.asarray([1.0], dtype=float)
    model_one = bb._fit_bo_gp_model(X_one, y_one, lengthscale=None, noise=1e-8)
    assert model_one["lengthscale"] > 0

    model_fixed = bb._fit_bo_gp_model(X_dup, y_dup, lengthscale=0.0, noise=1e-8)
    assert model_fixed["lengthscale"] >= 1e-6


def test_fit_bo_gp_model_cholesky_and_alpha_failure_paths(monkeypatch):
    X = np.asarray([[0.0], [1.0]], dtype=float)
    y = np.asarray([0.0, 1.0], dtype=float)

    monkeypatch.setattr(
        bb.np.linalg,
        "cholesky",
        lambda matrix: (_ for _ in ()).throw(np.linalg.LinAlgError()),
    )
    model_no_chol = bb._fit_bo_gp_model(X, y, lengthscale=None, noise=1e-8)
    assert model_no_chol["L"] is None

    monkeypatch.undo()
    monkeypatch.setattr(
        bb.np.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError()),
    )
    model_no_solve = bb._fit_bo_gp_model(X, y, lengthscale=None, noise=1e-8)
    assert np.allclose(model_no_solve["alpha"], np.zeros(2))


def test_predict_bo_gp_l_none_and_solve_failure(monkeypatch):
    model_none = {
        "X": np.asarray([[0.0]], dtype=float),
        "alpha": np.asarray([0.0], dtype=float),
        "L": None,
        "y_mean": 2.0,
        "lengthscale": 0.5,
        "signal": 1.0,
    }
    mu0, var0 = bb._predict_bo_gp(model_none, np.asarray([[0.1], [0.2]], dtype=float))
    assert np.allclose(mu0, [2.0, 2.0])
    assert np.all(var0 > 0)

    model = {
        "X": np.asarray([[0.0], [1.0]], dtype=float),
        "alpha": np.asarray([0.1, 0.2], dtype=float),
        "L": np.eye(2),
        "y_mean": 1.0,
        "lengthscale": 0.5,
        "signal": 1.0,
    }
    monkeypatch.setattr(
        bb.np.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError()),
    )
    mu1, var1 = bb._predict_bo_gp(model, np.asarray([[0.5]], dtype=float))
    assert mu1.shape == (1,)
    assert var1.shape == (1,)


def test_get_rbf_multiminima_empty_paths(monkeypatch):
    monkeypatch.setattr(
        bb,
        "_collect_rbf_surrogate_candidates",
        lambda **kwargs: (np.asarray([]), np.asarray([])),
    )
    out0 = bb._get_rbf_surrogate_multiminima_in_parallel(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        n_runs=1,
    )
    assert out0.size == 0

    monkeypatch.setattr(
        bb,
        "_collect_rbf_surrogate_candidates",
        lambda **kwargs: (np.asarray([[0.2]]), np.asarray([1.0])),
    )
    monkeypatch.setattr(bb, "_cluster_candidates", lambda **kwargs: [0])
    monkeypatch.setattr(
        bb, "_polish_candidates", lambda **kwargs: (np.asarray([]), np.asarray([]))
    )
    out1 = bb._get_rbf_surrogate_multiminima_in_parallel(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        n_runs=1,
    )
    assert out1.size == 0


def test_collect_rbf_candidates_reshape_and_pool_path(monkeypatch):
    monkeypatch.setattr(bb, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        bb,
        "_run_rbf_surrogate_single",
        lambda run, **kwargs: ([np.array([float(run)])], [float(run)]),
    )
    xs, fs = bb._collect_rbf_surrogate_candidates(
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([0.4]),
        args=(),
        n_runs=2,
        maxiter=1,
        seed=0,
        maxfevals=5,
        n_init=1,
        n_candidates=16,
        kernel="cubic",
        epsilon=1.0,
        smoothing=1e-8,
        degree=1,
        distance_tol=1e-6,
    )
    assert xs.shape == (2, 1)
    assert fs.shape == (2,)


def test_run_rbf_single_fixed_bounds_branch():
    xs, fs = bb._run_rbf_surrogate_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[1.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=5,
        seed=0,
        maxfevals=5,
        n_init=1,
        n_candidates=16,
        kernel="cubic",
        epsilon=1.0,
        smoothing=1e-8,
        degree=1,
        distance_tol=1e-6,
    )
    assert len(xs) == 1
    assert len(fs) == 1


def test_run_rbf_single_seed_and_budget_break():
    xs, fs = bb._run_rbf_surrogate_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=np.asarray([[0.3]]),
        args=(),
        maxiter=3,
        seed=0,
        maxfevals=1,
        n_init=1,
        n_candidates=16,
        kernel="cubic",
        epsilon=1.0,
        smoothing=1e-8,
        degree=1,
        distance_tol=1e-6,
    )
    assert len(xs) >= 1
    assert len(fs) >= 1


def test_run_rbf_single_nan_objective_fallback_minimum(monkeypatch):
    monkeypatch.setattr(
        bb, "_evaluate_scalar_objective", lambda func, x, args: float("nan")
    )
    xs, fs = bb._run_rbf_surrogate_single(
        run=0,
        func=_quadratic,
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        x0_ls=None,
        args=(),
        maxiter=0,
        seed=0,
        maxfevals=3,
        n_init=2,
        n_candidates=16,
        kernel="cubic",
        epsilon=1.0,
        smoothing=1e-8,
        degree=1,
        distance_tol=1e-6,
    )
    assert len(xs) == 1
    assert np.isnan(fs[0])


def test_fit_rbf_surrogate_model_branches(monkeypatch):
    X_small = np.asarray([[0.0]], dtype=float)
    y_small = np.asarray([1.0], dtype=float)
    assert bb._fit_rbf_surrogate_model(X_small, y_small, "cubic", 1.0, 1e-8, 1) is None

    monkeypatch.setattr(
        bb, "RBFInterpolator", lambda **kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    X = np.asarray([[0.0], [1.0]], dtype=float)
    y = np.asarray([0.0, 1.0], dtype=float)
    assert bb._fit_rbf_surrogate_model(X, y, "cubic", 1.0, 1e-8, 1) is None


def test_propose_rbf_candidate_model_none_returns_farthest():
    rng = np.random.default_rng(0)
    X_obs = np.asarray([[0.1], [0.2]], dtype=float)
    out = bb._propose_rbf_surrogate_candidate(
        model=None,
        X_obs=X_obs,
        rng=rng,
        n_candidates=16,
        n_dim=1,
        best_u=np.asarray([0.15]),
        merit_weight=0.5,
        distance_tol=1e-6,
    )
    assert out.shape == (1,)


def test_collect_da_candidates_pool_success(monkeypatch):
    monkeypatch.setattr(bb, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        bb,
        "_run_da_single",
        lambda run, **kwargs: ([np.array([float(run)])], [float(run)]),
    )
    xs, fs = bb._collect_da_candidates(
        func=_quadratic,
        bounds=((0.0, 1.0),),
        x0_ls=None,
        args=(),
        n_runs=2,
        maxiter=1,
        seed=0,
        initial_temp=10.0,
        restart_temp_ratio=1e-3,
        visit=2.0,
        accept=-5.0,
        maxfun=50,
    )
    assert xs.shape == (2, 1)
    assert fs.shape == (2,)


def test_cluster_candidates_zero_span_branch():
    xs = np.asarray([[1.0], [1.0]], dtype=float)
    fs = np.asarray([2.0, 1.0], dtype=float)
    lb = np.asarray([1.0], dtype=float)
    ub = np.asarray([1.0], dtype=float)
    idx = bb._cluster_candidates(xs=xs, fs=fs, lb=lb, ub=ub, tol_norm=0.1)
    assert idx == [1]


def test_polish_candidates_empty_single_worker_and_pool_paths(monkeypatch):
    out_empty = bb._polish_candidates(
        func=_quadratic,
        args=(),
        all_x=np.asarray([[0.2], [0.4]], dtype=float),
        basin_reps_idx=[],
        local_method="SLSQP",
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        constraints=(),
    )
    assert out_empty[0].size == 0

    monkeypatch.setattr(
        bb,
        "_polish_single_candidate",
        lambda idx, all_x, func, args, local_method, bounds_arg, constraints: (
            all_x[idx],
            float(idx),
        ),
    )
    monkeypatch.setattr(bb.os, "cpu_count", lambda: 1)
    x1, f1 = bb._polish_candidates(
        func=_quadratic,
        args=(),
        all_x=np.asarray([[0.2], [0.4]], dtype=float),
        basin_reps_idx=[0, 1],
        local_method="SLSQP",
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        constraints=(),
    )
    assert x1.shape == (2, 1)
    assert f1.shape == (2,)

    monkeypatch.setattr(bb.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(bb, "ProcessPoolExecutor", _FakePool)
    x2, f2 = bb._polish_candidates(
        func=_quadratic,
        args=(),
        all_x=np.asarray([[0.2], [0.4]], dtype=float),
        basin_reps_idx=[0, 1],
        local_method="SLSQP",
        bounds=np.asarray([[0.0, 1.0]], dtype=float),
        constraints=(),
    )
    assert x2.shape == (2, 1)
    assert f2.shape == (2,)


"""Regression tests for optimiser benchmarks utility helpers."""

import time

import numpy as np

from OpenPinch.utils.blackbox_minimisers import multiminima


def _rugged_noisy_surface(x, _):
    """Deterministic pseudo-noisy multimodal objective."""
    x = np.asarray(x, dtype=float)
    base = 0.25 * (x[0] + 0.8) ** 2 + 0.18 * (x[1] - 0.25) ** 2
    nonconvex = 0.7 * np.sin(2.8 * x[0]) * np.cos(3.4 * x[1])
    fuzzy = 0.05 * np.sin(29 * x[0] + 17 * x[1]) + 0.03 * np.cos(31 * x[0] - 13 * x[1])
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
    """Run benchmark for this test module."""
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
