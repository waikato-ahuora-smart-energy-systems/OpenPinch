import numpy as np
import pytest

import OpenPinch.utils.blackbox_minimisers as optimiser_module
from OpenPinch.lib.enums import BB_Minimiser
from OpenPinch.utils.blackbox_minimisers import multiminima


def _convex_quadratic(x, _):
    x = np.asarray(x, dtype=float)
    return {"obj": float((x[0] - 0.4) ** 2 + (x[1] + 0.25) ** 2)}


def _run_multiminima(handle):
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
