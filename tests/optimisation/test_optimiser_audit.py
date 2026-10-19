"""Regression tests for the generic optimiser audit fixes (area 5.6)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.optimisation import candidates
from OpenPinch.optimisation.backends.dual_annealing import _run_da_single
from OpenPinch.optimisation.backends.rbf import _fit_rbf_surrogate_model
from OpenPinch.optimisation.candidates import _polish_single_candidate
from OpenPinch.optimisation.errors import InvalidOptimisationProblemError
from OpenPinch.optimisation.models import OptimisationMethod, OptimisationProblem
from OpenPinch.optimisation.service import (
    _normalise_candidates,
    run_multistart_minimisation,
)


def test_rbf_surrogate_ignores_non_finite_observations():
    X = np.array([[0.1], [0.4], [0.7], [0.9]])
    y = np.array([1.0, np.nan, 0.5, np.inf])

    model = _fit_rbf_surrogate_model(
        X, y, kernel="thin_plate_spline", epsilon=1.0, smoothing=0.0, degree=1
    )

    assert model is not None
    assert np.isfinite(model(np.array([[0.5]]))).all()


def test_one_non_finite_candidate_is_skipped_not_fatal():
    points = np.array([[0.1, 0.2], [np.nan, 0.3], [0.4, 0.5]])
    objectives = np.array([2.0, 1.0, np.inf])

    candidates = _normalise_candidates(points, objectives, 2)

    assert [candidate.objective for candidate in candidates] == [2.0]


def test_dual_annealing_holds_fixed_coordinates():
    def objective(x):
        return float((x[0] - 0.3) ** 2 + (x[1] - 2.0) ** 2)

    xs, fs = _run_da_single(
        0,
        objective,
        ((0.0, 1.0), (2.0, 2.0)),
        None,
        (),
        maxiter=50,
        seed=1,
        initial_temp=5230.0,
        restart_temp_ratio=2e-5,
        visit=2.62,
        accept=-5.0,
        maxfun=2000,
    )

    best = xs[int(np.argmin(fs))]
    assert best[1] == pytest.approx(2.0)
    assert best[0] == pytest.approx(0.3, abs=1e-2)


def _shifted_quadratic(x):
    return float((x[0] - 1.0) ** 2)


@pytest.mark.parametrize(
    "method",
    [OptimisationMethod.DUAL_ANNEALING, OptimisationMethod.CMA_ES],
)
def test_a_feasible_polish_beats_an_infeasible_raw_point(method):
    # The unconstrained global search settles near x = 1, inside the
    # infeasible region x < 1.5. Its SLSQP polish moves to the constrained
    # optimum, which has a higher objective but is the only feasible point.
    problem = OptimisationProblem(
        objective=_shifted_quadratic,
        bounds=((-2.0, 2.0),),
        constraints=({"type": "ineq", "fun": lambda x: x[0] - 1.5},),
    )

    result = run_multistart_minimisation(
        problem, method=method, options={"n_runs": 1, "seed": 3}
    )

    assert result.best.point[0] == pytest.approx(1.5, abs=1e-6)
    assert result.best.objective == pytest.approx(0.25, abs=1e-6)


def test_polish_keeps_a_feasible_raw_point_over_an_infeasible_lower_one(
    monkeypatch,
):
    monkeypatch.setattr(
        candidates,
        "minimize",
        lambda **_: SimpleNamespace(x=np.array([1.0]), fun=0.0, success=True),
    )

    x, f = _polish_single_candidate(
        0,
        all_x=np.array([[1.8]]),
        all_f=np.array([0.64]),
        func=_shifted_quadratic,
        args=(),
        local_method="SLSQP",
        bounds_arg=np.array([[-2.0, 2.0]]),
        constraints=({"type": "ineq", "fun": lambda x: x[0] - 1.5},),
    )

    assert x[0] == pytest.approx(1.8)
    assert f == pytest.approx(0.64)


@pytest.mark.parametrize("local_method", ["Nelder-Mead", "Powell", "COBYLA"])
def test_every_local_method_polishes_inside_the_bounds(local_method):
    problem = OptimisationProblem(
        objective=lambda x: float((x[0] - 3.0) ** 2 + (x[1] + 3.0) ** 2),
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
    )

    result = run_multistart_minimisation(
        problem,
        options={"n_runs": 1, "seed": 3, "local_method": local_method},
    )

    assert result.best.point == pytest.approx((2.0, -2.0), abs=1e-3)
    assert result.best.objective == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("local_method", ["BFGS", "CG", "bogus"])
def test_local_methods_that_ignore_bounds_are_rejected(local_method):
    problem = OptimisationProblem(objective=_shifted_quadratic, bounds=((-2.0, 2.0),))

    with pytest.raises(InvalidOptimisationProblemError, match="local_method"):
        run_multistart_minimisation(problem, options={"local_method": local_method})
