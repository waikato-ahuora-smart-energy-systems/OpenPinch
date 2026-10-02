"""Regression tests for the generic optimiser audit fixes (area 5.6)."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.optimisation.backends.dual_annealing import _run_da_single
from OpenPinch.optimisation.backends.rbf import _fit_rbf_surrogate_model
from OpenPinch.optimisation.service import _normalise_candidates


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
