"""Regression tests for miscellaneous utility helpers."""

import math

import numpy as np
import pytest

from OpenPinch.analysis.economics import compute_capital_recovery_factor
from OpenPinch.analysis.numerics import (
    g_ineq_penalty,
    interp_with_plateaus,
    make_monotonic,
)

"""Test cases for the compute_capital_recovery_factor function."""


def test_crf_typical_case():
    """Test with typical values for interest and years."""
    i = 0.08
    n = 10
    result = compute_capital_recovery_factor(i, n)
    expected = i * (1 + i) ** n / ((1 + i) ** n - 1)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_crf_high_interest():
    """Test with a high interest rate."""
    result = compute_capital_recovery_factor(0.2, 5)
    assert result > 0.2


def test_crf_long_term():
    """Test for long project duration (n=30)."""
    result = compute_capital_recovery_factor(0.05, 30)
    assert result < 0.1


def test_crf_short_term():
    """Test with a short project duration (n=1)."""
    result = compute_capital_recovery_factor(0.1, 1)
    expected = 1.1
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_crf_zero_interest_raises():
    """Zero interest should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        compute_capital_recovery_factor(0.0, 10)


def test_crf_zero_years_raises():
    """Zero years should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        compute_capital_recovery_factor(0.08, 0)


def test_crf_negative_interest():
    """Negative interest should compute (though rarely used)."""
    result = compute_capital_recovery_factor(-0.01, 10)
    assert isinstance(result, float)


def test_g_ineq_penalty_square_default():
    assert g_ineq_penalty(2.0) == pytest.approx(40.0)


def test_g_ineq_penalty_square_custom_rho():
    assert g_ineq_penalty(-3.0, rho=2.0, form="square") == pytest.approx(18.0)


def test_g_ineq_penalty_square_root_smoothing():
    g, eta, rho = -0.2, 0.5, 4.0
    expected = 0.5 * rho * (g + (g**2 + eta**2) ** 0.5)
    assert g_ineq_penalty(
        g, eta=eta, rho=rho, form="square_root_smoothing"
    ) == pytest.approx(expected)


def test_g_ineq_penalty_invalid_form():
    with pytest.raises(ValueError, match="Unrecognised penalty function"):
        g_ineq_penalty(1.0, form="invalid")


def test_interp_with_plateaus_invalid_side():
    with pytest.raises(ValueError, match="side must be"):
        interp_with_plateaus(
            h_vals=np.array([0.0, 1.0]),
            t_vals=np.array([100.0, 90.0]),
            targets=np.array([0.5]),
            side="middle",
        )


def test_interp_with_plateaus_single_point_returns_constant():
    out = interp_with_plateaus(
        h_vals=np.array([2.0]),
        t_vals=np.array([95.0]),
        targets=np.array([1.0, 2.0, 3.0]),
        side="left",
    )
    assert np.allclose(out, [95.0, 95.0, 95.0])


def test_make_monotonic_size_one_is_identity():
    values = np.array([5.0])
    out = make_monotonic(values, side="right")
    assert np.allclose(out, values)
