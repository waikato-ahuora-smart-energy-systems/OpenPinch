"""Regression tests for costing utility helpers."""

import math

import pytest

from OpenPinch.analysis.economics import (
    compute_annual_capital_cost,
    compute_annual_energy_cost,
    compute_capital_cost,
    compute_capital_recovery_factor,
)
from OpenPinch.domain.value import Value

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


def test_compute_capital_cost_matches_formula():
    area = 120.0
    num_units = 3
    fixed = 1500.0
    variable = 80.0
    exponent = 0.65
    expected = num_units * fixed + num_units * variable * (area / num_units) ** exponent
    result = compute_capital_cost(area, num_units, fixed, variable, exponent)
    assert isinstance(result, Value)
    assert result.unit == "$"
    assert result.value == pytest.approx(expected)


def test_compute_annual_capital_cost_matches_crf_product():
    capital_cost = 50000.0
    discount_rate = 0.08
    service_life = 12
    expected = capital_cost * compute_capital_recovery_factor(
        discount_rate, service_life
    )
    result = compute_annual_capital_cost(capital_cost, discount_rate, service_life)
    assert isinstance(result, Value)
    assert result.unit == "$/y"
    assert result.value == pytest.approx(expected)


def test_compute_annual_capital_cost_clamps_nonpositive_economic_inputs():
    result = compute_annual_capital_cost(Value(50000.0, "$"), 0.0, 0.0)

    expected = 50000.0 * compute_capital_recovery_factor(1e-6, 1.0)
    assert result.unit == "$/y"
    assert result.value == pytest.approx(expected)


def test_compute_annual_energy_cost_converts_kw_hours_and_mwh_price():
    result = compute_annual_energy_cost(10.0, 100.0, 1000.0)

    assert isinstance(result, Value)
    assert result.unit == "$/y"
    assert result.value == pytest.approx(1000.0)


def test_compute_annual_energy_cost_clamps_negative_inputs():
    result = compute_annual_energy_cost(-10.0, -100.0, -1000.0)

    assert result.unit == "$/y"
    assert result.value == pytest.approx(0.0)
