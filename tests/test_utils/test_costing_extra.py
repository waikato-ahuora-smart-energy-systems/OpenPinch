"""Additional coverage tests for costing helpers."""

import pytest

from OpenPinch.utils.costing import (
    compute_annual_capital_cost,
    compute_capital_cost,
    compute_capital_recovery_factor,
)


def test_compute_capital_cost_matches_formula():
    area = 120.0
    num_units = 3
    fixed = 1500.0
    variable = 80.0
    exponent = 0.65
    expected = num_units * fixed + num_units * variable * (area / num_units) ** exponent
    assert compute_capital_cost(
        area, num_units, fixed, variable, exponent
    ) == pytest.approx(expected)


def test_compute_annual_capital_cost_matches_crf_product():
    capital_cost = 50000.0
    discount_rate = 0.08
    service_life = 12
    expected = capital_cost * compute_capital_recovery_factor(
        discount_rate, service_life
    )
    assert compute_annual_capital_cost(
        capital_cost, discount_rate, service_life
    ) == pytest.approx(expected)
