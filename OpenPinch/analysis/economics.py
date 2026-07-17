"""Utility helpers for equipment costing."""

from __future__ import annotations

from typing import Any

from ..domain.configuration import tol
from ..domain.value import Value

__all__ = [
    "compute_capital_recovery_factor",
    "compute_capital_cost",
    "compute_annual_capital_cost",
    "compute_annual_energy_cost",
]


################################################################################
# Public API
################################################################################


def compute_capital_recovery_factor(interest_rate: float, years: int) -> float:
    """Calculate the capital recovery factor, also called annualisation."""
    i = interest_rate
    n = years
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def compute_capital_cost(
    area: float,
    num_units: int,
    fixed_cost_factor: float,
    variable_cost_factor: float,
    n_exp_factor: float,
) -> Value:
    """Determine capital cost from installed capacity and unit-count assumptions."""
    capacity = max(float(area), 0.0)
    if capacity <= tol:
        return Value(0.0, "$")
    num_units = max(int(num_units), 1)
    fixed_cost_factor = max(float(fixed_cost_factor), 0.0)
    variable_cost_factor = max(float(variable_cost_factor), 0.0)
    n_exp_factor = max(float(n_exp_factor), 0.0)
    return Value(
        num_units * fixed_cost_factor
        + num_units * variable_cost_factor * (capacity / num_units) ** n_exp_factor,
        "$",
    )


def compute_annual_energy_cost(
    power_kw: float,
    price_per_mwh: float,
    annual_hours: float,
) -> Value:
    """Determine annual energy cost from power, price, and operating hours."""
    power_mw = Value(max(float(power_kw), 0.0), "kW").to("MW").value
    price = Value(max(float(price_per_mwh), 0.0), "$/MW/h").to("$/MW/h").value
    hours = max(float(annual_hours), 0.0)
    return Value(power_mw * price * hours, "$/y")


def compute_annual_capital_cost(
    capital_cost: Any,
    discount_rate: float,
    service_life: float,
) -> Value:
    """Determine the annualised capital cost."""
    capital = Value(capital_cost, "$").to("$").value
    if capital <= tol:
        return Value(0.0, "$/y")
    discount_rate = max(float(discount_rate), tol)
    service_life = max(float(service_life), 1.0)
    return Value(
        capital * compute_capital_recovery_factor(discount_rate, service_life),
        "$/y",
    )
