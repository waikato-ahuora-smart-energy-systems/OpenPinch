"""Utility helpers for costing heat exchangers."""


def compute_capital_recovery_factor(interest_rate: float, years: int) -> float:
    """Calculates the Capital Recovery Factor (CRF), also known as the annualisation factor."""
    i = interest_rate
    n = years
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def compute_capital_cost(
    area: float, 
    num_units: int, 
    fixed_cost_factor: float, 
    variable_cost_factor: float, 
    n_exp_factor: float,
) -> float:
    """Determine the capital cost of a heat exchanger."""
    return (
        num_units * fixed_cost_factor + num_units * variable_cost_factor * (area / num_units) ** n_exp_factor
    )


def compute_annual_capital_cost(
    capital_cost: float,   
    discount_rate: float, 
    service_life: float,
) -> float:
    """Determine the annualised capital cost of heat exchanger."""
    return capital_cost * compute_capital_recovery_factor(
        discount_rate, 
        service_life
    )

