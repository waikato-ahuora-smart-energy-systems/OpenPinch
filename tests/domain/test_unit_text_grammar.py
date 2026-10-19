"""Unit text is checked against a bounded grammar before Pint parses it."""

from __future__ import annotations

import pytest

from OpenPinch.domain.value import Q_, Value, ureg


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("kW", "kW"),
        ("kW/K", "kW/K"),
        ("m2", "m^2"),
        ("m^2", "m^2"),
        ("W/m^2/K", "W/K/m^2"),
        ("kJ/(kg*K)", "kJ/K/kg"),
        ("kilowatt / kelvin ** 2", "kW/K^2"),
        ("1/K", "1/K"),
        ("m^-1", "1/m"),
        ("$/kW/y", "$/kW/a"),
        ("°C", "degC"),
        ("delta_degC", "delta_degC"),
        ("dimensionless", "-"),
    ],
)
def test_ordinary_unit_spellings_are_accepted(unit, expected):
    assert Value(1.0, unit).unit == expected


@pytest.mark.parametrize(
    "unit",
    [
        "kW^2^2",
        "kW**2**2",
        "kW*2**2",
        "kW^(2)",
        "kW^123",
        "kW/((((K))))",
        "kW)",
        "(kW",
        "kW;K",
        "kW*" + "K*" * 60,
    ],
)
def test_unit_text_outside_the_grammar_is_rejected(unit):
    with pytest.raises(ValueError, match="unit"):
        Value(1.0, unit)
    with pytest.raises(ValueError, match="unit"):
        ureg.Unit(unit)
    with pytest.raises(ValueError, match="unit"):
        Q_(1.0, unit)
    with pytest.raises(ValueError, match="unit"):
        Value(1.0, "kW").to(unit)
