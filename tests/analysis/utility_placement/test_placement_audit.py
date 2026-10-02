"""Regression tests for the utility-placement audit fixes (area 6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.utility_placement.optimisation import (
    _coordinate_processing_order,
)
from OpenPinch.analysis.utility_placement.penalties import g_penalty
from OpenPinch.analysis.utility_placement.profiles import _calibrate_profile
from OpenPinch.analysis.utility_placement.units import convert_to_request_units
from OpenPinch.contracts.utility_placement import PlacementUnitSystem, QuantityValue


def _coordinate(template_key: str, field: str):
    return SimpleNamespace(
        coordinate=SimpleNamespace(
            template_key=template_key, field=SimpleNamespace(value=field)
        )
    )


# 6.1 Supply order follows template declaration


def test_supply_coordinates_are_chained_in_declaration_order():
    # Declared [sensible A, isothermal B]; the coordinate list puts the
    # isothermal level first, as the codec builds it.
    model = SimpleNamespace(
        templates=SimpleNamespace(
            all=(SimpleNamespace(key="A"), SimpleNamespace(key="B"))
        ),
        coordinates=(
            _coordinate("B", "supply_temperature"),
            _coordinate("A", "supply_temperature"),
            _coordinate("A", "temperature_span"),
        ),
    )

    assert _coordinate_processing_order(model) == [1, 0, 2]


# 6.2 Placement temperature units


def test_placement_units_must_be_temperatures():
    assert PlacementUnitSystem(absolute_temperature="K").absolute_temperature == "K"
    with pytest.raises(ValueError, match="absolute_temperature"):
        PlacementUnitSystem(absolute_temperature="kW")
    with pytest.raises(ValueError, match="temperature_difference"):
        PlacementUnitSystem(temperature_difference="bar")


def test_results_are_reported_in_the_request_units():
    supply = QuantityValue(value=25.0, unit="degC")
    span = QuantityValue(value=10.0, unit="delta_degC")

    kelvin = convert_to_request_units((supply, span), absolute="K", difference="K")

    assert kelvin[0].unit == "K"
    assert kelvin[0].value == pytest.approx(298.15)
    assert kelvin[1].value == pytest.approx(10.0)


# 6.5 Robustness


def test_rounding_noise_in_fallback_duty_is_not_penalised():
    noise = g_penalty(
        hot_fallback_duty=2e-8,
        cold_fallback_duty=0.0,
        required_hot_duty=0.0,
        required_cold_duty=0.0,
        coverage_tolerance=1e-6,
    )

    assert noise == 0.0


def test_calibration_snaps_a_tiny_residual_to_zero():
    assert _calibrate_profile((0.0, 0.0), residual_duty=1e-13) == (0.0, 0.0)
    assert _calibrate_profile((0.0, 5.0), residual_duty=-1e-13) == (0.0, 0.0)
