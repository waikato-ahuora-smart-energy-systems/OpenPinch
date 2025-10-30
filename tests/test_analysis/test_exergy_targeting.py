import math

import pytest

from OpenPinch.utils.miscellaneous import *
from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.utils import *
from OpenPinch.analysis.exergy_targeting import *


"""Test cases for the compute_exergetic_temperature function."""


def test_exergetic_temp_celsius_input():
    """Basic test with Celsius input."""
    result = compute_exergetic_temperature(100, T_ref_in_C=15, units_of_T="C")
    assert isinstance(result, float)
    assert result > 0


def test_exergetic_temp_kelvin_input():
    """Basic test with Kelvin input."""
    result = compute_exergetic_temperature(373.15, T_ref_in_C=15, units_of_T="K")
    assert isinstance(result, float)
    assert result > 0


def test_exergetic_temp_at_reference_returns_zero():
    """When T = T_ref, should return 0 exergetic temp."""
    result = compute_exergetic_temperature(15, T_ref_in_C=15, units_of_T="C")
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_exergetic_temp_invalid_unit_raises():
    """Should raise for invalid temperature units."""
    with pytest.raises(ValueError, match="units must be either 'C' or 'K'"):
        compute_exergetic_temperature(100, units_of_T="F")


def test_exergetic_temp_below_absolute_zero_raises():
    """Should raise when temperature is below 0 K."""
    with pytest.raises(ValueError, match="Absolute temperature must be > 0 K"):
        compute_exergetic_temperature(-300, units_of_T="C")
