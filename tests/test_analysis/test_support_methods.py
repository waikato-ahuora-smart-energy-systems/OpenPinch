import math

import pytest

from OpenPinch.utils.miscellaneous import *
from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.utils import *
from OpenPinch.analysis.exergy_targeting import *

"""Test cases for the get_value function."""


def test_get_value_with_float():
    assert get_value(3.14) == 3.14


def test_get_value_with_dict():
    assert get_value({"value": 42}) == 42


def test_get_value_with_missing_dict_key():
    with pytest.raises(KeyError):
        get_value({"not_value": 10})


def test_get_value_with_valuewithunit():
    vwu = ValueWithUnit(value=99.9, units="kW")
    assert get_value(vwu) == 99.9


def test_get_value_with_int_raises():
    with pytest.raises(TypeError):
        get_value(5)  # Int is not accepted


def test_get_value_with_string_raises():
    with pytest.raises(TypeError):
        get_value("100")


"""Test cases for the find_LMTD function."""


def test_lmtd_typical_counterflow():
    """Basic counter-current case with distinct ΔT1 and ΔT2."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 80
    result = compute_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    expected = (dT1 - dT2) / math.log(dT1 / dT2)
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_equal_temperature_difference_returns_deltaT():
    """Return arithmetic mean if ΔT1 == ΔT2."""
    T_hot_in = 100
    T_hot_out = 80
    T_cold_in = 40
    T_cold_out = 60
    result = compute_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
    expected = 40  # ΔT1 = ΔT2 = 40
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_one_deltaT_zero_returns_half_sum():
    """If one ΔT is zero, fall back to arithmetic mean."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 30
    result = compute_LMTD(
        T_hot_in, T_hot_out, T_cold_in, T_cold_out
    )  # Cold fluid at constant temperature (phase change)
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    expected = (dT1 - dT2) / math.log(dT1 / dT2)
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_negative_temperature_difference_raises_error():
    """Raise error if either ΔT1 or ΔT2 < 0."""
    with pytest.raises(ValueError, match="must heat up"):
        compute_LMTD(100, 80, 90, 70)  # cold fluid cooling


def test_lmtd_hot_fluid_heats_up_invalid():
    with pytest.raises(ValueError, match="Hot fluid must cool down"):
        compute_LMTD(90, 100, 40, 80)


def test_lmtd_cold_fluid_cools_down_invalid():
    with pytest.raises(ValueError, match="Cold fluid must heat up"):
        compute_LMTD(150, 100, 80, 60)


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
