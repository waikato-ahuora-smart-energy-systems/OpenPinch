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


def test_clean_composite_removes_redundant_points():
    x_vals = [0, 10, 20, 30, 30]
    y_vals = [0, 5, 10, 15, 30]
    y_clean, x_clean = clean_composite_curve(y_vals, x_vals)
    assert x_clean == [0, 30]
    assert y_clean == [0, 15]