import pytest
import math
from OpenPinch.classes import *
from OpenPinch.lib import * 
from OpenPinch.analysis.support_methods import *


"""Test cases for the insert_temperature_interval_into_pt function."""

def test_insert_temperature_interval_into_pt_basic():
    pt_ls = [
        [250, 200, 100],  
        [None, 50, 100],              
        [None, 2.0, 1.0],  
        [None, 2.0, 1.0],  
        [200.0, 100.0, 0.0],
        [None, 3.0, 2.0],            
        [None, 3.0, 2.0],       
        [350.0, 200.0, 0],   
        [None, -1.0, -1.0],        
        [None, -50.0, -100.0],       
        [150.0, 100.0, 0.0],       
    ]

    expected_ls = [
        [250, 225.0, 200, 100], 
        [None, 25.0, 25.0, 100], 
        [None, 2.0, 2.0, 1.0], 
        [None, 2.0, 2.0, 1.0],
        [200.0, 150.0, 100.0, 0.0], 
        [None, 3.0, 3.0, 2.0], 
        [None, 3.0, 3.0, 2.0], 
        [350.0, 275.0, 200.0, 0], 
        [None, -1.0, -1.0, -1.0], 
        [None, -25.0, -25.0, -100.0], 
        [150.0, 125.0, 100.0, 0.0],
    ]
    pt_real = ProblemTable(pt_ls)
    expected = ProblemTable(expected_ls)
    result: ProblemTable
    result, _ = insert_temperature_interval_into_pt(pt_real, [225.0])
    
    for row_index in range(expected.shape[0]):
        for col in [PT.T.value, PT.H_COLD.value, PT.H_HOT.value, PT.H_NET.value]:
            expected_val = expected.loc[row_index, col]
            result_val = result.loc[row_index, col]

            if np.isnan(expected_val) and np.isnan(result_val):
                continue

            assert expected_val == result_val or (
                isinstance(expected_val, float) and
                isinstance(result_val, float) and
                abs(expected_val - result_val) < 1e-6
            ), f"Mismatch at row {row_index}, col {col}: expected {expected_val}, got {result_val}"

def test_insert_temperature_interval_into_gcc_basic():
    pt_ls = [
        [250, 200, 100],        
        [150.0, 100.0, 0.0]     
    ]

    expected_ls = [
        [250, 225.0, 200, 100], 
        [150.0, 125.0, 100.0, 0.0]
    ]

    pt_real, expected = ProblemTable(pt_ls), ProblemTable(expected_ls)
    result: ProblemTable

    result, _ = insert_temperature_interval_into_pt(pt_real, [225.0])
    
    for row_index in range(expected.shape[0]):
        for col in [PT.T.value, PT.H_NET.value]:
            expected_val = expected.loc[row_index, col]
            result_val = result.loc[row_index, col]

            if np.isnan(expected_val) and np.isnan(result_val):
                continue

            assert expected_val == result_val or (
                isinstance(expected_val, float) and
                isinstance(result_val, float) and
                abs(expected_val - result_val) < 1e-6
            ), f"Mismatch at row {row_index}, col {col}: expected {expected_val}, got {result_val}"


"""Tests for get_pinch_loc"""

def test_single_zero_at_start():
    df = ProblemTable({PT.H_NET.value: [0.0, 10.0, 20.0]})
    assert get_pinch_loc(df) == (0, 0, True)

def test_single_zero_at_end():
    df = ProblemTable({PT.H_NET.value: [10.0, 5.0, 0.0]})
    assert get_pinch_loc(df) == (2, 2, True)

def test_multiple_near_zeros():
    df = ProblemTable({PT.H_NET.value: [10.0, 1e-7, -1e-7, 15.0]})
    assert get_pinch_loc(df) == (1, 2, True)

def test_no_zero():
    df = ProblemTable({PT.H_NET.value: [10.0, 5.0, 1.0]})
    assert get_pinch_loc(df) == (2, 0, False)

def test_all_zeros():
    df = ProblemTable({PT.H_NET.value: [0.0, 0.0, 0.0]})
    assert get_pinch_loc(df) == (2, 0, False)

def test_random_float_noise():
    df = ProblemTable({PT.H_NET.value: [1e-9, 2e-9, -1e-9, 5e-7]})
    assert get_pinch_loc(df) == (3, 0, False)


"""Tests for get_pinch_temperatures."""
from OpenPinch.analysis.operation_analysis import get_pinch_temperatures

@pytest.mark.parametrize("case, h_vals, t_vals, expected", [
    ("standard_case", [100, 0.0, 100], [300, 250, 200], (250, 250)),
    ("pinch_at_bottom", [100, 50, 0.0], [300, 250, 200], (200, 200)),
    ("pinch_at_top", [0.0, 50, 100], [300, 250, 200], (300, 300)),
    ("no_pinch", [100, 100, 100], [300, 250, 200], (None, None)),
    ("hot_below_cold", [100, 0.0, 0.0], [300, 250, 200], (250, 250)),
])
def test_get_pinch_temperatures(case, h_vals, t_vals, expected):
    df = ProblemTable({
        PT.T.value: t_vals,
        PT.H_NET.value: h_vals,
    })
    result = get_pinch_temperatures(df)
    assert result == expected, f"{case}: expected {expected}, got {result}"


"""Tests for shift_heat_cascade."""
from OpenPinch.analysis.operation_analysis import shift_heat_cascade

def test_shift_heat_cascade_with_enum_col():
    pt = ProblemTable({
        PT.H_NET.value: [0, 100, 200],
        PT.H_HOT.value: [0, 50, 150]
    })

    shifted = shift_heat_cascade(pt, 10.0, PT.H_NET.value)
    assert shifted[PT.H_NET.value].to_list() == [10.0, 110.0, 210.0]
    assert shifted[PT.H_HOT.value].to_list() == [0, 50, 150] # Unaffected column

def test_shift_heat_cascade_with_str_col():
    pt = ProblemTable({
        PT.H_NET.value: [0, 100, 200],
        PT.H_HOT.value: [0, 50, 150]
    })

    shifted = shift_heat_cascade(pt, -25.0, PT.H_NET.value)
    assert shifted[PT.H_NET.value].to_list() == [-25.0, 75.0, 175.0]
    assert shifted[PT.H_HOT.value].to_list() == [0, 50, 150]  # Unaffected column


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
    result = find_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
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
    result = find_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
    expected = 40  # ΔT1 = ΔT2 = 40
    assert math.isclose(result, expected, rel_tol=1e-6)

def test_lmtd_one_deltaT_zero_returns_half_sum():
    """If one ΔT is zero, fall back to arithmetic mean."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 30
    result = find_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out) #Cold fluid at constant temperature (phase change)
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    expected = (dT1 - dT2) / math.log(dT1 / dT2)
    assert math.isclose(result, expected, rel_tol=1e-6)

def test_lmtd_negative_temperature_difference_raises_error():
    """Raise error if either ΔT1 or ΔT2 < 0."""
    with pytest.raises(ValueError, match="must heat up"):
        find_LMTD(100, 80, 90, 70)  # cold fluid cooling

def test_lmtd_hot_fluid_heats_up_invalid():
    with pytest.raises(ValueError, match="Hot fluid must cool down"):
        find_LMTD(90, 100, 40, 80)

def test_lmtd_cold_fluid_cools_down_invalid():
    with pytest.raises(ValueError, match="Cold fluid must heat up"):
        find_LMTD(150, 100, 80, 60)


"""Test cases for the capital_recovery_factor function."""  

def test_crf_typical_case():
    """Test with typical values for interest and years."""
    i = 0.08
    n = 10
    result = capital_recovery_factor(i, n)
    expected = i * (1 + i) ** n / ((1 + i) ** n - 1)
    assert math.isclose(result, expected, rel_tol=1e-9)

def test_crf_high_interest():
    """Test with a high interest rate."""
    result = capital_recovery_factor(0.2, 5)
    assert result > 0.2

def test_crf_long_term():
    """Test for long project duration (n=30)."""
    result = capital_recovery_factor(0.05, 30)
    assert result < 0.1

def test_crf_short_term():
    """Test with a short project duration (n=1)."""
    result = capital_recovery_factor(0.1, 1)
    expected = 1.1
    assert math.isclose(result, expected, rel_tol=1e-9)

def test_crf_zero_interest_raises():
    """Zero interest should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        capital_recovery_factor(0.0, 10)

def test_crf_zero_years_raises():
    """Zero years should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        capital_recovery_factor(0.08, 0)

def test_crf_negative_interest():
    """Negative interest should compute (though rarely used)."""
    result = capital_recovery_factor(-0.01, 10)
    assert isinstance(result, float)


"""Test cases for linear_interpolation function."""

def test_linear_interpolation_midpoint():
    assert linear_interpolation(5, 0, 10, 0, 100) == 50

def test_linear_interpolation_negative_slope():
    assert linear_interpolation(2, 0, 4, 100, 0) == 50

def test_linear_interpolation_flat_line():
    assert linear_interpolation(5, 0, 10, 20, 20) == 20

def test_linear_interpolation_returns_y1_at_x1():
    assert linear_interpolation(0, 0, 10, 5, 15) == 5

def test_linear_interpolation_returns_y2_at_x2():
    assert linear_interpolation(10, 0, 10, 5, 15) == 15

def test_linear_interpolation_same_x_raises():
    with pytest.raises(ValueError, match="x1 == x2"):
        linear_interpolation(7, 3, 3, 100, 200)


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

