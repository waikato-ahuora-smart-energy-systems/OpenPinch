"""Regression tests for the LMTD heat-transfer helpers."""

import math

import pytest

import OpenPinch.analysis.heat_transfer as hx
from OpenPinch.analysis.heat_transfer import *

"""Test cases for the find_LMTD function."""


def test_lmtd_typical_counterflow():
    """Basic counter-current case with distinct ΔT1 and ΔT2."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 80
    result = compute_LMTD_from_ts(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
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
    result = compute_LMTD_from_ts(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
    expected = 40  # ΔT1 = ΔT2 = 40
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_one_deltaT_zero_returns_half_sum():
    """If one ΔT is zero, fall back to arithmetic mean."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 30
    result = compute_LMTD_from_ts(
        T_hot_in, T_hot_out, T_cold_in, T_cold_out
    )  # Cold fluid at constant temperature (phase change)
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    expected = (dT1 - dT2) / math.log(dT1 / dT2)
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_negative_temperature_difference_raises_error():
    """Raise error if either ΔT1 or ΔT2 < 0."""
    with pytest.raises(ValueError, match="must heat up"):
        compute_LMTD_from_ts(100, 80, 90, 70)  # cold fluid cooling


def test_lmtd_hot_fluid_heats_up_invalid():
    with pytest.raises(ValueError, match="Hot fluid must cool down"):
        compute_LMTD_from_ts(90, 100, 40, 80)


def test_lmtd_cold_fluid_cools_down_invalid():
    with pytest.raises(ValueError, match="Cold fluid must heat up"):
        compute_LMTD_from_ts(150, 100, 80, 60)


def test_compute_lmtd_from_dts_rejects_non_positive_values():
    with pytest.raises(ValueError, match="Invalid temperature differences"):
        hx.compute_LMTD_from_dts([10.0, -1.0], [5.0, 2.0])
