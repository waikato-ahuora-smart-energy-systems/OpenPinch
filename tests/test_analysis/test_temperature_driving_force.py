import numpy as np
import pandas as pd
import pytest

from OpenPinch.analysis.temperature_driving_force import (
    get_temperature_driving_forces,
)


def _base_curves():
    T_hot = np.array([400.0, 350.0, 300.0, 300.0])
    H_hot = np.array([200.0,   100.0, 100.0, 0.0])
    T_cold = np.array([325.0, 250.0, 100.0])
    H_cold = np.array([200.0,    50.0, 0.0])
    return T_hot, H_hot, T_cold, H_cold


def _get_expected_base_results():
    expected_h = np.array([0.0, 50.0, 100.0, 200.0])
    expected_delta_T1 = np.array([200.0, 50.0, 75.0])
    expected_delta_T2 = np.array([50.0, 25.0, 75.0])
    expected_dh = np.array([50.0, 50.0, 100.0])
    return expected_h, expected_delta_T1, expected_delta_T2, expected_dh


def test_get_temperature_driving_forces_balanced_returns_expected_values():
    T_hot, H_hot, T_cold, H_cold = _base_curves()
    expected_h, expected_delta_T1, expected_delta_T2, expected_dh = _get_expected_base_results()

    result = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)

    np.testing.assert_allclose(result["h_vals"], expected_h, atol=1e-9)
    np.testing.assert_allclose(result["delta_T1"], expected_delta_T1, atol=1e-9)
    np.testing.assert_allclose(result["delta_T2"], expected_delta_T2, atol=1e-9)
    np.testing.assert_allclose(result["dh_vals"], expected_dh, atol=1e-9)


def test_get_temperature_driving_forces_raises_when_unbalanced():
    T_hot = np.array([400.0, 350.0, 300.0])
    H_hot = np.array([0.0, 110.0, 260.0])
    T_cold = np.array([270.0, 230.0, 200.0])
    H_cold = np.array([0.0, 90.0, 210.0])

    with pytest.raises(ValueError, match="requires the inputted composite curves to be balanced"):
        get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)


def test_get_temperature_driving_forces_shifts_curves_to_common_origin():
    T_hot, H_hot, T_cold, H_cold = _base_curves()
    base = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)

    H_hot_shift = H_hot + 50.0
    H_cold_shift = H_cold + 80.0
    shifted = get_temperature_driving_forces(T_hot, H_hot_shift, T_cold, H_cold_shift)

    np.testing.assert_allclose(shifted["h_vals"], base["h_vals"], atol=1e-9)
    np.testing.assert_allclose(shifted["delta_T1"], base["delta_T1"], atol=1e-9)
    np.testing.assert_allclose(shifted["delta_T2"], base["delta_T2"], atol=1e-9)
    np.testing.assert_allclose(shifted["dh_vals"], base["dh_vals"], atol=1e-9)


def test_get_temperature_driving_forces_applies_minimum_delta_T():
    T_hot, H_hot, T_cold, H_cold = _base_curves()
    base = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)
    adjusted = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold, min_dT=10.0)

    np.testing.assert_allclose(adjusted["delta_T1"], base["delta_T1"] - 10.0, atol=1e-9)
    np.testing.assert_allclose(adjusted["delta_T2"], base["delta_T2"] - 10.0, atol=1e-9)
    np.testing.assert_allclose(adjusted["dh_vals"], base["dh_vals"], atol=1e-9)
