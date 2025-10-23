import numpy as np
import pandas as pd
import pytest

from OpenPinch.analysis.temperature_driving_force import (
    get_temperature_driving_forces,
)


def _base_curves():
    T_hot = np.array([400.0, 350.0, 300.0, 300.0])
    H_hot = np.array([0.0,   100.0, 100.0, 200.0])
    T_cold = np.array([300.0, 250.0, 100.0])
    H_cold = np.array([0.0,    50.0, 200.0])
    return T_hot, H_hot, T_cold, H_cold


def test_get_temperature_driving_forces_balanced_returns_expected_values():
    T_hot, H_hot, T_cold, H_cold = _base_curves()

    result = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)

    expected_h = np.array([0.0, 50.0, 100.0, 200.0])
    expected_delta_T1 = np.array([100.0, 125.0, 100.0])
    expected_delta_T2 = np.array([125.0, 200.0, 200.0])
    expected_dh = np.array([0.0, -80.0, -40.0])

    np.testing.assert_allclose(result["h_vals"], expected_h, atol=1e-9)
    np.testing.assert_allclose(result["delta_T1"], expected_delta_T1, atol=1e-9)
    np.testing.assert_allclose(result["delta_T2"], expected_delta_T2, atol=1e-9)
    np.testing.assert_allclose(result["dh_vals"], expected_dh, atol=1e-9)


# def test_get_temperature_driving_forces_raises_when_unbalanced():
#     T_hot = np.array([400.0, 350.0, 300.0])
#     H_hot = np.array([0.0, 110.0, 260.0])
#     T_cold = np.array([200.0, 230.0, 270.0])
#     H_cold = np.array([0.0, 90.0, 210.0])

#     with pytest.raises(ValueError, match="requires the inputted composite curves to be balanced"):
#         get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)


# def test_get_temperature_driving_forces_shifts_curves_to_common_origin():
#     T_hot, H_hot, T_cold, H_cold = _base_curves()
#     base = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)

#     H_hot_shift = H_hot + 50.0
#     H_cold_shift = H_cold + 80.0
#     shifted = get_temperature_driving_forces(T_hot, H_hot_shift, T_cold, H_cold_shift)

#     np.testing.assert_allclose(shifted["delta_T1"], base["delta_T1"], atol=1e-9)
#     np.testing.assert_allclose(shifted["delta_T2"], base["delta_T2"], atol=1e-9)
#     np.testing.assert_allclose(shifted["dh_vals"], base["dh_vals"], atol=1e-9)

#     expected_offset = np.full_like(base["h_vals"].to_numpy(), -300.0)
#     np.testing.assert_allclose(
#         shifted["h_vals"].to_numpy() - base["h_vals"].to_numpy(),
#         expected_offset,
#         atol=1e-9,
#     )


# def test_get_temperature_driving_forces_applies_minimum_delta_T():
#     T_hot, H_hot, T_cold, H_cold = _base_curves()
#     base = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)
#     adjusted = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold, min_dT=10.0)

#     np.testing.assert_allclose(adjusted["delta_T1"], base["delta_T1"] - 10.0, atol=1e-9)
#     np.testing.assert_allclose(adjusted["delta_T2"], base["delta_T2"] - 10.0, atol=1e-9)
#     np.testing.assert_allclose(adjusted["dh_vals"], base["dh_vals"], atol=1e-9)
