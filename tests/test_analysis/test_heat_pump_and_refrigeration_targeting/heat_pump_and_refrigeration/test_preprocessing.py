import numpy as np
import pytest

from OpenPinch.analysis.heat_pump_and_refrigeration import preprocessing as hp_pre


def test_get_reduced_bckgrd_cascade_till_Q_target_returns_full_profile_when_target_matches_peak():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 450.0, 200.0, 0.0])
    Q_hpr_target = np.abs(H_cold).max()

    T_out, H_out = hp_pre._get_reduced_bckgrd_cascade_till_Q_target(
        Q_hpr_target, T_cold.copy(), H_cold.copy()
    )

    np.testing.assert_allclose(T_out, T_cold)
    np.testing.assert_allclose(H_out, H_cold)


def test_get_reduced_bckgrd_cascade_till_Q_target_interpolates_to_target():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])

    T_out, H_out = hp_pre._get_reduced_bckgrd_cascade_till_Q_target(
        250.0, T_cold.copy(), H_cold.copy()
    )

    np.testing.assert_allclose(T_out[0], 144.0)
    np.testing.assert_allclose(H_out[0], 250.0)
    np.testing.assert_allclose(T_out[1:], np.array([140.0, 130.0]))
    np.testing.assert_allclose(H_out[1:], np.array([150.0, 0.0]))


def test_get_reduced_bckgrd_cascade_till_Q_target_interpolates_to_target_at_existing_temperature():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])

    T_out, H_out = hp_pre._get_reduced_bckgrd_cascade_till_Q_target(
        150.0, T_cold.copy(), H_cold.copy()
    )

    np.testing.assert_allclose(T_out, np.array([140.0, 130.0]))
    np.testing.assert_allclose(H_out, np.array([150.0, 0.0]))


def test_get_reduced_bckgrd_cascade_till_Q_target_handles_zero_target_without_index_error():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])

    with pytest.raises(ValueError, match="Target for heat pumping cannot be zero."):
        hp_pre._get_reduced_bckgrd_cascade_till_Q_target(
            0.0, T_cold.copy(), H_cold.copy()
        )


def test_temperature_shift_and_h_column_edge_branches():
    T_hot, T_cold = hp_pre._apply_temperature_shift_for_hpr_stream_dtmin_cont(
        np.array([100.0, 80.0]), 10.0
    )
    np.testing.assert_allclose(T_hot, np.array([90.0, 70.0]))
    np.testing.assert_allclose(T_cold, np.array([110.0, 90.0]))

    t_i0, h_i0 = hp_pre._get_reduced_bckgrd_cascade_till_Q_target(
        1000.0,
        np.array([100.0, 80.0]),
        np.array([5.0, 0.0]),
    )
    np.testing.assert_allclose(t_i0, np.array([100.0, 80.0]))
    np.testing.assert_allclose(h_i0, np.array([5.0, 0.0]))

    with pytest.raises(ValueError, match="Target for heat pumping cannot be zero."):
        hp_pre._get_reduced_bckgrd_cascade_till_Q_target(
            0.1,
            np.array([100.0, 90.0, 80.0]),
            np.array([1.0, 0.5, 0.1]),
        )
