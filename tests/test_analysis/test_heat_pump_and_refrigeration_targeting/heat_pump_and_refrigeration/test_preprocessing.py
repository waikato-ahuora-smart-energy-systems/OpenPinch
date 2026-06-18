import numpy as np
import pytest

from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import Configuration
from OpenPinch.services.heat_pump_integration.common import (
    preprocessing as hp_pre,
)


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


def test_prepare_hpr_background_profile_trims_and_builds_stream_collection():
    zone_config = type(
        "Cfg",
        (),
        {
            "T_ENV": 20.0,
            "DT_ENV_CONT": 5.0,
            "DT_CONT_HP": 5.0,
            "DT_PHASE_CHANGE": 1.0,
        },
    )()

    T_out, H_out, z_amb, streams = hp_pre._prepare_hpr_background_profile(
        Q_hpr_target=150.0,
        T_vals=np.array([120.0, 100.0, 80.0]),
        H_vals=np.array([200.0, 150.0, 0.0]),
        zone_config=zone_config,
        is_heat_pumping=True,
        is_cold=True,
    )

    assert np.max(H_out) <= 150.0
    assert z_amb.shape == T_out.shape
    assert isinstance(streams, StreamCollection)


def test_construct_hpr_target_inputs_carries_penalty_options_from_config():
    zone_config = Configuration(
        options={
            "ETA_PENALTY": 0.123,
            "RHO_PENALTY": 4.5,
            "REFRIGERANTS": " water ; r134a ",
            "MVR_FLUIDS": " Water ; R245FA ",
            "ETA_II_HE_CARNOT": 0.42,
            "ALLOW_INTEGRATED_EXPANDER": True,
        }
    )

    args = hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -50.0, -100.0]),
        H_cold=np.array([100.0, 50.0, 0.0]),
        zone_config=zone_config,
    )

    assert args.eta_penalty == pytest.approx(0.123)
    assert args.rho_penalty == pytest.approx(4.5)
    assert args.refrigerant_ls == ["WATER", "R134A"]
    assert args.mvr_fluid_ls == ["Water", "R245FA"]
    assert args.eta_ii_he_carnot == pytest.approx(0.42)


def test_add_t_amb_interval_aligns_profile_to_ambient_breakpoints():
    T_out, H_out = hp_pre._add_T_amb_interval(
        T_vals=np.array([120.0, 80.0]),
        H_vals=np.array([10.0, 0.0]),
        T_amb=100.0,
        dt_phase_change=5.0,
        is_cold=True,
    )

    np.testing.assert_allclose(T_out, np.array([120.0, 105.0, 100.0, 80.0]))
    np.testing.assert_allclose(H_out, np.array([10.0, 6.25, 5.0, 0.0]))
