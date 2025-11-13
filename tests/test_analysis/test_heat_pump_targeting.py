from types import SimpleNamespace

import numpy as np

from OpenPinch.utils import *
from OpenPinch.analysis.heat_pump_targeting import (
    _prepare_data_for_minimizer,
    _compute_entropic_average_temperature_in_K,
    _set_initial_values_for_condenser_and_evaporator,
    _compute_COP_estimate_from_carnot_limit,
    _get_Q_vals_from_T_hp_vals,
    _parse_carnot_hp_state_variables,
    _get_H_col_till_target_Q,
    _prepare_latent_hp_profile,
    _map_T_to_x_cond,
    _map_T_to_x_evap,
    _map_x_to_T_cond,
    _map_x_to_T_evap,
)


def get_temperatures():
    return np.array([140.0, 90.0, 75.0, 60.0, 50.0, 40.0, 20.0])


def get_cold_cc():
    return np.array([1000.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def get_hot_cc():
    return np.array([0.0, 0.0, 0.0, 0.0, -400.0, -400.0, -800.0])


def test_get_carnot_COP_returns_expected_value():
    T_cond = np.array([150.0, 150.0, 150.0])
    Q_cond = np.array([60.0, 25.0, 15.0])
    T_evap = np.array([30.0, 30.0])
    Q_evap = np.array([70.0, 40.0])
    eff = 0.65

    expected = ((150 + 273.15) / (150 - 30) - 1)* eff + 1

    result = _compute_COP_estimate_from_carnot_limit(T_cond, Q_cond, T_evap, Q_evap, eff=eff)

    np.testing.assert_allclose(result, expected)


def test_set_initial_values_for_condenser_and_evaporator_even_spacing():    
    T_hot, H_hot = clean_composite_curve(get_temperatures(), get_hot_cc())
    T_cold, H_cold = clean_composite_curve(get_temperatures(), get_cold_cc())

    T_cond, T_evap = _set_initial_values_for_condenser_and_evaporator(
        2, 3, T_hot, H_hot, T_cold, H_cold
    )

    expected_cond = [140., 90.]
    expected_evap = [50., 40., 20.]

    assert len(T_cond) == 2
    assert len(T_evap) == 3
    np.testing.assert_allclose(T_cond, expected_cond)
    np.testing.assert_allclose(T_evap, expected_evap)


def test_set_initial_values_for_condenser_and_evaporator_even_spacing():    
    T_hot, H_hot = clean_composite_curve(get_temperatures(), get_hot_cc())
    T_cold, H_cold = clean_composite_curve(get_temperatures(), get_cold_cc())

    T_cond, T_evap = _set_initial_values_for_condenser_and_evaporator(
        5, 10, T_hot, H_hot, T_cold, H_cold
    )

    assert len(T_cond) == 5
    assert len(T_evap) == 10


def test_set_initial_values_for_condenser_and_evaporator_min_segments():
    T_hot, H_hot = clean_composite_curve(get_temperatures(), get_hot_cc())
    T_cold, H_cold = clean_composite_curve(get_temperatures(), get_cold_cc())

    T_cond, T_evap = _set_initial_values_for_condenser_and_evaporator(
        0, -2, T_hot, H_hot, T_cold, H_cold
    )

    assert len(T_cond) == 1
    assert len(T_evap) == 1
    np.testing.assert_allclose(T_cond, [140.0])
    np.testing.assert_allclose(T_evap, [20.0])


def test_prepare_data_for_minimize_multiple_segments():
    T_cond_init = np.array([220.0, 210.0, 205.0])
    T_evap_init = np.array([40.0, 35.0, 30.0])
    T_bnds = {"HU": (200.0, 230.0), "CU": (25.0, 55.0)}
    x_cond = _map_T_to_x_cond(T_cond_init, T_bnds["HU"][1], T_bnds["HU"][1] - T_bnds["HU"][0])
    x_evap = _map_T_to_x_evap(T_evap_init, T_bnds["CU"][0], T_bnds["CU"][1] - T_bnds["CU"][0])
    x0, bnds = _prepare_data_for_minimizer(
        x_cond,
        x_evap,
    )

    np.testing.assert_allclose(x0, [0.33333333, 0.16666667, 0.16666667, 0.16666667])
    assert bnds == [(0,1), (0,1), (0,1), (0,1)]


def test_prepare_data_for_minimize_single_segment():
    T_cond_init = np.array([215.0])
    T_evap_init = np.array([25.0])
    T_bnds = {"HU": (210.0, 240.0), "CU": (20.0, 40.0)}
    x_cond = _map_T_to_x_cond(T_cond_init, T_bnds["HU"][0], T_bnds["HU"][1])
    x_evap = _map_T_to_x_evap(T_evap_init, T_bnds["CU"][0], T_bnds["CU"][1])

    x0, bnds = _prepare_data_for_minimizer(
        x_cond,
        x_evap,
    )

    np.testing.assert_allclose(x0, np.array([]))
    assert x0.shape == (0,)
    assert bnds == []


def test_compute_entropic_average_temperature_in_K_constant_temperature():
    T = np.array([60.0, 60.0, 60.0])
    Q = np.array([100.0, 150.0, 200.0])
    result = _compute_entropic_average_temperature_in_K(T, Q)
    np.testing.assert_allclose(result, 60.0 + 273.15)


def test_compute_entropic_average_temperature_in_K_zero_net_duty_uses_arithmetic_mean():
    T = np.array([40.0, 60.0, 80.0])
    Q = np.zeros_like(T)
    result = _compute_entropic_average_temperature_in_K(T, Q)
    np.testing.assert_allclose(result, T.mean() + 273.15)


def test_scale_and_unscale_cond_roundtrip_and_bounds():
    T_cond = np.array([210.0, 190.0, 175.0, 160.0])
    T_bounds = (150.0, 230.0)

    scaled = _map_T_to_x_cond(T_cond, T_bounds[1], T_bounds[1] - T_bounds[0])

    assert scaled.size == T_cond.size
    assert np.all(scaled >= -1e-12)
    assert np.all(scaled <= 1 + 1e-12)

    restored = _map_x_to_T_cond(scaled, T_bounds[1], T_bounds[1] - T_bounds[0])
    np.testing.assert_allclose(restored, T_cond)


def test_scale_and_unscale_evap_roundtrip_and_bounds():
    T_evap = np.array([65.0, 55.0, 45.0, 35.0])
    T_bounds = (20.0, 80.0)

    scaled = _map_T_to_x_evap(T_evap, T_bounds[0], T_bounds[1] - T_bounds[0])

    assert scaled.size == T_evap.size
    np.testing.assert_allclose(scaled, [0.16666667, 0.16666667, 0.16666667, 0.25])

    restored = _map_x_to_T_evap(scaled, T_bounds[0], T_bounds[1] - T_bounds[0])
    np.testing.assert_allclose(restored, T_evap)


def test_get_H_vals_from_T_hp_vals_appends_origin_for_condenser_and_evaporator():
    T_vals = np.array([160.0, 140.0, 120.0, 90.0])
    H_vals = np.array([700.0, 500.0, 300.0, 0.0])
    T_hp = np.array([150.0, 110.0])

    Q_cond = _get_Q_vals_from_T_hp_vals(T_hp, T_vals, H_vals, is_cond=True)
    Q_evap = _get_Q_vals_from_T_hp_vals(T_hp, T_vals, H_vals - H_vals[0], is_cond=False)

    np.testing.assert_allclose(Q_cond, np.array([400.0, 200.0]))
    np.testing.assert_allclose(Q_evap, np.array([100.0, 400.0]))


def test_get_H_vals_from_T_hp_vals_appends_origin_for_condenser_and_evaporator_with_out_of_range():
    T_vals = np.array([160.0, 140.0, 120.0, 90.0])
    H_vals = np.array([700.0, 500.0, 300.0, 0.0])
    T_hp = np.array([150.0, 70.0])

    Q_cond = _get_Q_vals_from_T_hp_vals(T_hp, T_vals, H_vals, is_cond=True)
    Q_evap = _get_Q_vals_from_T_hp_vals(T_hp, T_vals, H_vals - H_vals[0], is_cond=False)

    np.testing.assert_allclose(Q_cond, np.array([600.0, 0.0]))
    np.testing.assert_allclose(Q_evap, np.array([100.0, 600.0]))    


def test_parse_carnot_hp_state_temperatures_reconstructs_state_vectors():
    args = SimpleNamespace(
        n_cond=3,
        n_evap=2,
        T_cond_hi=160.0,
        T_evap_lo=40.0,
    )
    x = np.array([0.5, 0.5, 0.5])

    T_cond, T_evap = _parse_carnot_hp_state_variables(x, args)

    np.testing.assert_allclose(T_cond, np.array([0.0, 0.5, 0.5]))
    np.testing.assert_allclose(T_evap, np.array([0.5, 0.0]))


def test_get_H_col_till_target_Q_returns_full_profile_when_target_matches_peak():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 450.0, 200.0, 0.0])
    Q_hp_target = np.abs(H_cold).max()

    T_out, H_out = _get_H_col_till_target_Q(Q_hp_target, T_cold.copy(), H_cold.copy())

    np.testing.assert_allclose(T_out, T_cold)
    np.testing.assert_allclose(H_out, H_cold)


def test_get_H_col_till_target_Q_interpolates_to_target():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])
    Q_hp_target = 250.0

    T_out, H_out = _get_H_col_till_target_Q(Q_hp_target, T_cold.copy(), H_cold.copy())

    np.testing.assert_allclose(T_out[0], 144.0)
    np.testing.assert_allclose(H_out[0], Q_hp_target)
    np.testing.assert_allclose(T_out[1:], np.array([140.0, 130.0]))
    np.testing.assert_allclose(H_out[1:], np.array([150.0, 0.0]))


def test_get_H_col_till_target_Q_interpolates_to_target_at_exisiting_temperature():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])
    Q_hp_target = 150.0

    T_out, H_out = _get_H_col_till_target_Q(Q_hp_target, T_cold.copy(), H_cold.copy())

    np.testing.assert_allclose(T_out[0], 140.0)
    np.testing.assert_allclose(H_out[0], Q_hp_target)
    np.testing.assert_allclose(T_out[:], np.array([140.0, 130.0]))
    np.testing.assert_allclose(H_out[:], np.array([150.0, 0.0]))


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty():
    T_hp = [150.0, 149.7, 130.0]
    Q_hp = [10.0, 5.0, 20.0]

    T_out, Q_out = _prepare_latent_hp_profile(T_hp, Q_hp, dT_phase_change=1.0, is_hot=True)

    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty_consecutivily():
    T_hp = [150.0, 149.7, 149.01, 130.0]
    Q_hp = [10.0, 2.0, 3.0, 20.0]

    T_out, Q_out = _prepare_latent_hp_profile(T_hp, Q_hp, dT_phase_change=1.0, is_hot=True)

    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_lower_temperature():
    T_hp = [60.0, 45.0, 30.4, 30.0]
    Q_hp = [12.5, 10.0, 7.5, 5.0]

    T_out, Q_out = _prepare_latent_hp_profile(T_hp, Q_hp, dT_phase_change=1.0, is_hot=False)

    np.testing.assert_allclose(T_out, np.array([60.0, 45.0, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 10.0, 12.5]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_complex_temperature():
    T_hp = [31.5, 30.8, 30.4, 30.0]
    Q_hp = [12.5, 10.0, 7.5, 5.0]

    T_out, Q_out = _prepare_latent_hp_profile(T_hp, Q_hp, dT_phase_change=1.0, is_hot=False)

    np.testing.assert_allclose(T_out, np.array([31.5, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 22.5]))    


def test_prepare_latent_hp_profile_keeps_unique_sequences_unchanged():
    T_hp = [180.0, 160.0, 140.0]
    Q_hp = [8.0, 6.0, 4.0]

    T_out, Q_out = _prepare_latent_hp_profile(T_hp, Q_hp, dT_phase_change=0.5, is_hot=True)

    np.testing.assert_allclose(T_out, T_hp)
    np.testing.assert_allclose(Q_out, Q_hp)


def test_prepare_latent_hp_profile_handles_empty_input():
    T_hp = []
    Q_hp = []

    T_out, Q_out = _prepare_latent_hp_profile(T_hp, Q_hp, dT_phase_change=1.0, is_hot=True)

    assert len(T_out) == 0
    assert len(Q_out) == 0
