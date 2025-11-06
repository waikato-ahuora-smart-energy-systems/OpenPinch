from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis.heat_pump_targeting import (
    _get_first_or_last_zero_value_idx,
    _get_extreme_temperatures_idx,
    _prepare_data_for_minimizer,
    _compute_entropic_average_temperature_in_K,
    _get_Q_from_H,
    _set_initial_values_for_condenser_and_evaporator,
    _compute_COP_estimate_from_carnot_limit,
    _convert_idx_to_temperatures,
    _get_H_vals_from_T_hp_vals,
    _parse_carnot_hp_state_temperatures,
    _get_H_col_till_target_Q,
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

    expected = (150 + 273.15) / (150 - 30) * eff

    result = _compute_COP_estimate_from_carnot_limit(T_cond, Q_cond, T_evap, Q_evap, eff=eff)

    np.testing.assert_allclose(result, expected)


def test_get_first_or_last_zero_value_idx_ccc_min_hot():
    h_vals = get_cold_cc()
    idx = _get_first_or_last_zero_value_idx(h_vals, True)
    assert idx == 2


def test_get_first_or_last_zero_value_idx_ccc_max_hot():
    h_vals = get_cold_cc()
    h_vals = h_vals - h_vals[0]
    idx = _get_first_or_last_zero_value_idx(h_vals, False)
    assert idx == 0


def test_get_first_or_last_zero_value_idx_hcc_min_cold():
    h_vals = get_hot_cc()
    h_vals = h_vals - h_vals[-1]
    idx = _get_first_or_last_zero_value_idx(h_vals, True)
    assert idx == 6


def test_get_first_or_last_zero_value_idx_hcc_max_cold():
    h_vals = get_hot_cc()
    idx = _get_first_or_last_zero_value_idx(h_vals, False)
    assert idx == 3


def test_get_extreme_temperatures_idx():
    H_hot = get_hot_cc()
    H_cold = get_cold_cc()
    idx_dict = _get_extreme_temperatures_idx(H_hot, H_cold)
    assert idx_dict["HU"][0] == 2
    assert idx_dict["HU"][1] == 0
    assert idx_dict["CU"][0] == 6
    assert idx_dict["CU"][1] == 3


def test_set_initial_values_for_condenser_and_evaporator_even_spacing():
    T_bnds = {"HU": (150.0, 200.0), "CU": (20.0, 70.0)}
    T_cond, T_evap = _set_initial_values_for_condenser_and_evaporator(2, 3, T_bnds)
    expected_cond = np.linspace(200.0, 150.0, 3)[:-1]
    expected_evap = np.linspace(70.0, 20.0, 4)[1:]

    assert len(T_cond) == 2
    assert len(T_evap) == 3
    np.testing.assert_allclose(T_cond, expected_cond)
    np.testing.assert_allclose(T_evap, expected_evap)


def test_set_initial_values_for_condenser_and_evaporator_min_segments():
    T_bnds = {"HU": (120.0, 180.0), "CU": (30.0, 90.0)}
    T_cond, T_evap = _set_initial_values_for_condenser_and_evaporator(0, -2, T_bnds)

    assert len(T_cond) == 1
    assert len(T_evap) == 1
    np.testing.assert_allclose(T_cond, [180.0])
    np.testing.assert_allclose(T_evap, [30.0])


def test_prepare_data_for_minimize_multiple_segments():
    T_cond_init = np.array([220.0, 210.0, 205.0])
    T_evap_init = np.array([40.0, 35.0, 30.0])
    T_bnds = {"HU": (200.0, 230.0), "CU": (25.0, 55.0)}

    x0, bnds = _prepare_data_for_minimizer(
        T_cond_init,
        T_evap_init,
        T_bnds["HU"],
        T_bnds["CU"],
    )

    np.testing.assert_allclose(x0, np.array([210.0, 205.0, 40.0, 35.0]))
    assert bnds == [T_bnds["HU"], T_bnds["HU"], T_bnds["CU"], T_bnds["CU"]]


def test_prepare_data_for_minimize_single_segment():
    T_cond_init = np.array([215.0])
    T_evap_init = np.array([25.0])
    T_bnds = {"HU": (210.0, 240.0), "CU": (20.0, 40.0)}

    x0, bnds = _prepare_data_for_minimizer(
        T_cond_init,
        T_evap_init,
        T_bnds["HU"],
        T_bnds["CU"],
    )

    assert isinstance(x0, np.ndarray)
    assert x0.shape == (0,)
    assert bnds == []


def test_get_Q_from_H_computes_interval_duties():
    H = np.array([500.0, 350.0, 150.0, 50.0, 0.0])
    result = _get_Q_from_H(H)
    np.testing.assert_allclose(result, np.array([150.0, 200.0, 100.0, 50.0]))


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


def test_convert_idx_to_temperatures_maps_indices_to_bounds():
    T_hot = np.array([200.0, 180.0, 140.0, 120.0])
    T_cold = np.array([150.0, 130.0, 90.0, 70.0])
    idx_dict = {"HU": (1, 3), "CU": (0, 2)}

    result = _convert_idx_to_temperatures(idx_dict, T_hot, T_cold)

    assert result["HU"] == (T_cold[1], T_cold[3])
    assert result["CU"] == (T_hot[0], T_hot[2])


def test_get_H_vals_from_T_hp_vals_appends_origin_for_condenser_and_evaporator():
    T_vals = np.array([160.0, 140.0, 120.0, 90.0])
    H_vals = np.array([700.0, 500.0, 300.0, 0.0])
    T_hp = np.array([150.0, 110.0])

    condenser_profile = _get_H_vals_from_T_hp_vals(T_hp, T_vals, H_vals, is_cond=True)
    evaporator_profile = _get_H_vals_from_T_hp_vals(T_hp, T_vals, -H_vals, is_cond=False)

    np.testing.assert_allclose(condenser_profile[:-1], np.array([600.0, 200.0]))
    assert condenser_profile[-1] == 0.0
    assert evaporator_profile[0] == 0.0
    np.testing.assert_allclose(evaporator_profile[1:], np.array([-600.0, -200.0]))


def test_parse_carnot_hp_state_temperatures_reconstructs_state_vectors():
    args = SimpleNamespace(
        n_cond=3,
        n_evap=2,
        T_cond_hi=160.0,
        T_evap_lo=40.0,
    )
    x = np.array([150.0, 140.0, 90.0])

    T_cond, T_evap = _parse_carnot_hp_state_temperatures(x, args)

    np.testing.assert_allclose(T_cond, np.array([160.0, 150.0, 140.0]))
    np.testing.assert_allclose(T_evap, np.array([90.0, 40.0]))


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
    