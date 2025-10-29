import numpy as np

from OpenPinch.analysis.heat_pump_targeting import (
    _get_first_or_last_zero_value_idx,
    _get_extreme_temperatures_idx,
    _prepare_data_for_minimizer,
    _set_initial_values_for_condenser_and_evaporator,
)

def get_temperatures():
    return np.array([300.0, 250.0, 200.0, 150.0, 100.0])

def get_cold_cc():
    return np.array([1000.0, 1000.0, 800.0, 300.0, 0.0, 0.0])


def get_hot_cc():
    return np.array([0.0, 0.0, -400.0, -400.0, -1000.0, -1000.0])


def test_get_first_or_last_zero_value_idx_ccc_min_hot():
    h_vals = get_cold_cc()
    idx = _get_first_or_last_zero_value_idx(h_vals, True)
    assert idx == 4


def test_get_first_or_last_zero_value_idx_ccc_max_hot():
    h_vals = get_cold_cc()
    h_vals = h_vals - h_vals[0]
    idx = _get_first_or_last_zero_value_idx(h_vals, False)
    assert idx == 1


def test_get_first_or_last_zero_value_idx_hcc_min_cold():
    h_vals = get_hot_cc()
    h_vals = h_vals - h_vals[-1]
    idx = _get_first_or_last_zero_value_idx(h_vals, True)
    assert idx == 4


def test_get_first_or_last_zero_value_idx_hcc_max_cold():
    h_vals = get_hot_cc()
    idx = _get_first_or_last_zero_value_idx(h_vals, False)
    assert idx == 1


def test_get_extreme_temperatures_idx():
    H_hot = get_hot_cc()
    H_cold = get_cold_cc()
    idx_dict = _get_extreme_temperatures_idx(H_hot, H_cold)
    assert idx_dict["HU"][0] == 4
    assert idx_dict["HU"][1] == 1
    assert idx_dict["CU"][0] == 4
    assert idx_dict["CU"][1] == 1


def test_set_initial_values_for_condenser_and_evaporator_even_spacing():
    T_bnds = {"HU": (150.0, 200.0), "CU": (20.0, 70.0)}
    T_cond, T_evap = _set_initial_values_for_condenser_and_evaporator(2, 3, T_bnds)
    expected_cond = np.linspace(200.0, 150.0, 3)[:-1]
    expected_evap = np.linspace(20.0, 70.0, 4)[:-1]

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
    T_evap_init = np.array([30.0, 35.0, 40.0])
    T_bnds = {"HU": (200.0, 230.0), "CU": (25.0, 55.0)}

    x0, args, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)

    np.testing.assert_allclose(x0, np.array([210.0, 205.0, 35.0, 40.0]))
    assert args == {"T_cond_hi": 220.0, "T_evap_lo": 30.0}
    assert bnds == [T_bnds["HU"], T_bnds["HU"], T_bnds["CU"], T_bnds["CU"]]


def test_prepare_data_for_minimize_single_segment():
    T_cond_init = np.array([215.0])
    T_evap_init = np.array([25.0])
    T_bnds = {"HU": (210.0, 240.0), "CU": (20.0, 40.0)}

    x0, args, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)

    assert isinstance(x0, np.ndarray)
    assert x0.shape == (0,)
    assert args == {"T_cond_hi": 215.0, "T_evap_lo": 25.0}
    assert bnds == []

