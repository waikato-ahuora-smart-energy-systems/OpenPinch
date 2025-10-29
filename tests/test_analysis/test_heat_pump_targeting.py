import numpy as np

from OpenPinch.analysis.heat_pump_targeting import (
    _initialise_heat_pump_temperatures,
    _get_first_or_last_zero_value_idx,
    _get_extreme_temperatures_idx,
    _prepare_data_for_minimizer,
    _get_entropic_average_temperature,
    _set_initial_values_for_condenser_and_evaporator,
    _get_carnot_COP,
)
from OpenPinch.lib import *

def get_temperatures():
    return np.array([300.0, 250.0, 200.0, 150.0, 100.0, 50.0])


def get_cold_cc():
    return np.array([1000.0, 1000.0, 800.0, 300.0, 0.0, 0.0])


def get_hot_cc():
    return np.array([0.0, 0.0, -400.0, -400.0, -1000.0, -1000.0])


def test_get_carnot_COP_returns_expected_value():
    T_cond = np.array([150.0, 130.0, 110.0])
    H_cond = np.array([60.0, 25.0, 15.0])
    T_evap = np.array([40.0, 30.0])
    H_evap = np.array([30.0, 70.0])
    eff = 0.65

    T_hi = _get_entropic_average_temperature(T_cond, H_cond)
    T_lo = _get_entropic_average_temperature(T_evap, -H_evap)
    expected = T_hi / (T_hi - T_lo) * eff

    result = _get_carnot_COP(T_cond, H_cond, T_evap, H_evap, eff=eff)

    np.testing.assert_allclose(result, expected)


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

    x0, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)

    np.testing.assert_allclose(x0, np.array([210.0, 205.0, 35.0, 40.0]))
    assert bnds == [T_bnds["HU"], T_bnds["HU"], T_bnds["CU"], T_bnds["CU"]]


def test_prepare_data_for_minimize_single_segment():
    T_cond_init = np.array([215.0])
    T_evap_init = np.array([25.0])
    T_bnds = {"HU": (210.0, 240.0), "CU": (20.0, 40.0)}

    x0, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)

    assert isinstance(x0, np.ndarray)
    assert x0.shape == (0,)
    assert bnds == []


def test_initialise_heat_pump_temperatures_constructs_minimizer_inputs():
    T_vals = get_temperatures()
    H_cold = get_cold_cc()
    H_hot = get_hot_cc()

    x0, args, bnds = _initialise_heat_pump_temperatures(
        T_vals,
        H_hot,
        H_cold,
        n_cond=2,
        n_evap=3,
        eff_isen=0.75,
        dtmin_hp=10.0,
        is_T_vals_shifted=False,
    )

    assert isinstance(args, HeatPumpPlacementArgs)
    np.testing.assert_allclose(x0, np.array([175.0, 150.0, 200.0]))
    assert bnds == [(100.0, 250.0), (100.0, 250.0), (100.0, 250.0)]
    assert args.T_cond_hi == 250.0
    assert args.T_evap_lo == 100.0
    np.testing.assert_array_equal(args.T_vals, T_vals)
    np.testing.assert_array_equal(args.H_hot, H_hot)
    np.testing.assert_array_equal(args.H_cold, H_cold)
    assert args.n_cond == 2
    assert args.n_evap == 3
    assert args.eff_isen == 0.75
    assert args.dtmin_hp == 10.0
    assert args.is_T_vals_shifted is False
