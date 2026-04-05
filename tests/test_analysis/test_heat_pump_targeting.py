"""Regression tests for heat pump targeting analysis routines."""

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.utils import *
from OpenPinch.analysis.heat_pump_targeting import (
    _balance_hot_and_cold_heat_loads_with_ambient_air,
    _compute_entropic_average_temperature_in_K,
    _compute_COP_estimate_from_carnot_limit,
    _get_Q_vals_from_T_hp_vals,
    _parse_multi_temperature_carnot_hp_state_variables,
    _compute_multi_simple_carnot_hp_opt_obj,
    _get_H_col_till_target_Q,
    _prepare_latent_hp_profile,
    _map_T_to_x_cond,
    _map_T_to_x_evap,
    _map_x_to_T_cond,
    _map_x_to_T_evap,
    _optimise_brayton_heat_pump_placement,
    _validate_vapour_hp_refrigerant_ls,
    _get_bounds_for_multi_simple_carnot_hp_opt,
    _get_x0_for_multi_single_hp_opt,
    _get_bounds_for_multi_single_hp_opt,
    _get_x0_for_cascade_hp_opt,
    _get_bounds_for_cascade_hp_opt,
    _get_x0_for_brayton_hp_opt,
    _get_bounds_for_brayton_hp_opt,
)
from OpenPinch.analysis import heat_pump_targeting as hp_targeting_module


def get_temperatures():
    """Return temperatures used by this test module."""
    return np.array([140.0, 90.0, 75.0, 60.0, 50.0, 40.0, 20.0])


def get_cold_cc():
    """Return cold cc used by this test module."""
    return np.array([1000.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def get_hot_cc():
    """Return hot cc used by this test module."""
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
    x = np.array([0.5, 0.5, 0.5])
    n_cond = 3

    T_cond, T_evap = _parse_multi_temperature_carnot_hp_state_variables(x, n_cond)

    np.testing.assert_allclose(T_cond, np.array([0.5, 0.5, 0.5]))
    np.testing.assert_allclose(T_evap, np.array([0.0]))


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


def test_get_H_col_till_target_Q_handles_zero_target_without_index_error():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])

    T_out, H_out = _get_H_col_till_target_Q(0.0, T_cold.copy(), H_cold.copy())

    np.testing.assert_allclose(T_out, np.array([130.0]))
    np.testing.assert_allclose(H_out, np.array([0.0]))


def test_balance_hot_and_cold_heat_loads_handles_excess_hot_source_branch():
    T_hot = np.array([150.0, 120.0, 90.0, 60.0])
    H_hot = np.array([0.0, -200.0, -500.0, -800.0])
    T_cold = np.array([150.0, 120.0, 90.0, 60.0])
    H_cold = np.array([400.0, 200.0, 0.0, 0.0])

    T_hot_out, H_hot_out, _, _, Q_amb_max = _balance_hot_and_cold_heat_loads_with_ambient_air(
        T_hot=T_hot,
        H_hot=H_hot,
        T_cold=T_cold,
        H_cold=H_cold,
        dtcont=5.0,
        T_env=25.0,
        dT_env_cont=10.0,
        dt_phase_change=1.0,
        is_heat_pumping=True,
    )

    np.testing.assert_allclose(np.abs(H_hot_out).max(), np.abs(H_cold).max())
    assert T_hot_out.shape == H_hot_out.shape
    assert Q_amb_max == 0.0


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


def test_compute_multi_simple_carnot_objective_handles_mixed_lift_without_ambiguous_truth():
    args = SimpleNamespace(
        n_cond=2,
        n_evap=2,
        T_cold=np.array([100.0, 70.0, 40.0]),
        H_cold=np.array([300.0, 150.0, 0.0]),
        T_hot=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -120.0, -260.0]),
        dt_range_max=80.0,
        eta_hp_carnot=0.5,
        eta_he_carnot=0.5,
        Q_hp_target=300.0,
        Q_amb_max=0.0,
        price_ratio=1.0,
    )
    x = np.array([0.0, 0.0, 0.25])

    res = _compute_multi_simple_carnot_hp_opt_obj(x, args)

    assert np.isfinite(res["obj"])
    assert np.isfinite(res["utility_tot"])
    assert res["Q_cond"].shape == (2,)


def test_optimise_brayton_heat_pump_placement_raises_on_failed_solver(monkeypatch):
    class DummyResult:
        success = False
        message = "forced failure"

    def fake_minimize(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr(hp_targeting_module, "minimize", fake_minimize)

    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        refrigerant_ls=["AIR"],
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([200.0, 100.0, 0.0]),
        T_hot=np.array([100.0, 70.0, 30.0]),
        H_hot=np.array([0.0, -80.0, -160.0]),
        dt_range_max=90.0,
        Q_hp_target=200.0,
        eta_comp=0.75,
        eta_exp=0.75,
        dt_phase_change=1.0,
        Q_amb_max=0.0,
        dtcont_hp=5.0,
        dt_env_cont=5.0,
        T_env=20.0,
    )

    with pytest.raises(ValueError, match="Brayton heat pump targeting failed"):
        _optimise_brayton_heat_pump_placement(args)


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


def test_validate_vapour_hp_refrigerant_ls_defaults_to_water_and_matches_length():
    args = SimpleNamespace(refrigerant_ls=[])
    n = 3
    refrigerants = _validate_vapour_hp_refrigerant_ls(n, args)

    assert len(refrigerants) == n
    assert refrigerants == ["water", "water", "water"]
    assert all(isinstance(ref, str) for ref in refrigerants)


def test_validate_vapour_hp_refrigerant_ls_preserves_length_when_provided():
    args = SimpleNamespace(refrigerant_ls=["Water", "Ammonia"], do_refrigerant_sort=True)
    n = 2
    refrigerants = _validate_vapour_hp_refrigerant_ls(n, args)

    assert len(refrigerants) == n
    assert refrigerants[0] == "Water"
    assert all(isinstance(ref, str) for ref in refrigerants)


def test_validate_vapour_hp_refrigerant_ls_extends_to_match_n_cond():
    args = SimpleNamespace(refrigerant_ls=["Ammonia", "Water"], do_refrigerant_sort=True)
    n = 4
    refrigerants = _validate_vapour_hp_refrigerant_ls(n, args)

    assert len(refrigerants) == n
    assert refrigerants[:2] == ["Water", "Ammonia"]
    assert refrigerants[2:] == ["Ammonia", "Ammonia"]
    assert all(isinstance(ref, str) for ref in refrigerants)


def test_multi_single_hp_x0_and_bounds_shapes_are_consistent():
    CoolProp = pytest.importorskip("CoolProp")
    _ = CoolProp  # silence lint in environments where ruff is enabled

    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        n_cond=2,
        n_evap=2,
        dt_range_max=130.0,
        dt_phase_change=1.0,
        Q_hp_target=1000.0,
        refrigerant_ls=["R134A", "R134A"],
    )

    T_cond = np.array([120.0, 90.0])
    T_evap = np.array([50.0, 20.0])
    Q_cond = np.array([600.0, 400.0])

    bnds = _get_bounds_for_multi_single_hp_opt(args)
    x0_ls = _get_x0_for_multi_single_hp_opt(T_cond, Q_cond, T_evap, args, bnds)

    assert x0_ls.shape == (1, 10)
    assert len(bnds) == x0_ls.shape[1]
    assert np.all((x0_ls[0] >= np.array([b[0] for b in bnds])) & (x0_ls[0] <= np.array([b[1] for b in bnds])))


def test_cascade_hp_x0_and_bounds_shapes_are_consistent():
    CoolProp = pytest.importorskip("CoolProp")
    _ = CoolProp  # silence lint in environments where ruff is enabled

    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        n_cond=2,
        n_evap=2,
        dt_range_max=130.0,
        dt_phase_change=1.0,
        Q_hp_target=1000.0,
        refrigerant_ls=["R134A", "R134A", "R134A"],
    )

    T_cond = np.array([120.0, 90.0])
    T_evap = np.array([50.0, 20.0])
    Q_heat = np.array([600.0, 400.0])
    Q_cool = np.array([500.0, 400.0])

    bnds = _get_bounds_for_cascade_hp_opt(args)
    x0_ls = _get_x0_for_cascade_hp_opt(T_cond, Q_heat, T_evap, Q_cool, args, bnds)

    assert x0_ls.shape == (1, 9)
    assert len(bnds) == x0_ls.shape[1]
    assert np.all((x0_ls[0] >= np.array([b[0] for b in bnds])) & (x0_ls[0] <= np.array([b[1] for b in bnds])))


def test_brayton_x0_and_bounds_shapes_are_consistent():
    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        dt_range_max=130.0,
    )

    x0 = _get_x0_for_brayton_hp_opt(args)
    bnds = _get_bounds_for_brayton_hp_opt(args)

    assert len(x0) == 4
    assert len(bnds) == 4
    assert np.all((np.array(x0) >= np.array([b[0] for b in bnds])) & (np.array(x0) <= np.array([b[1] for b in bnds])))
