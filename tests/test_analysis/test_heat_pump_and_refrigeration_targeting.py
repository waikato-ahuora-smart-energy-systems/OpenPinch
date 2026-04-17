"""Regression tests for heat pump targeting analysis routines."""

from types import SimpleNamespace
import numpy as np
import pytest
from OpenPinch.utils import *
from OpenPinch.analysis.heat_pump_and_refrigeration_targeting import (
    _compute_entropic_mean_temperature,
    _compute_COP_estimate_from_carnot_limit,
    _parse_multi_temperature_carnot_cycle_state_variables,
    _compute_multi_simple_carnot_hp_opt_obj,
    _get_reduced_bckgrd_cascade_till_Q_target,
    _get_carnot_hpr_cycle_cascade_profile,
    _map_x_arr_to_T_arr,
    _optimise_brayton_heat_pump_placement,
    _validate_vapour_hp_refrigerant_ls,
    _get_x0_for_multi_single_hp_opt,
    _get_bounds_for_multi_single_hp_opt,
    _get_x0_for_cascade_hp_opt,
    _get_bounds_for_cascade_hp_opt,
    _get_x0_for_brayton_hp_opt,
    _get_bounds_for_brayton_hp_opt,
)
from OpenPinch.analysis import heat_pump_and_refrigeration_targeting as hp
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import *
from OpenPinch.lib.enums import *


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

    expected = ((150 + 273.15) / (150 - 30) - 1) * eff + 1

    result = _compute_COP_estimate_from_carnot_limit(
        T_cond, Q_cond, T_evap, Q_evap, eff=eff
    )

    np.testing.assert_allclose(result, expected)


def test_compute_entropic_average_temperature_in_K_constant_temperature():
    T = np.array([60.0, 60.0, 60.0])
    Q = np.array([100.0, 150.0, 200.0])
    result = _compute_entropic_mean_temperature(T, Q)
    np.testing.assert_allclose(result, 60.0 + 273.15)


def test_compute_entropic_average_temperature_in_K_zero_net_duty_uses_arithmetic_mean():
    T = np.array([40.0, 60.0, 80.0])
    Q = np.zeros_like(T)
    result = _compute_entropic_mean_temperature(T, Q)
    np.testing.assert_allclose(result, T.mean() + 273.15)


def test_map_x_to_T_returns_expected_descending_temperatures():
    x = np.array([0.1, 0.2, 0.3])
    T_0 = 200.0
    T_1 = 100.0

    result = _map_x_arr_to_T_arr(x, T_0, T_1)

    np.testing.assert_allclose(result, np.array([190.0, 172.0, 150.4]))


def test_map_x_to_T_output_is_monotonically_descending():
    x = np.array([0.4, 0.2, 0.1, 0.3])
    T_0 = 180.0
    T_1 = 60.0

    result = _map_x_arr_to_T_arr(x, T_0, T_1)

    assert result.size == x.size
    assert np.all(np.diff(result) <= 0.0)


def test_get_reduced_bckgrd_cascade_till_Q_target_returns_full_profile_when_target_matches_peak():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 450.0, 200.0, 0.0])
    Q_hpr_target_max = np.abs(H_cold).max()

    T_out, H_out = _get_reduced_bckgrd_cascade_till_Q_target(
        Q_hpr_target_max, T_cold.copy(), H_cold.copy()
    )

    np.testing.assert_allclose(T_out, T_cold)
    np.testing.assert_allclose(H_out, H_cold)


def test_get_reduced_bckgrd_cascade_till_Q_target_interpolates_to_target():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])
    Q_hpr_target_max = 250.0

    T_out, H_out = _get_reduced_bckgrd_cascade_till_Q_target(
        Q_hpr_target_max, T_cold.copy(), H_cold.copy()
    )

    np.testing.assert_allclose(T_out[0], 144.0)
    np.testing.assert_allclose(H_out[0], Q_hpr_target_max)
    np.testing.assert_allclose(T_out[1:], np.array([140.0, 130.0]))
    np.testing.assert_allclose(H_out[1:], np.array([150.0, 0.0]))


def test_get_reduced_bckgrd_cascade_till_Q_target_interpolates_to_target_at_exisiting_temperature():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])
    Q_hpr_target_max = 150.0

    T_out, H_out = _get_reduced_bckgrd_cascade_till_Q_target(
        Q_hpr_target_max, T_cold.copy(), H_cold.copy()
    )

    np.testing.assert_allclose(T_out[0], 140.0)
    np.testing.assert_allclose(H_out[0], Q_hpr_target_max)
    np.testing.assert_allclose(T_out[:], np.array([140.0, 130.0]))
    np.testing.assert_allclose(H_out[:], np.array([150.0, 0.0]))


def test_get_reduced_bckgrd_cascade_till_Q_target_handles_zero_target_without_index_error():
    T_cold = np.array([160.0, 150.0, 140.0, 130.0])
    H_cold = np.array([600.0, 400.0, 150.0, 0.0])
    with pytest.raises(ValueError, match="Target for heat pumping cannot be zero."):
        _get_reduced_bckgrd_cascade_till_Q_target(0.0, T_cold.copy(), H_cold.copy())


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty():
    T_hpr = [150.0, 149.7, 130.0]
    Q_hpr = [10.0, 5.0, 20.0]

    T_out, Q_out = _get_carnot_hpr_cycle_cascade_profile(
        T_hpr, Q_hpr, dT_phase_change=1.0, is_hot=True
    )

    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty_consecutivily():
    T_hpr = [150.0, 149.7, 149.01, 130.0]
    Q_hpr = [10.0, 2.0, 3.0, 20.0]

    T_out, Q_out = _get_carnot_hpr_cycle_cascade_profile(
        T_hpr, Q_hpr, dT_phase_change=1.0, is_hot=True
    )

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
        z_amb_hot=np.zeros(3),
        z_amb_cold=np.zeros(3),
        dt_range_max=80.0,
        eta_hp_carnot=0.5,
        eta_he_carnot=0.5,
        Q_hpr_target_max=300.0,
        Q_amb_max=0.0,
        heat_to_power_ratio=1.0,
        cold_to_power_ratio=0.0,
        rho_penalty=10,
        allow_integrated_expander=False,
    )
    x = np.array([0.0, 0.0, 0.25, 0.25, 0.0])

    res = _compute_multi_simple_carnot_hp_opt_obj(x, args)

    assert np.isfinite(res["obj"])
    assert np.isfinite(res["utility_tot"])
    assert res["Q_cond"].shape == (2,)


def test_parse_multi_temperature_carnot_cycle_state_variables_returns_expected_profiles():
    args = SimpleNamespace(
        n_cond=2,
        n_evap=2,
        Q_hpr_target_max=300.0,
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([300.0, 120.0, 0.0]),
        T_hot=np.array([150.0, 90.0, 30.0]),
        H_hot=np.array([0.0, -100.0, -280.0]),
    )
    x = np.array([0.5, 0.5, 0.25, 0.5, 0.5])

    T_cond, T_evap, Q_amb_hot, Q_amb_cold = (
        _parse_multi_temperature_carnot_cycle_state_variables(x, args)
    )

    np.testing.assert_allclose(T_cond, np.array([80.0, 60.0]))
    np.testing.assert_allclose(T_evap, np.array([105.0, 60.0]))
    assert Q_amb_hot == 0.0
    assert Q_amb_cold == pytest.approx(150.0)


def test_parse_multi_temperature_carnot_cycle_state_variables_respects_cond_evap_split_sizes():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=3,
        Q_hpr_target_max=300.0,
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([300.0, 120.0, 0.0]),
        T_hot=np.array([150.0, 90.0, 30.0]),
        H_hot=np.array([0.0, -100.0, -280.0]),
    )
    x = np.array([0.4, 0.2, 0.5, 0.8, -0.1])

    T_cond, T_evap, Q_amb_hot, Q_amb_cold = (
        _parse_multi_temperature_carnot_cycle_state_variables(x, args)
    )

    assert T_cond.shape == (1,)
    assert T_evap.shape == (3,)
    assert np.all(np.diff(T_evap) <= 0.0)
    assert Q_amb_hot == pytest.approx(30.0)
    assert Q_amb_cold == 0.0


def test_optimise_brayton_heat_pump_placement_raises_on_failed_solver(monkeypatch):
    class DummyResult:
        success = False
        message = "forced failure"

    def fake_minimize(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr(hp, "minimize", fake_minimize)

    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        refrigerant_ls=["AIR"],
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([200.0, 100.0, 0.0]),
        T_hot=np.array([100.0, 70.0, 30.0]),
        H_hot=np.array([0.0, -80.0, -160.0]),
        dt_range_max=90.0,
        Q_hpr_target_max=200.0,
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
    T_hpr = [60.0, 45.0, 30.4, 30.0]
    Q_hpr = [12.5, 10.0, 7.5, 5.0]

    T_out, Q_out = _get_carnot_hpr_cycle_cascade_profile(
        T_hpr, Q_hpr, dT_phase_change=1.0, is_hot=False
    )

    np.testing.assert_allclose(T_out, np.array([60.0, 45.0, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 10.0, 12.5]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_complex_temperature():
    T_hpr = [31.5, 30.8, 30.4, 30.0]
    Q_hpr = [12.5, 10.0, 7.5, 5.0]

    T_out, Q_out = _get_carnot_hpr_cycle_cascade_profile(
        T_hpr, Q_hpr, dT_phase_change=1.0, is_hot=False
    )

    np.testing.assert_allclose(T_out, np.array([31.5, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 22.5]))


def test_prepare_latent_hp_profile_keeps_unique_sequences_unchanged():
    T_hpr = [180.0, 160.0, 140.0]
    Q_hpr = [8.0, 6.0, 4.0]

    T_out, Q_out = _get_carnot_hpr_cycle_cascade_profile(
        T_hpr, Q_hpr, dT_phase_change=0.5, is_hot=True
    )

    np.testing.assert_allclose(T_out, T_hpr)
    np.testing.assert_allclose(Q_out, Q_hpr)


def test_prepare_latent_hp_profile_handles_empty_input():
    T_hpr = []
    Q_hpr = []

    T_out, Q_out = _get_carnot_hpr_cycle_cascade_profile(
        T_hpr, Q_hpr, dT_phase_change=1.0, is_hot=True
    )

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
    args = SimpleNamespace(
        refrigerant_ls=["Water", "Ammonia"], do_refrigerant_sort=True
    )
    n = 2
    refrigerants = _validate_vapour_hp_refrigerant_ls(n, args)

    assert len(refrigerants) == n
    assert refrigerants[0] == "Water"
    assert all(isinstance(ref, str) for ref in refrigerants)


def test_validate_vapour_hp_refrigerant_ls_extends_to_match_n_cond():
    args = SimpleNamespace(
        refrigerant_ls=["Ammonia", "Water"], do_refrigerant_sort=True
    )
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
        Q_hpr_target_max=1000.0,
        refrigerant_ls=["R134A", "R134A"],
    )

    T_cond = np.array([120.0, 90.0])
    T_evap = np.array([50.0, 20.0])
    Q_cond = np.array([600.0, 400.0])

    bnds = _get_bounds_for_multi_single_hp_opt(args)
    x0_ls = _get_x0_for_multi_single_hp_opt(T_cond, Q_cond, T_evap, args, bnds)

    assert x0_ls.shape == (1, 10)
    assert len(bnds) == x0_ls.shape[1]
    assert np.all(
        (x0_ls[0] >= np.array([b[0] for b in bnds]))
        & (x0_ls[0] <= np.array([b[1] for b in bnds]))
    )


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
        Q_hpr_target_max=1000.0,
        refrigerant_ls=["R134A", "R134A", "R134A"],
    )
    init = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        T_evap=np.array([50.0, 20.0]),
        Q_cond=np.array([600.0, 400.0]),
        Q_evap=np.array([500.0, 400.0]),
    )

    bnds = _get_bounds_for_cascade_hp_opt(args)
    x0_ls = _get_x0_for_cascade_hp_opt(init_res=init, args=args, bnds=bnds)

    assert x0_ls.shape == (1, 9)
    assert len(bnds) == x0_ls.shape[1]
    assert np.all(
        (x0_ls[0] >= np.array([b[0] for b in bnds]))
        & (x0_ls[0] <= np.array([b[1] for b in bnds]))
    )


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
    assert np.all(
        (np.array(x0) >= np.array([b[0] for b in bnds]))
        & (np.array(x0) <= np.array([b[1] for b in bnds]))
    )


def _sc(*streams):
    coll = StreamCollection()
    for s in streams:
        coll.add(s)
    return coll


def _base_args(**overrides):
    args = {
        "system_type": HPRcycle.MultiTempCarnot.value,
        "Q_hpr_target_max": 200.0,
        "Q_amb_max": 20.0,
        "dt_range_max": 110.0,
        "T_hot": np.array([140.0, 90.0, 40.0]),
        "H_hot": np.array([0.0, -80.0, -160.0]),
        "T_cold": np.array([130.0, 80.0, 30.0]),
        "H_cold": np.array([200.0, 100.0, 0.0]),
        "z_amb_hot": np.zeros(3),
        "z_amb_cold": np.zeros(3),
        "n_cond": 2,
        "n_evap": 2,
        "eta_comp": 0.75,
        "eta_exp": 0.7,
        "dtcont_hp": 5.0,
        "dt_hp_ihx": 3.0,
        "dt_cascade_hx": 2.0,
        "dt_phase_change": 1.0,
        "heat_to_power_ratio": 1.0,
        "cold_to_power_ratio": 0.0,
        "load_fraction": 1.0,
        "is_heat_pumping": True,
        "max_multi_start": 2,
        "T_env": 20.0,
        "dt_env_cont": 5.0,
        "eta_hp_carnot": 0.6,
        "eta_he_carnot": 0.4,
        "refrigerant_ls": ["R134A", "R134A", "R134A"],
        "do_refrigerant_sort": False,
        "initialise_simulated_cycle": True,
        "allow_integrated_expander": True,
        "dT_subcool": None,
        "dT_superheat": None,
        "bckgrd_hot_streams": _sc(
            Stream(name="H", t_supply=120.0, t_target=80.0, heat_flow=50.0)
        ),
        "bckgrd_cold_streams": _sc(
            Stream(name="C", t_supply=30.0, t_target=60.0, heat_flow=40.0)
        ),
        "bb_minimiser": "rbf",
        "eta_penalty": 0.001,
        "rho_penalty": 10.0,
        "debug": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _pt_with_hnet(h0, h1):
    return ProblemTable({PT.T.value: [120.0, 60.0], PT.H_NET.value: [h0, h1]})


def _patch_output_model_validate(monkeypatch):
    monkeypatch.setattr(
        hp.HPRTargetOutputs,
        "model_validate",
        classmethod(lambda cls, v: v),
    )


@pytest.mark.parametrize(
    ("q_amb_hot", "q_amb_cold", "expected_hot", "expected_cold", "expected_w_air"),
    [
        (
            20.0,
            0.0,
            np.array([-2.0, 3.0]),
            np.array([3.0, 3.0]),
            np.array([5.0, 0.0]),
        ),
        (
            0.0,
            20.0,
            np.array([2.0, 2.0]),
            np.array([7.0, 2.0]),
            np.array([5.0, 0.0]),
        ),
        (
            0.0,
            0.0,
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.array([1.0, 1.0]),
        ),
    ],
)
def test_calc_heat_pump_and_refrigeration_cascade_branches(
    monkeypatch, q_amb_hot, q_amb_cold, expected_hot, expected_cold, expected_w_air
):
    pt = ProblemTable(
        {
            PT.T.value: [120.0, 60.0],
            PT.H_NET_A.value: [1.0, 1.0],
            PT.H_NET_HOT.value: [2.0, 2.0],
            PT.H_NET_COLD.value: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted=True: ProblemTable({PT.T.value: [120.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **_kwargs: {
            PT.H_NET_UT.value: np.array([0.0, 0.0]),
            PT.H_HOT_UT.value: np.array([0.0, 0.0]),
            PT.H_COLD_UT.value: np.array([0.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        hp, "get_process_heat_cascade", lambda **_kwargs: _pt_with_hnet(4.0, -1.0)
    )

    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        Q_amb_hot=q_amb_hot,
        Q_amb_cold=q_amb_cold,
        amb_streams=hp._get_ambient_air_stream(q_amb_hot, q_amb_cold, _base_args()),
    )
    out = hp.calc_heat_pump_and_refrigeration_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
    )
    assert isinstance(out, ProblemTable)
    np.testing.assert_allclose(out.col[PT.H_NET_HOT.value], expected_hot)
    np.testing.assert_allclose(out.col[PT.H_NET_COLD.value], expected_cold)
    np.testing.assert_allclose(out.col[PT.H_NET_W_AIR.value], expected_w_air)


def test_multi_temp_carnot_optimiser_success_and_failure(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    _patch_output_model_validate(monkeypatch)
    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([[0.2, 0.6]]))

    monkeypatch.setattr(
        hp,
        "_compute_multi_temperature_carnot_cycle_obj",
        lambda x, args, debug=False: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "net_work": 0.5,
            "Q_ext": 0.0,
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "cop_h": 3.0,
            "success": True,
            "T_cond": np.array([100.0]),
            "Q_cond": np.array([50.0]),
            "T_evap": np.array([60.0]),
            "Q_evap": np.array([40.0]),
        },
    )
    monkeypatch.setattr(
        hp,
        "_get_carnot_hpr_cycle_streams",
        lambda *_args, **_kwargs: {
            "hot_streams": StreamCollection(),
            "cold_streams": StreamCollection(),
        },
    )
    out = hp._optimise_multi_temperature_carnot_heat_pump_placement(args)
    assert out["success"] is True

    monkeypatch.setattr(
        hp,
        "_compute_multi_temperature_carnot_cycle_obj",
        lambda x, args, debug=False: {"success": False},
    )
    with pytest.raises(ValueError, match="failed to return an optimal result"):
        hp._optimise_multi_temperature_carnot_heat_pump_placement(args)


def test_multi_temp_carnot_objective_debug_branch(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    x = np.array([0.2, 0.7, 0.0])
    called = {"plot": 0}
    monkeypatch.setattr(
        hp,
        "plot_multi_hp_profiles_from_results",
        lambda *a, **k: called.__setitem__("plot", called["plot"] + 1),
    )
    out = hp._compute_multi_temperature_carnot_cycle_obj(x, args, debug=True)
    assert out["success"] is True
    assert called["plot"] == 1


def test_cascade_optimiser_and_compute_branches(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_cycle=True)
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp,
        "_optimise_multi_temperature_carnot_heat_pump_placement",
        lambda _args: SimpleNamespace(
            T_cond=np.array([100.0, 80.0]),
            Q_cond=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
            Q_evap=np.array([90.0, 60.0]),
        ),
    )
    monkeypatch.setattr(
        hp, "_validate_vapour_hp_refrigerant_ls", lambda n, a: ["R134A"] * n
    )
    monkeypatch.setattr(
        hp, "_get_bounds_for_cascade_hp_opt", lambda _args: [(0.0, 1.0)] * 9
    )
    monkeypatch.setattr(
        hp, "_get_x0_for_cascade_hp_opt", lambda **_kwargs: np.array([[0.5] * 9])
    )
    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([[0.5] * 9]))
    monkeypatch.setattr(
        hp,
        "_compute_cascade_hp_system_performance",
        lambda x, args, debug=False: {
            "obj": 0.2,
            "utility_tot": 1.0,
            "net_work": np.array([1.0, 2.0]),
            "Q_ext": 0.0,
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "cop_h": 2.0,
            "hot_streams": StreamCollection(),
            "cold_streams": StreamCollection(),
        },
    )

    out = hp._optimise_cascade_heat_pump_placement(args)
    assert out["success"] is True

    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([]))
    with pytest.raises(ValueError, match="failed"):
        hp._optimise_cascade_heat_pump_placement(args)


def test_cascade_x0_bounds_and_parse(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    bnds = [(0.0, 0.2)] * 9
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
    )
    x0 = hp._get_x0_for_cascade_hp_opt(
        init_res=init_res,
        args=args,
        bnds=bnds,
    )
    assert x0.shape == (1, 9)
    assert np.all(x0[0] >= 0.0)

    with pytest.raises(ValueError, match="Bounds size must match x0 size"):
        hp._get_x0_for_cascade_hp_opt(
            init_res=init_res,
            args=args,
            bnds=[(0.0, 1.0)],
        )

    monkeypatch.setattr(
        hp, "PropsSI", lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0
    )
    b = hp._get_bounds_for_cascade_hp_opt(args)
    assert len(b) == 9

    T_cond, dT_sc, Q_heat, T_evap, Q_cool = hp._parse_cascade_hp_state_variables(
        np.array([0.1] * 9), args
    )
    assert T_cond.shape == (2,)
    assert dT_sc.shape == (2,)
    assert T_evap.shape == (2,)
    assert Q_cool.shape[0] == 2


def test_compute_cascade_hp_system_performance_unsolved_and_solved(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    x = np.array([0.2] * 9)

    class _FakeCascadeUnsolved:
        solved = False
        work = 50.0

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(hp, "CascadeVapourCompressionCycle", _FakeCascadeUnsolved)
    out_unsolved = hp._compute_cascade_hp_system_performance(x, args)
    assert "obj" in out_unsolved

    class _FakeCascadeSolved:
        solved = True
        work = 40.0
        work_arr = np.array([10.0, 30.0])
        Q_cool = 70.0
        Q_heat_arr = np.array([120.0, 80.0])
        Q_cool_arr = np.array([50.0, 20.0])
        penalty = 1.0

        def solve(self, **_kwargs):
            return None

        def build_stream_collection(self, **_kwargs):
            return _sc(
                Stream(
                    name="HP",
                    t_supply=100.0,
                    t_target=90.0,
                    heat_flow=10.0,
                    is_process_stream=False,
                )
            )

    monkeypatch.setattr(hp, "CascadeVapourCompressionCycle", _FakeCascadeSolved)
    seq = iter([_pt_with_hnet(5.0, -1.0), _pt_with_hnet(6.0, -2.0)])
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **_kwargs: next(seq))
    calls = {"plot": 0}
    monkeypatch.setattr(
        hp,
        "plot_multi_hp_profiles_from_results",
        lambda *a, **k: calls.__setitem__("plot", calls["plot"] + 1),
    )
    out = hp._compute_cascade_hp_system_performance(x, args, debug=True)
    assert "hpr_hot_streams" in out
    assert calls["plot"] == 1


def test_multi_simple_carnot_and_multi_simple_simulated_paths(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_cycle=True)
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp, "multiminima", lambda **_kwargs: np.array([[0.4, 0.4, 0.5, 0.5]])
    )
    monkeypatch.setattr(
        hp,
        "_get_carnot_hpr_cycle_streams",
        lambda *_args, **_kwargs: {
            "hot_streams": StreamCollection(),
            "cold_streams": StreamCollection(),
        },
    )
    out_carnot = hp._optimise_multi_simple_carnot_heat_pump_placement(args)
    assert out_carnot["success"] is True

    monkeypatch.setattr(
        hp,
        "_optimise_multi_simple_carnot_heat_pump_placement",
        lambda _args: SimpleNamespace(
            T_cond=np.array([120.0, 100.0]),
            Q_cond=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
        ),
    )
    monkeypatch.setattr(
        hp, "_validate_vapour_hp_refrigerant_ls", lambda n, a: ["R134A"] * n
    )
    monkeypatch.setattr(
        hp, "_get_bounds_for_multi_single_hp_opt", lambda _args: [(0.0, 1.0)] * 10
    )
    monkeypatch.setattr(
        hp, "_get_x0_for_multi_single_hp_opt", lambda **_kwargs: np.array([[0.2] * 10])
    )
    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([[0.2] * 10]))
    monkeypatch.setattr(
        hp,
        "_compute_multi_simple_hp_system_performance",
        lambda x, args, debug=False: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "net_work": np.array([1.0, 2.0]),
            "Q_ext": 0.0,
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "cop_h": 2.5,
            "hot_streams": StreamCollection(),
            "cold_streams": StreamCollection(),
        },
    )
    out_sim = hp._optimise_multi_simple_heat_pump_placement(args)
    assert out_sim["success"] is True

    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([]))
    with pytest.raises(ValueError, match="failed"):
        hp._optimise_multi_simple_heat_pump_placement(args)


def test_multi_single_x0_bounds_parse_and_performance(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    bnds = [(0.0, 0.8)] * 10
    x0 = hp._get_x0_for_multi_single_hp_opt(
        T_cond=np.array([120.0, 100.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        args=args,
        bnds=bnds,
    )
    assert x0.shape == (1, 10)

    with pytest.raises(ValueError, match="Bounds size must match x0 size"):
        hp._get_x0_for_multi_single_hp_opt(
            T_cond=np.array([120.0, 100.0]),
            Q_cond=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
            args=args,
            bnds=[(0.0, 1.0)],
        )

    monkeypatch.setattr(
        hp, "PropsSI", lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0
    )
    b = hp._get_bounds_for_multi_single_hp_opt(args)
    assert len(b) == 10

    x_bad = np.array([0.9] * 10)
    out_bad = hp._compute_multi_simple_hp_system_performance(x_bad, args)
    assert np.isinf(out_bad["obj"])

    class _FakeMultiSimple:
        work = 30.0
        work_arr = np.array([10.0, 20.0])
        Q_cool = 60.0
        Q_heat_arr = np.array([120.0, 80.0])
        Q_cool_arr = np.array([40.0, 20.0])
        penalty = 1.0

        def solve(self, **_kwargs):
            return None

        def build_stream_collection(self, **_kwargs):
            return _sc(
                Stream(
                    name="HP",
                    t_supply=100.0,
                    t_target=90.0,
                    heat_flow=10.0,
                    is_process_stream=False,
                )
            )

    monkeypatch.setattr(hp, "ParallelVapourCompressionCycles", _FakeMultiSimple)
    seq = iter([_pt_with_hnet(5.0, -1.0), _pt_with_hnet(6.0, -2.0)])
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **_kwargs: next(seq))
    out = hp._compute_multi_simple_hp_system_performance(
        np.array([0.2] * 10), args, debug=False
    )
    assert "model" in out


def test_brayton_paths_and_helpers(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    _patch_output_model_validate(monkeypatch)

    class _OptRes:
        success = True
        x = np.array([0.1, 0.2, 0.3, 0.9])
        message = "ok"

    monkeypatch.setattr(hp, "minimize", lambda **_kwargs: _OptRes())
    monkeypatch.setattr(
        hp,
        "_compute_brayton_hp_system_performance",
        lambda x, args: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "net_work": 0.5,
            "Q_ext": 0.0,
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "cop_h": 2.0,
            "hot_streams": StreamCollection(),
            "cold_streams": StreamCollection(),
        },
    )
    out = hp._optimise_brayton_heat_pump_placement(args)
    assert out["success"] is True

    T_co, dT_c, dT_gc, q_h = hp._parse_brayton_hp_state_variables(
        np.array([0.1, 0.2, 0.3, 0.9]), args
    )
    assert len(T_co) == len(dT_c) == len(dT_gc) == len(q_h) == 1

    class _FakeBraytonCycle:
        def __init__(self):
            self.work_net = 15.0
            self.Q_cool = 40.0
            self.cycle_states = [{}, {}, {}, {"T": 50.0}]

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(hp, "SimpleBraytonHeatPumpCycle", _FakeBraytonCycle)
    hp_list = hp._create_brayton_hp_list(
        T_comp_out=np.array([120.0]),
        dT_gc=np.array([20.0]),
        Q_gc=np.array([100.0]),
        dT_comp=np.array([10.0]),
        args=args,
    )
    assert len(hp_list) == 1

    class _FakeHPObj:
        def __init__(self):
            self.work_net = 12.0
            self.Q_cool = 35.0
            self.cycle_states = [{}, {}, {}, {"T": 42.0}]

    monkeypatch.setattr(hp, "_create_brayton_hp_list", lambda **_kwargs: [_FakeHPObj()])
    monkeypatch.setattr(
        hp,
        "_build_simulated_hpr_streams",
        lambda hp_list, **_kwargs: _sc(
            Stream(
                name="S",
                t_supply=100.0,
                t_target=90.0,
                heat_flow=10.0,
                is_process_stream=False,
            )
        ),
    )
    seq = iter([_pt_with_hnet(8.0, -1.0), _pt_with_hnet(7.0, -2.0)])
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **_kwargs: next(seq))
    out_perf = hp._compute_brayton_hp_system_performance(
        np.array([0.1, 0.2, 0.3, 0.9]), args
    )
    assert "cop_h" in out_perf


def test_misc_heat_pump_helpers_and_stream_builders(monkeypatch):
    with pytest.raises(ValueError, match="Infeasible temperature interval"):
        hp._create_stream_collection_of_background_profile(
            T_vals=np.array([100.0, 100.0]),
            H_vals=np.array([0.0, 10.0]),
        )

    hot = hp._create_stream_collection_of_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, -30.0, 20.0]),
        is_source=True,
    )
    assert isinstance(hot, StreamCollection)
    cold = hp._create_stream_collection_of_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, 30.0, -20.0]),
        is_source=False,
    )
    assert isinstance(cold, StreamCollection)

    q_vals = hp._get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_hpr=np.array([100.0, 60.0]),
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([100.0, 50.0, 0.0]),
        is_cond=False,
    )
    assert q_vals.shape == (2,)

    t_avg = hp._compute_entropic_mean_temperature(
        np.array([300.0, 310.0]), np.array([10.0, 5.0]), input_T_units="K"
    )
    assert t_avg > 0.0

    args = _base_args(refrigerant_ls=["R134A", "R600"], do_refrigerant_sort=False)
    refs = hp._validate_vapour_hp_refrigerant_ls(1, args)
    assert refs == ["R134A"]

    streams = hp._build_latent_streams(
        T_ls=np.array([110.0, 109.2, 80.0]),
        dT_phase_change=1.0,
        Q_ls=np.array([20.0, 10.0, 30.0]),
        is_hot=True,
        prefix="HP",
    )
    assert len(streams) >= 2

    class _FakeCycle:
        def build_stream_collection(self, **_kwargs):
            return _sc(
                Stream(
                    name="C",
                    t_supply=90.0,
                    t_target=70.0,
                    heat_flow=10.0,
                    is_process_stream=False,
                )
            )

    agg = hp._build_simulated_hpr_streams(
        [_FakeCycle()], include_cond=True, include_evap=True, dtcont_hp=5.0
    )
    assert len(agg) >= 1

    amb0 = hp._get_ambient_air_stream(0.0, 0.0, _base_args())
    amb_pos = hp._get_ambient_air_stream(10.0, 0.0, _base_args())
    amb_neg = hp._get_ambient_air_stream(0.0, 10.0, _base_args())
    assert len(amb0) == 0
    assert len(amb_pos) == 1
    assert len(amb_neg) == 1

    assert hp._calc_obj(
        10.0, 5.0, 0.0, 100.0, heat_to_power_ratio=2.0, penalty=1.0
    ) == pytest.approx(0.21)
    assert hp._calc_Q_amb(50.0, 30.0, 5.0) == pytest.approx(25.0)


def _sc(*streams):
    coll = StreamCollection()
    for s in streams:
        coll.add(s)
    return coll


def _stream(
    name: str,
    t_supply: float,
    t_target: float,
    heat_flow: float,
    *,
    is_process_stream: bool = True,
):
    return Stream(
        name=name,
        t_supply=t_supply,
        t_target=t_target,
        heat_flow=heat_flow,
        dt_cont=0.0,
        htc=1.0,
        is_process_stream=is_process_stream,
    )


def test_temperature_shift_and_h_column_edge_branches():
    T_hot, T_cold = hp._apply_temperature_shift_for_hpr_stream_dtmin_cont(
        np.array([100.0, 80.0]), 10.0
    )
    np.testing.assert_allclose(T_hot, np.array([90.0, 70.0]))
    np.testing.assert_allclose(T_cold, np.array([110.0, 90.0]))

    t_i0, h_i0 = hp._get_reduced_bckgrd_cascade_till_Q_target(
        1000.0,
        np.array([100.0, 80.0]),
        np.array([5.0, 0.0]),
    )
    np.testing.assert_allclose(t_i0, np.array([100.0, 80.0]))
    np.testing.assert_allclose(h_i0, np.array([5.0, 0.0]))

    with pytest.raises(ValueError, match="Target for heat pumping cannot be zero."):
        hp._get_reduced_bckgrd_cascade_till_Q_target(
            0.1,
            np.array([100.0, 90.0, 80.0]),
            np.array([1.0, 0.5, 0.1]),
        )


def test_cascade_and_multi_single_bounds_and_x0_branches(monkeypatch):
    args_cascade = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        dt_phase_change=20.0,
        dt_range_max=10.0,
        T_cold=np.array([50.0, 45.0]),
        T_hot=np.array([40.0, 35.0]),
        Q_hpr_target_max=100.0,
        refrigerant_ls=["R134A", "R134A"],
    )
    init_res = SimpleNamespace(
        T_cond=np.array([46.0]),
        Q_cond=np.array([70.0]),
        T_evap=np.array([36.0]),
        Q_evap=np.array([65.0]),
    )
    x0 = hp._get_x0_for_cascade_hp_opt(
        init_res=init_res,
        args=args_cascade,
        bnds=[(0.0, 1.0)] * 4,
    )
    assert x0.shape == (1, 4)

    monkeypatch.setattr(
        hp, "PropsSI", lambda prop, *_args: 260.0 if prop == "Tmin" else 340.0
    )
    bnds_cascade = hp._get_bounds_for_cascade_hp_opt(args_cascade)
    assert bnds_cascade[1][0] == bnds_cascade[1][1]

    args_ms = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        dt_phase_change=20.0,
        dt_range_max=10.0,
        T_cold=np.array([50.0, 45.0]),
        T_hot=np.array([40.0, 35.0]),
        Q_hpr_target_max=100.0,
        refrigerant_ls=["R134A"],
    )
    x0_ms = hp._get_x0_for_multi_single_hp_opt(
        T_cond=np.array([49.0]),
        Q_cond=np.array([50.0]),
        T_evap=np.array([36.0]),
        args=args_ms,
        bnds=[(0.0, 0.0)] * 5,
    )
    np.testing.assert_allclose(x0_ms, np.zeros((1, 5)))

    bnds_ms = hp._get_bounds_for_multi_single_hp_opt(args_ms)
    assert bnds_ms[1][0] == bnds_ms[1][1]
    assert bnds_ms[4][0] == bnds_ms[4][1]


def test_multi_simple_and_brayton_performance_debug_and_full_paths(monkeypatch):
    args = SimpleNamespace(
        n_cond=1,
        T_cold=np.array([100.0, 60.0]),
        T_hot=np.array([90.0, 40.0]),
        H_cold=np.array([30.0, 0.0]),
        H_hot=np.array([0.0, -30.0]),
        dt_range_max=50.0,
        dtcont_hp=2.0,
        eta_comp=0.8,
        refrigerant_ls=["R134A"],
        dt_hp_ihx=1.0,
        bckgrd_hot_streams=StreamCollection(),
        bckgrd_cold_streams=StreamCollection(),
        Q_hpr_target_max=40.0,
        Q_amb_max=0.0,
        eta_penalty=0.001,
        rho_penalty=10.0,
        heat_to_power_ratio=1.0,
        cold_to_power_ratio=0.0,
        is_heat_pumping=True,
    )

    monkeypatch.setattr(
        hp,
        "_parse_multi_simple_hp_state_temperatures",
        lambda x, args: (
            np.array([80.0]),
            np.array([2.0]),
            np.array([30.0]),
            np.array([60.0]),
            np.array([1.0]),
        ),
    )

    class _FakeMS:
        work = 10.0
        work_arr = np.array([10.0])
        Q_cool = 20.0
        Q_heat_arr = np.array([30.0])
        Q_cool_arr = np.array([20.0])
        penalty = 0.0

        def solve(self, **kwargs):
            return None

        def build_stream_collection(self, **kwargs):
            if kwargs.get("include_cond"):
                return _sc(_stream("H", 90.0, 80.0, 30.0, is_process_stream=False))
            return _sc(_stream("C", 50.0, 60.0, 20.0, is_process_stream=False))

    monkeypatch.setattr(hp, "ParallelVapourCompressionCycles", _FakeMS)
    seq = iter(
        [
            ProblemTable({PT.T.value: [100.0, 50.0], PT.H_NET.value: [4.0, -1.0]}),
            ProblemTable({PT.T.value: [100.0, 50.0], PT.H_NET.value: [3.0, -1.0]}),
        ]
    )
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **kwargs: next(seq))
    called = {"plot": 0}
    monkeypatch.setattr(
        hp,
        "plot_multi_hp_profiles_from_results",
        lambda *args, **kwargs: called.__setitem__("plot", called["plot"] + 1),
    )
    out = hp._compute_multi_simple_hp_system_performance(
        np.array([0.2] * 5), args, debug=True
    )
    assert out["cop_h"] > 0.0
    assert called["plot"] == 1

    class _FakeBraytonHP:
        def __init__(self):
            self.work_net = 12.0
            self.Q_cool = 15.0
            self.cycle_states = [{}, {}, {}, {"T": 45.0}]

    monkeypatch.setattr(
        hp,
        "_parse_brayton_hp_state_variables",
        lambda x, args: ([120.0], [10.0], [15.0], [35.0]),
    )
    monkeypatch.setattr(
        hp, "_create_brayton_hp_list", lambda **kwargs: [_FakeBraytonHP()]
    )
    monkeypatch.setattr(
        hp,
        "_build_simulated_hpr_streams",
        lambda hp_list, **kwargs: (
            _sc(_stream("GC", 120.0, 100.0, 35.0, is_process_stream=False))
            if kwargs.get("include_cond")
            else _sc(_stream("GH", 40.0, 50.0, 15.0, is_process_stream=False))
        ),
    )
    seq2 = iter(
        [
            ProblemTable({PT.T.value: [120.0, 80.0], PT.H_NET.value: [5.0, -1.0]}),
            ProblemTable({PT.T.value: [120.0, 80.0], PT.H_NET.value: [4.0, -2.0]}),
        ]
    )
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **kwargs: next(seq2))

    out_b = hp._compute_brayton_hp_system_performance(
        np.array([0.1, 0.2, 0.3, 0.8]),
        SimpleNamespace(
            Q_hpr_target_max=40.0,
            H_hot=np.array([0.0, -20.0]),
            bckgrd_hot_streams=StreamCollection(),
            bckgrd_cold_streams=StreamCollection(),
            Q_amb_max=0.0,
            heat_to_power_ratio=1.0,
            cold_to_power_ratio=0.0,
            is_heat_pumping=True,
        ),
    )
    assert out_b["cop_h"] > 0.0


def test_get_heat_pump_cascade_helper(monkeypatch):
    hot = _sc(_stream("H", 120.0, 110.0, 5.0, is_process_stream=False))
    cold = _sc(_stream("C", 70.0, 80.0, 5.0, is_process_stream=False))

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted: ProblemTable(
            {
                PT.T.value: [120.0, 80.0],
                PT.H_HOT_UT.value: [0.0, 0.0],
                PT.H_COLD_UT.value: [0.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda *args, **kwargs: {
            PT.H_HOT_UT.value: np.array([5.0, 0.0]),
            PT.H_COLD_UT.value: np.array([3.0, 0.0]),
        },
    )
    out = hp._get_hpr_cascade(hot, cold)
    assert len(out) == 3
