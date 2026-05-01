from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_pump_and_refrigeration_placement import (
    multi_temperature_carnot as hp_multi_temp_carnot,
)
from OpenPinch.analysis.heat_pump_and_refrigeration_placement import shared as hp_shared
from OpenPinch.analysis.heat_pump_and_refrigeration_placement.multi_temperature_carnot import (
    _get_multi_temperature_carnot_stage_duties_and_work,
    _parse_multi_temperature_carnot_cycle_state_variables,
)
from OpenPinch.analysis.heat_pump_and_refrigeration_placement.shared import (
    compute_entropic_mean_temperature,
)
from OpenPinch.classes.stream_collection import StreamCollection

from ..helpers import (
    _base_args,
    _build_multi_temperature_profiles,
    _patch_output_model_validate,
)


def test_get_multi_temperature_carnot_stage_duties_and_work_returns_entropic_mean_cop():
    T_cond = np.array([150.0, 120.0])
    Q_cond = np.array([60.0, 40.0])
    T_evap = np.array([60.0, 30.0])
    Q_evap = np.array([80.0, 40.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    expected = (
        compute_entropic_mean_temperature(T_evap, Q_evap)
        / (
            compute_entropic_mean_temperature(T_cond, Q_cond)
            - compute_entropic_mean_temperature(T_evap, Q_evap)
        )
        * args.eta_ii_hpr_carnot
        + 1.0
    )

    np.testing.assert_allclose(cycle_results["cop"], expected)
    assert cycle_results["Qc"].sum() == pytest.approx(
        cycle_results["w_hpr"] * expected
    )
    assert cycle_results["Qe"].sum() == pytest.approx(
        cycle_results["Qc"].sum() - cycle_results["w_hpr"]
    )


def test_get_multi_temperature_carnot_stage_duties_and_work_positive_lift_scales_evaporator_side():
    T_cond = np.array([80.0])
    Q_cond = np.array([200.0])
    T_evap = np.array([20.0])
    Q_evap = np.array([90.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    expected_cop = (T_evap[0] + 273.15) / (
        T_cond[0] - T_evap[0]
    ) * args.eta_ii_hpr_carnot + 1
    expected_work_use = Q_evap[0] / (expected_cop - 1.0)
    expected_Q_cond = Q_evap[0] + expected_work_use

    assert cycle_results["w_he"] == pytest.approx(0.0)
    assert cycle_results["heat_recovery"] == pytest.approx(0.0)
    assert cycle_results["cop"] == pytest.approx(expected_cop)
    assert cycle_results["w_hpr"] == pytest.approx(expected_work_use)
    np.testing.assert_allclose(cycle_results["Qc"], np.array([expected_Q_cond]))
    np.testing.assert_allclose(cycle_results["Qe"], Q_evap)


def test_get_multi_temperature_carnot_stage_duties_and_work_zero_lift_returns_no_useful_work():
    T_cond = np.array([50.0])
    Q_cond = np.array([100.0])
    T_evap = np.array([50.0])
    Q_evap = np.array([100.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    assert cycle_results["w_hpr"] == pytest.approx(0.0)
    assert cycle_results["w_he"] == pytest.approx(0.0)
    assert cycle_results["heat_recovery"] == pytest.approx(100.0)
    assert cycle_results["cop"] == pytest.approx(1.0)
    np.testing.assert_allclose(cycle_results["Qc"], Q_cond)
    np.testing.assert_allclose(cycle_results["Qe"], Q_evap)


def test_get_multi_temperature_carnot_stage_duties_and_work_negative_lift_generates_work():
    T_cond = np.array([30.0])
    Q_cond = np.array([100.0])
    T_evap = np.array([60.0])
    Q_evap = np.array([120.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.5
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    expected_eta_he = args.eta_ii_he_carnot * (
        1
        - compute_entropic_mean_temperature(T_cond, Q_cond)
        / compute_entropic_mean_temperature(T_evap, Q_evap)
    )
    expected_Q_evap = min(Q_evap.sum(), Q_cond.sum() / (1.0 - expected_eta_he))
    expected_work_gen = expected_Q_evap * expected_eta_he
    expected_Q_cond = expected_Q_evap - expected_work_gen

    assert cycle_results["w_hpr"] == pytest.approx(0.0)
    assert cycle_results["w_he"] == pytest.approx(expected_work_gen)
    assert cycle_results["heat_recovery"] == pytest.approx(0.0)
    assert cycle_results["cop"] == pytest.approx(1.0)
    np.testing.assert_allclose(cycle_results["Qc"], np.array([expected_Q_cond]))
    np.testing.assert_allclose(cycle_results["Qe"], np.array([expected_Q_evap]))


def test_get_multi_temperature_carnot_stage_duties_and_work_negative_pool_counts_each_evaporator_once():
    T_cond = np.array([50.0, 30.0])
    Q_cond = np.array([10.0, 20.0])
    T_evap = np.array([60.0])
    Q_evap = np.array([100.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.5
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    expected_eta_he = args.eta_ii_he_carnot * (
        1
        - compute_entropic_mean_temperature(T_cond, Q_cond)
        / compute_entropic_mean_temperature(T_evap, Q_evap)
    )
    expected_Q_evap = min(Q_evap.sum(), Q_cond.sum() / (1.0 - expected_eta_he))
    expected_work_gen = expected_Q_evap * expected_eta_he

    assert cycle_results["w_hpr"] == pytest.approx(0.0)
    assert cycle_results["w_he"] == pytest.approx(expected_work_gen)
    assert cycle_results["heat_recovery"] == pytest.approx(0.0)
    assert cycle_results["cop"] == pytest.approx(1.0)
    assert cycle_results["Qc"].sum() == pytest.approx(Q_cond.sum())
    assert cycle_results["Qe"].sum() == pytest.approx(expected_Q_evap)


def test_get_multi_temperature_carnot_stage_duties_and_work_negative_lift_without_engine_becomes_heat_exchange():
    T_cond = np.array([30.0])
    Q_cond = np.array([100.0])
    T_evap = np.array([60.0])
    Q_evap = np.array([120.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    assert cycle_results["w_hpr"] == pytest.approx(0.0)
    assert cycle_results["w_he"] == pytest.approx(0.0)
    assert cycle_results["heat_recovery"] == pytest.approx(100.0)
    assert cycle_results["cop"] == pytest.approx(1.0)
    np.testing.assert_allclose(cycle_results["Qc"], Q_cond)
    np.testing.assert_allclose(cycle_results["Qe"], np.array([100.0]))


def test_get_multi_temperature_carnot_stage_duties_and_work_zero_hp_efficiency_returns_no_positive_lift_transfer():
    T_cond = np.array([80.0])
    Q_cond = np.array([120.0])
    T_evap = np.array([20.0])
    Q_evap = np.array([200.0])
    args, H_hot, H_cold = _build_multi_temperature_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.0, eta_he=0.0
    )

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond.copy(),
        T_evap.copy(),
        H_hot.copy(),
        H_cold.copy(),
        args,
    )

    assert cycle_results["w_hpr"] == pytest.approx(0.0)
    assert cycle_results["w_he"] == pytest.approx(0.0)
    assert cycle_results["heat_recovery"] == pytest.approx(0.0)
    assert cycle_results["cop"] == pytest.approx(1.0)
    np.testing.assert_allclose(cycle_results["Qc"], np.array([0.0]))
    np.testing.assert_allclose(cycle_results["Qe"], np.array([0.0]))


def test_parse_multi_temperature_carnot_cycle_state_variables_returns_expected_profiles():
    args = SimpleNamespace(
        n_cond=2,
        n_evap=2,
        Q_hpr_target=300.0,
        Q_heat_max=300.0,
        Q_cool_max=280.0,
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([300.0, 120.0, 0.0]),
        T_hot=np.array([150.0, 90.0, 30.0]),
        H_hot=np.array([0.0, -100.0, -280.0]),
    )

    vars = _parse_multi_temperature_carnot_cycle_state_variables(
        np.array([0.5, 0.5, 0.5, 0.25, 0.5]), args
    )

    np.testing.assert_allclose(vars["T_cond"], np.array([80.0, 60.0]))
    np.testing.assert_allclose(vars["T_evap"], np.array([105.0, 60.0]))
    assert vars["Q_amb_hot"] == 0.0
    assert vars["Q_amb_cold"] == pytest.approx(150.0)


def test_parse_multi_temperature_carnot_cycle_state_variables_respects_cond_evap_split_sizes():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=3,
        Q_hpr_target=300.0,
        Q_heat_max=300.0,
        Q_cool_max=280.0,
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([300.0, 120.0, 0.0]),
        T_hot=np.array([150.0, 90.0, 30.0]),
        H_hot=np.array([0.0, -100.0, -280.0]),
    )

    vars = _parse_multi_temperature_carnot_cycle_state_variables(
        np.array([-0.1, 0.2, 0.5, 0.8, 0.4]), args
    )

    assert vars["T_cond"].shape == (1,)
    assert vars["T_evap"].shape == (3,)
    assert np.all(np.diff(vars["T_evap"]) <= 0.0)
    assert vars["Q_amb_hot"] == pytest.approx(30.0)
    assert vars["Q_amb_cold"] == 0.0


def test_multi_temp_carnot_optimiser_success_and_failure(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    _patch_output_model_validate(monkeypatch)
    monkeypatch.setattr(hp_shared, "multiminima", lambda **_kwargs: np.array([[0.2, 0.6]]))
    monkeypatch.setattr(
        hp_multi_temp_carnot,
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
        hp_multi_temp_carnot,
        "get_carnot_hpr_cycle_streams",
        lambda *_args, **_kwargs: {
            "hpr_hot_streams": StreamCollection(),
            "hpr_cold_streams": StreamCollection(),
        },
    )

    out = hp_multi_temp_carnot.optimise_multi_temperature_carnot_heat_pump_placement(
        args
    )
    assert out["success"] is True

    monkeypatch.setattr(
        hp_multi_temp_carnot,
        "_compute_multi_temperature_carnot_cycle_obj",
        lambda x, args, debug=False: {"success": False},
    )
    with pytest.raises(ValueError, match="failed to return an optimal result"):
        hp_multi_temp_carnot.optimise_multi_temperature_carnot_heat_pump_placement(
            args
        )


def test_multi_temp_carnot_objective_debug_branch(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    called = {"plot": 0}
    monkeypatch.setattr(
        hp_multi_temp_carnot,
        "plot_multi_hp_profiles_from_results",
        lambda *args, **kwargs: called.__setitem__("plot", called["plot"] + 1),
    )

    out = hp_multi_temp_carnot._compute_multi_temperature_carnot_cycle_obj(
        np.array([0.2, 0.7, 0.0]), args, debug=True
    )

    assert out["success"] is True
    assert called["plot"] == 1
