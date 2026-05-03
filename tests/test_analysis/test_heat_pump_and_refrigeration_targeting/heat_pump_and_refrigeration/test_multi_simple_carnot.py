from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis.heat_pump_and_refrigeration_placement import (
    multi_simple_carnot as hp_multi_simple_carnot,
)
from OpenPinch.analysis.heat_pump_and_refrigeration_placement.multi_simple_carnot import (
    _compute_multi_simple_carnot_hp_opt_obj,
    _get_multi_simple_carnot_stage_duties_and_work,
)
from OpenPinch.analysis.heat_pump_and_refrigeration_placement.shared import (
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
)


def test_get_multi_simple_carnot_stage_duties_and_work_positive_lift_uses_absolute_temperatures():
    args = SimpleNamespace(
        T_hot=np.array([120.0, 20.0]),
        T_cold=np.array([80.0, 40.0]),
        eta_ii_hpr_carnot=0.5,
        eta_ii_he_carnot=0.0,
    )

    cycle_results = _get_multi_simple_carnot_stage_duties_and_work(
        T_cond=np.array([80.0]),
        T_evap=np.array([20.0]),
        H_hot_with_amb=np.array([0.0, -200.0]),
        H_cold_with_amb=np.array([120.0, 0.0]),
        args=args,
    )

    expected_cop = (20.0 + 273.15) / (80.0 - 20.0) * args.eta_ii_hpr_carnot + 1.0
    expected_work = 120.0 / expected_cop
    expected_Q_evap = 120.0 - expected_work

    np.testing.assert_allclose(cycle_results["Qc"], np.array([120.0]))
    np.testing.assert_allclose(cycle_results["Qe"], np.array([expected_Q_evap]))
    np.testing.assert_allclose(cycle_results["w_hpr"], np.array([expected_work]))
    np.testing.assert_allclose(cycle_results["w_he"], np.array([0.0]))
    assert cycle_results["heat_recovery"] == 0.0


def test_get_multi_simple_carnot_stage_duties_and_work_shares_hot_profile_across_modes():
    args = SimpleNamespace(
        T_hot=np.array([120.0, 80.0, 40.0]),
        T_cold=np.array([100.0, 70.0, 40.0]),
        eta_ii_hpr_carnot=0.5,
        eta_ii_he_carnot=0.5,
    )
    T_cond = np.array([100.0, 70.0])
    T_evap = np.array([100.0, 110.0])
    H_hot = np.array([0.0, -120.0, -200.0])
    H_cold = np.array([300.0, 150.0, 0.0])

    cycle_results = _get_multi_simple_carnot_stage_duties_and_work(
        T_cond=T_cond,
        T_evap=T_evap,
        H_hot_with_amb=H_hot,
        H_cold_with_amb=H_cold,
        args=args,
    )

    available_he = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        np.array([110.0]), args.T_hot, H_hot, is_cond=False
    )[0]
    available_total = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        np.array([100.0]), args.T_hot, H_hot, is_cond=False
    )[0]
    expected_eta_he = args.eta_ii_he_carnot * (1.0 - (70.0 + 273.15) / (110.0 + 273.15))
    expected_Qc_he = available_he * (1.0 - expected_eta_he)
    expected_w_he = available_he - expected_Qc_he
    expected_Q_hx = available_total - available_he

    np.testing.assert_allclose(
        cycle_results["Qe"], np.array([expected_Q_hx, available_he])
    )
    np.testing.assert_allclose(
        cycle_results["Qc"], np.array([expected_Q_hx, expected_Qc_he])
    )
    np.testing.assert_allclose(cycle_results["w_hpr"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(cycle_results["w_he"], np.array([0.0, expected_w_he]))
    assert cycle_results["heat_recovery"] == expected_Q_hx
    assert cycle_results["Qe"].sum() == available_total


def test_compute_multi_simple_carnot_objective_handles_mixed_lift_without_ambiguous_truth(
    monkeypatch,
):
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
        eta_ii_hpr_carnot=0.5,
        eta_ii_he_carnot=0.5,
        Q_hpr_target=300.0,
        Q_heat_max=300.0,
        Q_cool_max=260.0,
        heat_to_power_ratio=1.0,
        cold_to_power_ratio=0.0,
        rho_penalty=10,
        allow_integrated_expander=False,
    )
    monkeypatch.setattr(
        hp_multi_simple_carnot, "g_ineq_penalty", lambda *args, **kwargs: 0.0
    )

    res = _compute_multi_simple_carnot_hp_opt_obj(
        np.array([0.0, 0.0, 0.25, 0.25, 0.0]), args
    )

    assert np.isfinite(res["obj"])
    assert np.isfinite(res["utility_tot"])
    assert res["Q_cond"].shape == (2,)
