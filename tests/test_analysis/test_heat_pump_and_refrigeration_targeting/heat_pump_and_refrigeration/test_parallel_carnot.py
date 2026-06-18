from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration.common.encoding import encode_duty_splits
from OpenPinch.services.heat_pump_integration.common.shared import (
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
)
from OpenPinch.services.heat_pump_integration.targeting_services.parallel_carnot import (
    _compute_parallel_carnot_hp_opt_obj,
    _parse_parallel_carnot_hp_state_variables,
)
from OpenPinch.services.heat_pump_integration.unit_models import ParallelCarnotCycles


def test_parallel_carnot_cycle_positive_lift_uses_absolute_temperatures():
    args = SimpleNamespace(
        T_hot=np.array([120.0, 20.0]),
        T_cold=np.array([80.0, 40.0]),
        eta_ii_hpr_carnot=0.5,
        eta_ii_he_carnot=0.0,
    )

    Q_heat_available = np.array([120.0])
    cycle = ParallelCarnotCycles()
    cycle.solve(
        T_cond=np.array([80.0]),
        T_evap=np.array([20.0]),
        Q_heat_base=float(Q_heat_available.sum()),
        x_heat_split=encode_duty_splits(Q_heat_available, Q_heat_available.sum()),
        Q_heat_available=Q_heat_available,
        Q_cool_available=np.array([200.0]),
        eta_ii_hpr_carnot=args.eta_ii_hpr_carnot,
        eta_ii_he_carnot=args.eta_ii_he_carnot,
        args=args,
    )

    expected_cop = (20.0 + 273.15) / (80.0 - 20.0) * args.eta_ii_hpr_carnot + 1.0
    expected_work = 120.0 / expected_cop
    expected_Q_evap = 120.0 - expected_work

    np.testing.assert_allclose(cycle.Q_cond, np.array([120.0]))
    np.testing.assert_allclose(cycle.Q_evap, np.array([expected_Q_evap]))
    np.testing.assert_allclose(cycle.w_hpr, np.array([expected_work]))
    np.testing.assert_allclose(cycle.w_he, np.array([0.0]))
    assert cycle.heat_recovery == 0.0


def test_parallel_carnot_cycle_uses_per_stage_pools():
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

    Q_heat_available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold, is_cond=True
    )
    Q_cool_available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap, args.T_hot, H_hot, is_cond=False
    )
    cycle = ParallelCarnotCycles()
    cycle.solve(
        T_cond=T_cond,
        T_evap=T_evap,
        Q_heat_base=float(Q_heat_available.sum()),
        x_heat_split=encode_duty_splits(Q_heat_available, Q_heat_available.sum()),
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
        eta_ii_hpr_carnot=args.eta_ii_hpr_carnot,
        eta_ii_he_carnot=args.eta_ii_he_carnot,
        args=args,
    )

    available_total = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        np.array([100.0]), args.T_hot, H_hot, is_cond=False
    )[0]

    np.testing.assert_allclose(cycle.Q_evap, np.array([available_total, 0.0]))
    np.testing.assert_allclose(cycle.Q_cond, np.array([available_total, 0.0]))
    np.testing.assert_allclose(cycle.w_hpr, np.array([0.0, 0.0]))
    np.testing.assert_allclose(cycle.w_he, np.array([0.0, 0.0]))
    assert cycle.heat_recovery == available_total
    assert cycle.Q_evap.sum() == available_total


def test_compute_parallel_carnot_objective_handles_mixed_lift_without_ambiguous_truth():
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
        eta_penalty=0.001,
        rho_penalty=10,
        allow_integrated_expander=False,
    )

    res = _compute_parallel_carnot_hp_opt_obj(
        np.array([0.0, 0.0, 0.0, 0.25, 0.25, 1.0, 0.5, 1.0]), args
    )

    assert np.isfinite(res["obj"])
    assert np.isfinite(res["utility_tot"])
    assert res["Q_cond"].shape == (2,)


def test_compute_parallel_carnot_utility_total_includes_residual_cold_utility():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        T_cold=np.array([100.0, 50.0]),
        H_cold=np.array([200.0, 0.0]),
        T_hot=np.array([90.0, 40.0]),
        H_hot=np.array([0.0, -20.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        eta_ii_hpr_carnot=0.5,
        eta_ii_he_carnot=0.0,
        Q_hpr_target=200.0,
        Q_heat_max=200.0,
        Q_cool_max=20.0,
        heat_to_power_ratio=1.0,
        cold_to_power_ratio=1.0,
        eta_penalty=0.001,
        rho_penalty=10.0,
    )

    res = _compute_parallel_carnot_hp_opt_obj(np.array([0.0, 0.5, 0.5, 1.0, 1.0]), args)

    assert res["Q_ext"] > 0.0
    assert np.isclose(res["utility_tot"], res["w_net"] + res["Q_ext"])


def test_parse_parallel_carnot_state_variables_uses_bounded_ambient_mapping():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        T_cold=np.array([100.0, 50.0]),
        H_cold=np.array([200.0, 0.0]),
        T_hot=np.array([90.0, 40.0]),
        H_hot=np.array([0.0, -160.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        Q_heat_max=200.0,
        Q_cool_max=160.0,
    )

    vars = _parse_parallel_carnot_hp_state_variables(
        np.array([0.5, 0.5, 0.5, 1.0, 1.0]),
        args,
    )

    np.testing.assert_allclose(vars["T_cond"], np.array([75.0]))
    np.testing.assert_allclose(vars["T_evap"], np.array([65.0]))
    assert vars["Q_amb_hot"] == 0.0
    assert np.isclose(vars["Q_amb_cold"], 200.0 * np.arctanh(0.5))
    assert vars["Q_heat_base"] == pytest.approx(200.0 + 200.0 * np.arctanh(0.5))
    np.testing.assert_allclose(vars["x_heat_split"], np.array([1.0]))
