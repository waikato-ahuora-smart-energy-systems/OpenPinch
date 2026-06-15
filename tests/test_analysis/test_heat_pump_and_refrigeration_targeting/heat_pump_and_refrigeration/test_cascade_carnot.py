from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.schemas.hpr import HPRBackendResult
from OpenPinch.services.heat_pump_integration.common import shared as hp_shared
from OpenPinch.services.heat_pump_integration.common.encoding import encode_duty_splits
from OpenPinch.services.heat_pump_integration.common.shared import (
    compute_entropic_mean_temperature,
)
from OpenPinch.services.heat_pump_integration.targeting_services import (
    cascade_carnot as hp_cascade_carnot,
)
from OpenPinch.services.heat_pump_integration.targeting_services.cascade_carnot import (
    _parse_cascade_carnot_cycle_state_variables,
)
from OpenPinch.services.heat_pump_integration.unit_models import (
    CascadeCarnotCycle,
)
from OpenPinch.services.heat_pump_integration.unit_models import (
    carnot_cycles as hp_carnot_cycles,
)

from ..helpers import (
    _base_args,
    _build_cascade_carnot_profiles,
)


def _solve_cascade_carnot_cycle(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    args: SimpleNamespace,
) -> CascadeCarnotCycle:
    Q_heat_base = float(Q_cond.sum())
    cycle = CascadeCarnotCycle()
    cycle.solve(
        T_cond=T_cond,
        T_evap=T_evap,
        Q_heat_base=Q_heat_base,
        x_heat_split=encode_duty_splits(Q_cond, Q_heat_base),
        Q_heat_available=Q_cond,
        Q_cool_available=Q_evap,
        eta_ii_hpr_carnot=args.eta_ii_hpr_carnot,
        eta_ii_he_carnot=args.eta_ii_he_carnot,
        args=args,
    )
    return cycle


def test_cascade_carnot_cycle_returns_entropic_mean_cop():
    T_cond = np.array([150.0, 120.0])
    Q_cond = np.array([60.0, 40.0])
    T_evap = np.array([60.0, 30.0])
    Q_evap = np.array([80.0, 40.0])
    args, H_hot, H_cold = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    expected = (
        compute_entropic_mean_temperature(T_evap, Q_evap)
        / (
            compute_entropic_mean_temperature(T_cond, Q_cond)
            - compute_entropic_mean_temperature(T_evap, Q_evap)
        )
        * args.eta_ii_hpr_carnot
        + 1.0
    )

    assert H_hot is not None
    assert H_cold is not None
    np.testing.assert_allclose(cycle.COP_h, expected)
    assert cycle.Q_cond.sum() == pytest.approx(cycle.w_hpr * expected)
    assert cycle.Q_evap.sum() == pytest.approx(cycle.Q_cond.sum() - cycle.w_hpr)


def test_cascade_carnot_heat_engine_split_uses_bounded_scalar_minimiser(
    monkeypatch,
):
    calls = {}

    def fake_minimize_scalar(*, fun, bounds, method):
        calls["bounds"] = bounds
        calls["method"] = method
        return SimpleNamespace(x=0.5)

    monkeypatch.setattr(hp_carnot_cycles, "minimize_scalar", fake_minimize_scalar)
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond=np.array([80.0]),
        Q_cond=np.array([60.0]),
        T_evap=np.array([120.0]),
        Q_evap=np.array([80.0]),
        eta_hp=0.5,
        eta_he=0.5,
    )

    _solve_cascade_carnot_cycle(
        T_cond=np.array([80.0]),
        Q_cond=np.array([60.0]),
        T_evap=np.array([120.0]),
        Q_evap=np.array([80.0]),
        args=args,
    )

    assert calls == {"bounds": (0, 1), "method": "bounded"}


def test_cascade_carnot_cycle_positive_lift_scales_evaporator_side():
    T_cond = np.array([80.0])
    Q_cond = np.array([200.0])
    T_evap = np.array([20.0])
    Q_evap = np.array([90.0])
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    expected_cop = (T_evap[0] + 273.15) / (
        T_cond[0] - T_evap[0]
    ) * args.eta_ii_hpr_carnot + 1
    expected_work_use = Q_evap[0] / (expected_cop - 1.0)
    expected_Q_cond = Q_evap[0] + expected_work_use

    assert cycle.w_he == pytest.approx(0.0)
    assert cycle.heat_recovery.sum() == pytest.approx(0.0)
    assert cycle.COP_h == pytest.approx(expected_cop)
    assert cycle.w_hpr == pytest.approx(expected_work_use)
    np.testing.assert_allclose(cycle.Q_cond, np.array([expected_Q_cond]))
    np.testing.assert_allclose(cycle.Q_evap, Q_evap)


def test_cascade_carnot_cycle_zero_lift_returns_no_useful_work():
    T_cond = np.array([50.0])
    Q_cond = np.array([100.0])
    T_evap = np.array([50.0])
    Q_evap = np.array([100.0])
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    assert cycle.w_hpr == pytest.approx(0.0, abs=0.01)
    assert cycle.w_he == pytest.approx(0.0, abs=0.01)
    assert cycle.heat_recovery.sum() == pytest.approx(100.0, abs=0.1)
    assert cycle.COP_h == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(cycle.Q_cond, Q_cond, atol=0.01)
    np.testing.assert_allclose(cycle.Q_evap, Q_evap, atol=0.01)


def test_cascade_carnot_cycle_negative_lift_generates_work():
    T_cond = np.array([30.0])
    Q_cond = np.array([100.0])
    T_evap = np.array([60.0])
    Q_evap = np.array([120.0])
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.5
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    expected_eta_he = args.eta_ii_he_carnot * (
        1
        - compute_entropic_mean_temperature(T_cond, Q_cond)
        / compute_entropic_mean_temperature(T_evap, Q_evap)
    )
    expected_Q_evap = min(Q_evap.sum(), Q_cond.sum() / (1.0 - expected_eta_he))
    expected_work_gen = expected_Q_evap * expected_eta_he
    expected_Q_cond = expected_Q_evap - expected_work_gen

    assert cycle.w_hpr == pytest.approx(0.0, abs=0.01)
    assert cycle.w_he == pytest.approx(expected_work_gen, abs=0.01)
    assert cycle.heat_recovery.sum() == pytest.approx(expected_Q_cond, abs=0.01)
    assert cycle.COP_h == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(cycle.Q_cond, np.array([expected_Q_cond]), atol=0.01)
    np.testing.assert_allclose(cycle.Q_evap, np.array([expected_Q_evap]), atol=0.01)


def test_cascade_carnot_cycle_negative_pool_counts_each_evaporator_once():
    T_cond = np.array([50.0, 30.0])
    Q_cond = np.array([10.0, 20.0])
    T_evap = np.array([60.0])
    Q_evap = np.array([100.0])
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.5
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    expected_eta_he = args.eta_ii_he_carnot * (
        1
        - compute_entropic_mean_temperature(T_cond, Q_cond)
        / compute_entropic_mean_temperature(T_evap, Q_evap)
    )
    expected_Q_evap = min(Q_evap.sum(), Q_cond.sum() / (1.0 - expected_eta_he))
    expected_work_gen = expected_Q_evap * expected_eta_he
    expected_Q_cond = expected_Q_evap - expected_work_gen

    assert cycle.w_hpr == pytest.approx(0.0, abs=0.01)
    assert cycle.w_he == pytest.approx(expected_work_gen, abs=0.01)
    assert cycle.heat_recovery.sum() == pytest.approx(expected_Q_cond, abs=0.01)
    assert cycle.COP_h == pytest.approx(1.0, abs=0.01)
    assert cycle.Q_cond.sum() == pytest.approx(Q_cond.sum(), abs=0.01)
    assert cycle.Q_evap.sum() == pytest.approx(expected_Q_evap, abs=0.01)


def test_cascade_carnot_cycle_negative_lift_without_engine_becomes_heat_exchange():
    T_cond = np.array([30.0])
    Q_cond = np.array([100.0])
    T_evap = np.array([60.0])
    Q_evap = np.array([120.0])
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.5, eta_he=0.0
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    assert cycle.w_hpr == pytest.approx(0.0, abs=0.01)
    assert cycle.w_he == pytest.approx(0.0, abs=0.01)
    assert cycle.heat_recovery.sum() == pytest.approx(100.0, abs=0.01)
    assert cycle.COP_h == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(cycle.Q_cond, Q_cond, atol=0.01)
    np.testing.assert_allclose(cycle.Q_evap, np.array([100.0]), atol=0.01)


def test_cascade_carnot_cycle_zero_hp_efficiency_returns_no_positive_lift_transfer():
    T_cond = np.array([80.0])
    Q_cond = np.array([120.0])
    T_evap = np.array([20.0])
    Q_evap = np.array([200.0])
    args, _, _ = _build_cascade_carnot_profiles(
        T_cond, Q_cond, T_evap, Q_evap, eta_hp=0.0, eta_he=0.0
    )

    cycle = _solve_cascade_carnot_cycle(T_cond, Q_cond, T_evap, Q_evap, args)

    assert cycle.w_hpr == pytest.approx(0.0, abs=0.01)
    assert cycle.w_he == pytest.approx(0.0, abs=0.01)
    assert cycle.heat_recovery.sum() == pytest.approx(0.0, abs=0.01)
    assert cycle.COP_h == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(cycle.Q_cond, np.array([0.0]), atol=0.01)
    np.testing.assert_allclose(cycle.Q_evap, np.array([0.0]), atol=0.01)


def test_parse_cascade_carnot_cycle_state_variables_returns_expected_profiles():
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
        z_amb_hot=np.zeros(3),
        z_amb_cold=np.zeros(3),
    )

    vars = _parse_cascade_carnot_cycle_state_variables(
        np.array([0.5, 0.5, 0.5, 0.25, 0.5, 1.0, 0.5, 1.0]), args
    )

    np.testing.assert_allclose(vars["T_cond"], np.array([80.0, 60.0]), atol=0.01)
    np.testing.assert_allclose(vars["T_evap"], np.array([105.0, 60.0]), atol=0.01)
    assert vars["Q_amb_hot"] == 0.0
    assert vars["Q_amb_cold"] == pytest.approx(300.0 * np.arctanh(0.5), abs=0.01)
    assert vars["Q_heat_base"] == pytest.approx(
        300.0 + 300.0 * np.arctanh(0.5),
        abs=0.01,
    )
    np.testing.assert_allclose(vars["x_heat_split"], np.array([0.5, 1.0]))


def test_parse_cascade_carnot_cycle_state_variables_respects_cond_evap_split_sizes():
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
        z_amb_hot=np.zeros(3),
        z_amb_cold=np.zeros(3),
    )

    vars = _parse_cascade_carnot_cycle_state_variables(
        np.array([-0.1, 0.2, 0.5, 0.8, 0.4, 1.0, 1.0]), args
    )

    assert vars["T_cond"].shape == (1,)
    assert vars["T_evap"].shape == (3,)
    assert np.all(np.diff(vars["T_evap"]) <= 0.0)
    assert vars["Q_amb_hot"] == pytest.approx(300.0 * np.arctanh(0.1))
    assert vars["Q_amb_cold"] == 0.0


def test_cascade_carnot_optimiser_success_and_failure(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    monkeypatch.setattr(
        hp_shared,
        "multiminima",
        lambda **_kwargs: (np.array([[0.2, 0.6, 0.0, 1.0, 1.0]]), np.array([0.1])),
    )
    monkeypatch.setattr(
        hp_cascade_carnot,
        "_compute_cascade_carnot_cycle_obj",
        lambda x, args, debug=False: HPRBackendResult(
            obj=0.1,
            utility_tot=1.0,
            w_net=0.5,
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            cop_h=3.0,
            T_cond=np.array([100.0]),
            Q_cond=np.array([50.0]),
            T_evap=np.array([60.0]),
            Q_evap=np.array([40.0]),
        ),
    )

    out = hp_cascade_carnot.optimise_cascade_carnot_heat_pump_placement(
        args
    )
    assert out.success is True
    assert isinstance(out.amb_streams, StreamCollection)

    monkeypatch.setattr(
        hp_cascade_carnot,
        "_compute_cascade_carnot_cycle_obj",
        lambda x, args, debug=False: HPRBackendResult.failure(),
    )
    with pytest.raises(ValueError, match="failed to return an optimal result"):
        hp_cascade_carnot.optimise_cascade_carnot_heat_pump_placement(args)


def test_cascade_carnot_objective_debug_branch(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1, idx=0)
    called = {"plot": 0}
    monkeypatch.setattr(
        hp_shared,
        "plot_multi_hp_profiles_from_results",
        lambda *args, **kwargs: called.__setitem__("plot", called["plot"] + 1),
    )

    out = hp_cascade_carnot._compute_cascade_carnot_cycle_obj(
        np.array([0.2, 0.7, 0.0, 1.0, 1.0]), args, debug=True
    )

    assert out["success"] is True
    assert called["plot"] == 1
