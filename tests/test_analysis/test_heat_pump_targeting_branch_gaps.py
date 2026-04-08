"""Additional branch-coverage tests for heat pump targeting gaps."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis import heat_pump_targeting as hp
from OpenPinch.classes import ProblemTable, Stream, StreamCollection
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import ProblemTableLabel as PT


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


def test_plot_multi_hp_profiles_and_prepare_inputs_wrapper(monkeypatch):
    for fn in (
        "figure",
        "plot",
        "title",
        "xlabel",
        "ylabel",
        "grid",
        "legend",
        "axvline",
        "tight_layout",
        "show",
    ):
        monkeypatch.setattr(hp.plt, fn, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hp, "clean_composite_curve_ends", lambda t, h: (np.asarray(t), np.asarray(h))
    )
    monkeypatch.setattr(
        hp,
        "_get_heat_pump_cascade",
        lambda **kwargs: {
            PT.T.value: np.array([100.0, 80.0]),
            PT.H_HOT_UT.value: np.array([10.0, 0.0]),
            PT.H_COLD_UT.value: np.array([8.0, 0.0]),
        },
    )
    hp.plot_multi_hp_profiles_from_results(
        T_hot=np.array([120.0, 80.0]),
        H_hot=np.array([10.0, 0.0]),
        T_cold=np.array([120.0, 80.0]),
        H_cold=np.array([8.0, 0.0]),
        hp_hot_streams=StreamCollection(),
        hp_cold_streams=StreamCollection(),
        title="t",
    )

    cfg = Configuration()
    monkeypatch.setattr(
        hp,
        "_apply_temperature_shift_for_heat_pump_stream_dtmin_cont",
        lambda T_vals, dtmin_hp, is_di: (
            np.array([95.0, 85.0, 75.0]),
            np.array([105.0, 95.0, 85.0]),
            5.0,
        ),
    )
    monkeypatch.setattr(hp, "_get_H_col_till_target_Q", lambda q, t, h: (t, h))
    monkeypatch.setattr(
        hp,
        "_balance_hot_and_cold_heat_loads_with_ambient_air",
        lambda **kwargs: (
            np.array([95.0, 85.0, 75.0]),
            np.array([0.0, -5.0, -10.0]),
            np.array([105.0, 95.0, 85.0]),
            np.array([10.0, 5.0, 0.0]),
            1.0,
        ),
    )
    monkeypatch.setattr(
        hp, "clean_composite_curve", lambda t, h: (np.asarray(t), np.asarray(h))
    )
    calls = {"n": 0}

    def _fake_bg_streams(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _sc(_stream("NH", 95.0, 75.0, 10.0)), StreamCollection()
        return StreamCollection(), _sc(_stream("NC", 85.0, 105.0, -8.0))

    monkeypatch.setattr(
        hp,
        "_create_net_hot_and_cold_stream_collections_for_background_profile",
        _fake_bg_streams,
    )
    out = hp._prepare_heat_pump_target_inputs(
        Q_hp_target=10.0,
        T_vals=np.array([100.0, 90.0, 80.0]),
        H_hot=np.array([0.0, -5.0, -10.0]),
        H_cold=np.array([10.0, 5.0, 0.0]),
        zone_config=cfg,
    )
    assert len(out.net_hot_streams) == 1
    assert len(out.net_cold_streams) == 1


def test_temperature_shift_and_h_column_edge_branches():
    T_hot, T_cold, dT = hp._apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
        np.array([100.0, 80.0]), 10.0, False
    )
    np.testing.assert_allclose(T_hot, np.array([85.0, 65.0]))
    np.testing.assert_allclose(T_cold, np.array([115.0, 95.0]))
    assert dT == 15.0

    t_i0, h_i0 = hp._get_H_col_till_target_Q(
        1000.0,
        np.array([100.0, 80.0]),
        np.array([5.0, 0.0]),
    )
    assert t_i0.shape == (1,)
    assert h_i0.shape == (1,)

    t_last, h_last = hp._get_H_col_till_target_Q(
        0.1,
        np.array([100.0, 90.0, 80.0]),
        np.array([1.0, 0.5, 0.1]),
    )
    np.testing.assert_allclose(t_last, np.array([80.0]))
    np.testing.assert_allclose(h_last, np.array([0.1]))


def test_balance_hot_and_cold_with_ambient_positive_delta_branch():
    T_hot, H_hot, T_cold, H_cold, q_amb_max = (
        hp._balance_hot_and_cold_heat_loads_with_ambient_air(
            T_hot=np.array([140.0, 110.0, 80.0]),
            H_hot=np.array([0.0, 50.0, 100.0]),
            T_cold=np.array([140.0, 110.0, 80.0]),
            H_cold=np.array([0.0, 100.0, 200.0]),
            dtcont=5.0,
            T_env=25.0,
            dT_env_cont=5.0,
            dt_phase_change=2.0,
            is_heat_pumping=True,
        )
    )
    assert q_amb_max > 0.0
    assert T_hot.shape == H_hot.shape
    assert T_cold.shape == H_cold.shape


def test_cascade_and_multi_single_bounds_and_x0_branches(monkeypatch):
    args_cascade = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        dt_phase_change=20.0,
        dt_range_max=10.0,
        T_cold=np.array([50.0, 45.0]),
        T_hot=np.array([40.0, 35.0]),
        Q_hp_target=100.0,
        refrigerant_ls=["R134A", "R134A"],
    )
    x0 = hp._get_x0_for_cascade_hp_opt(
        T_cond=np.array([46.0]),
        Q_heat=np.array([70.0]),
        T_evap=np.array([36.0]),
        Q_cool=np.array([65.0]),
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
        Q_hp_target=100.0,
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


def test_multi_simple_carnot_zero_duty_and_debug_branches(monkeypatch):
    monkeypatch.setattr(
        hp,
        "_parse_multi_simple_carnot_hp_state_variables",
        lambda x, args: (np.array([100.0]), np.array([0.0]), np.array([80.0])),
    )
    out_zero = hp._compute_multi_simple_carnot_hp_opt_obj(
        np.array([0.1, 0.2]),
        SimpleNamespace(Q_hp_target=100.0, price_ratio=1.0),
    )
    assert "obj" in out_zero
    assert len(out_zero) == 1

    monkeypatch.setattr(
        hp,
        "_parse_multi_simple_carnot_hp_state_variables",
        lambda x, args: (np.array([105.0]), np.array([50.0]), np.array([80.0])),
    )
    monkeypatch.setattr(
        hp, "_get_Q_vals_from_T_vals", lambda *args, **kwargs: np.array([60.0])
    )
    monkeypatch.setattr(hp, "g_ineq_penalty", lambda g, rho, form: 0.0)
    monkeypatch.setattr(
        hp,
        "_get_carnot_hp_streams",
        lambda *args, **kwargs: {
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
        },
    )
    called = {"plot": 0}
    monkeypatch.setattr(
        hp,
        "plot_multi_hp_profiles_from_results",
        lambda *args, **kwargs: called.__setitem__("plot", called["plot"] + 1),
    )
    out_debug = hp._compute_multi_simple_carnot_hp_opt_obj(
        np.array([0.1, 0.2]),
        SimpleNamespace(
            Q_hp_target=100.0,
            price_ratio=1.0,
            T_hot=np.array([120.0, 80.0]),
            H_hot=np.array([0.0, -40.0]),
            T_cold=np.array([100.0, 60.0]),
            H_cold=np.array([40.0, 0.0]),
            eta_hp_carnot=0.5,
            eta_he_carnot=0.4,
            allow_integrated_expander=False,
            H_hot_end=40.0,
            H_hot_last=40.0,
            Q_amb_max=0.0,
            rho_penalty=10.0,
            T_env=25.0,
            dt_env_cont=5.0,
            dt_phase_change=1.0,
        ),
        debug=True,
    )
    assert out_debug["opt_success"] is True
    assert called["plot"] == 1


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
        net_hot_streams=StreamCollection(),
        net_cold_streams=StreamCollection(),
        Q_hp_target=40.0,
        Q_amb_max=0.0,
        eta_penalty=0.001,
        rho_penalty=10.0,
        price_ratio=1.0,
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
    monkeypatch.setattr(hp, "_constrain_min_temperature_lift", lambda x, args: 1.0)

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

    monkeypatch.setattr(hp, "MultiSimpleHeatPumpCycle", _FakeMS)
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
    assert out["cop"] > 0.0
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
        "_build_simulated_hps_streams",
        lambda hp_list: (
            _sc(_stream("GC", 120.0, 100.0, 35.0, is_process_stream=False)),
            _sc(_stream("GH", 40.0, 50.0, 15.0, is_process_stream=False)),
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
            Q_hp_target=40.0,
            H_hot=np.array([0.0, -20.0]),
            net_hot_streams=StreamCollection(),
            net_cold_streams=StreamCollection(),
            Q_amb_max=0.0,
        ),
    )
    assert out_b["cop"] > 0.0


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
    out = hp._get_heat_pump_cascade(hot, cold)
    assert set(out.keys()) == {PT.T.value, PT.H_HOT_UT.value, PT.H_COLD_UT.value}
