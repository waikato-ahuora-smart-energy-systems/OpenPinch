from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_pump_and_refrigeration_placement import (
    cascade_vapour_compression as hp_cascade,
)
from OpenPinch.analysis.heat_pump_and_refrigeration_placement import shared as hp_shared

from ..helpers import _base_args, _pt_with_hnet, _sc, _stream


def test_cascade_hp_x0_and_bounds_shapes_are_consistent():
    pytest.importorskip("CoolProp")

    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        n_cond=2,
        n_evap=2,
        dt_range_max=130.0,
        dt_phase_change=1.0,
        Q_hpr_target=1000.0,
        refrigerant_ls=["R134A", "R134A", "R134A"],
        Q_cool_max=1000.0,
        Q_heat_max=900.0,
    )
    init = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        T_evap=np.array([50.0, 20.0]),
        Q_cond=np.array([600.0, 400.0]),
        Q_evap=np.array([500.0, 400.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    bnds = hp_cascade._get_bounds_for_cascade_hp_opt(args)
    x0 = hp_cascade._get_x0_for_cascade_hp_opt(init_res=init, args=args)

    assert x0.shape == (13,)
    assert x0[0] == pytest.approx(0.0)
    assert len(bnds) == x0.shape[0]
    assert np.all(
        (x0[0] >= np.array([b[0] for b in bnds]))
        & (x0[0] <= np.array([b[1] for b in bnds]))
    )


def test_cascade_x0_bounds_and_parse(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    x0 = hp_cascade._get_x0_for_cascade_hp_opt(init_res=init_res, args=args)
    assert x0.shape[0] == 13
    assert np.all(x0[0] >= 0.0)

    monkeypatch.setattr(
        hp_shared,
        "PropsSI",
        lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0,
    )
    bnds = hp_cascade._get_bounds_for_cascade_hp_opt(args)
    assert len(bnds) == 13

    vars = hp_cascade._parse_cascade_hp_state_variables(
        np.array([0.1, 0.2, 0.4, 0.3, 0.5, 0.01, 0.02, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1]),
        args,
    )
    np.testing.assert_allclose(vars["T_cond"], np.array([110.0, 78.0]))
    np.testing.assert_allclose(vars["T_evap"], np.array([105.0, 70.0]))
    np.testing.assert_allclose(vars["dT_subcool"], np.array([0.2, 1.04]))
    np.testing.assert_allclose(vars["Q_heat"], np.array([120.0, 140.0]))
    np.testing.assert_allclose(vars["Q_cool"][:-1], np.array([144.0]))
    assert np.isnan(vars["Q_cool"][-1])
    assert vars["Q_amb_hot"] == pytest.approx(0.0)
    assert vars["Q_amb_cold"] == pytest.approx(
        0.1 * max(args.Q_heat_max, args.Q_cool_max)
    )


def test_compute_cascade_hp_system_obj_unsolved_and_solved(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    x = np.array([0.2] * 9)
    monkeypatch.setattr(
        hp_cascade,
        "_parse_cascade_hp_state_variables",
        lambda _x, _args: {
            "T_cond": np.array([110.0, 90.0]),
            "dT_subcool": np.array([5.0, 5.0]),
            "Q_heat": np.array([120.0, 80.0]),
            "T_evap": np.array([70.0, 50.0]),
            "Q_cool": np.array([50.0, np.nan]),
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "dT_ihx_gas_side": np.array([0.0, 0.0, 0.0]),
        },
    )

    class _FakeCascadeUnsolved:
        solved = False
        work = 50.0

        def solve(self, **kwargs):
            return None

    monkeypatch.setattr(
        hp_cascade, "CascadeVapourCompressionCycle", _FakeCascadeUnsolved
    )
    out_unsolved = hp_cascade._compute_cascade_hp_system_obj(x, args)
    assert "obj" in out_unsolved

    class _FakeCascadeSolved:
        solved = True
        work = 40.0
        work_arr = np.array([10.0, 30.0])
        Q_cool = 70.0
        Q_heat_arr = np.array([120.0, 80.0])
        Q_cool_arr = np.array([50.0, 20.0])
        penalty = 1.0

        def solve(self, **kwargs):
            return None

        def build_stream_collection(self, **kwargs):
            return _sc(_stream("HP", 100.0, 90.0, 10.0, is_process_stream=False))

    monkeypatch.setattr(hp_cascade, "CascadeVapourCompressionCycle", _FakeCascadeSolved)
    seq = iter([_pt_with_hnet(5.0, -1.0), _pt_with_hnet(6.0, -2.0)])
    monkeypatch.setattr(
        hp_cascade, "get_process_heat_cascade", lambda **kwargs: next(seq)
    )
    calls = {"plot": 0}
    monkeypatch.setattr(
        hp_cascade,
        "plot_multi_hp_profiles_from_results",
        lambda *args, **kwargs: calls.__setitem__("plot", calls["plot"] + 1),
    )

    out = hp_cascade._compute_cascade_hp_system_obj(x, args, debug=True)
    assert "hpr_hot_streams" in out
    assert out["Q_ext"] == pytest.approx(3.0)
    assert out["utility_tot"] == pytest.approx(43.0)
    assert out["w_net"] == pytest.approx(40.0)
    assert out["cop_h"] == pytest.approx(5.0)
    assert calls["plot"] == 1


def test_cascade_x0_branch_for_single_stage():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        dt_phase_change=20.0,
        dt_range_max=10.0,
        T_cold=np.array([50.0, 45.0]),
        T_hot=np.array([40.0, 35.0]),
        Q_hpr_target=100.0,
        refrigerant_ls=["R134A", "R134A"],
        Q_cool_max=100.0,
        Q_heat_max=100.0,
    )
    init_res = SimpleNamespace(
        T_cond=np.array([46.0]),
        Q_cond=np.array([70.0]),
        T_evap=np.array([36.0]),
        Q_evap=np.array([65.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    x0 = hp_cascade._get_x0_for_cascade_hp_opt(init_res=init_res, args=args)
    assert x0.shape == (6,)
