from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration.common import shared as hp_shared
from OpenPinch.services.heat_pump_integration.common.encoding import decode_duty_splits
from OpenPinch.services.heat_pump_integration.targeting_services import (
    parallel_vapour_compression as hp_parallel_vapour,
)

from ..helpers import (
    _base_args,
    _patch_output_model_validate,
    _pt_with_hnet,
    _sc,
    _stream,
)


def test_parallel_hp_x0_and_bounds_shapes_are_consistent():
    pytest.importorskip("CoolProp")

    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        n_cond=2,
        n_evap=2,
        dt_range_max=130.0,
        dt_phase_change=1.0,
        Q_hpr_target=1000.0,
        refrigerant_ls=["R134A", "R134A"],
        Q_cool_max=1000.0,
        Q_heat_max=1000.0,
    )
    init_results = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        T_evap=np.array([50.0, 20.0]),
        Q_cond=np.array([600.0, 400.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    x0, bnds = hp_parallel_vapour._get_parallel_hp_opt_setup(init_results, args)

    assert x0.shape == (12,)
    assert len(bnds) == len(x0)
    assert np.all(
        (x0 >= np.array([b[0] for b in bnds])) & (x0 <= np.array([b[1] for b in bnds]))
    )


def test_parallel_x0_round_trips_with_ambient_cooling_seed():
    args = _base_args(n_cond=2, n_evap=2)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=20.0,
    )

    x0, _ = hp_parallel_vapour._get_parallel_hp_opt_setup(
        init_res=init_res,
        args=args,
    )
    vars = hp_parallel_vapour._parse_parallel_hp_state_temperatures(x0, args)

    assert abs(x0[0]) < 1.0
    np.testing.assert_allclose(vars["T_evap"], init_res.T_evap)
    assert vars["Q_amb_hot"] == pytest.approx(init_res.Q_amb_hot)
    assert vars["Q_amb_cold"] == pytest.approx(init_res.Q_amb_cold)
    assert vars["Q_heat_base"] == pytest.approx(init_res.Q_cond.sum())
    np.testing.assert_allclose(
        decode_duty_splits(vars["x_heat_split"], vars["Q_heat_base"]),
        init_res.Q_cond,
    )
    assert vars["Q_heat_available"].shape == init_res.Q_cond.shape


def test_parallel_refrigeration_maps_primary_duty_to_cooling():
    args = _base_args(n_cond=2, n_evap=2, is_heat_pumping=False)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=20.0,
        Q_amb_cold=0.0,
    )

    x0, _ = hp_parallel_vapour._get_parallel_hp_opt_setup(
        init_res=init_res,
        args=args,
    )
    vars = hp_parallel_vapour._parse_parallel_hp_state_temperatures(x0, args)

    assert vars["Q_heat_base"] is None
    assert vars["Q_heat_available"] is None
    assert vars["Q_cool_base"] == pytest.approx(init_res.Q_evap.sum())
    np.testing.assert_allclose(
        decode_duty_splits(vars["x_cool_split"], vars["Q_cool_base"]),
        init_res.Q_evap,
    )


def test_parallel_x0_bounds_parse_and_performance(monkeypatch):
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    args = _base_args(n_cond=2, n_evap=2)
    x0, bnds = hp_parallel_vapour._get_parallel_hp_opt_setup(
        init_res=init_res,
        args=args,
    )
    assert x0.shape[0] == 12

    monkeypatch.setattr(
        hp_shared,
        "PropsSI",
        lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0,
    )
    assert len(bnds) == 12

    vars = hp_parallel_vapour._parse_parallel_hp_state_temperatures(
        np.array([0.1] * 12), args
    )
    assert vars["T_cond"].shape == (2,)
    assert vars["dT_subcool"].shape == (2,)
    assert vars["x_heat_split"].shape == (2,)
    assert vars["Q_heat_available"].shape == (2,)
    np.testing.assert_allclose(vars["T_evap"], np.array([59.0, 50.0]))
    assert vars["dT_ihx_gas_side"].shape == (2,)
    assert vars["Q_amb_hot"] == pytest.approx(0.0)
    assert vars["Q_amb_cold"] == pytest.approx(
        max(args.Q_heat_max, args.Q_cool_max) * np.arctanh(0.1)
    )

    monkeypatch.setattr(
        hp_parallel_vapour,
        "_parse_parallel_hp_state_temperatures",
        lambda x, _args: (
            {
                "T_cond": np.array([100.0, 90.0]),
                "dT_subcool": np.array([10.0, 10.0]),
                "T_evap": np.array([90.0, 80.0]),
                "dT_superheat": np.array([5.0, 5.0]),
                "Q_amb_hot": 0.0,
                "Q_amb_cold": 0.0,
                "Q_heat_base": 200.0,
                "x_heat_split": np.array([0.6, 1.0]),
                "Q_heat_available": np.array([120.0, 80.0]),
            }
            if x[0] > 0.5
            else {
                "T_cond": np.array([120.0, 100.0]),
                "dT_subcool": np.array([5.0, 5.0]),
                "T_evap": np.array([70.0, 50.0]),
                "dT_superheat": np.array([2.0, 2.0]),
                "Q_amb_hot": 0.0,
                "Q_amb_cold": 0.0,
                "Q_heat_base": 200.0,
                "x_heat_split": np.array([0.6, 1.0]),
                "Q_heat_available": np.array([120.0, 80.0]),
            }
        ),
    )

    out_bad = hp_parallel_vapour._compute_parallel_hp_system_obj(
        np.array([0.9] * 10), args
    )
    assert np.isinf(out_bad["obj"])


def test_parallel_refrigeration_objective_solves_refrigeration_mode(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1, is_heat_pumping=False)
    captured = {}
    monkeypatch.setattr(
        hp_parallel_vapour,
        "_parse_parallel_hp_state_temperatures",
        lambda _x, _args: {
            "T_cond": np.array([100.0]),
            "dT_subcool": np.array([5.0]),
            "T_evap": np.array([60.0]),
            "dT_superheat": np.array([0.0]),
            "dT_ihx_gas_side": np.array([0.0]),
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "Q_cool_base": 80.0,
            "x_cool_split": np.array([1.0]),
            "Q_cool_available": np.array([80.0]),
        },
    )

    class _FakeParallelSolved:
        solved = True
        work = 20.0
        work_arr = np.array([20.0])
        Q_heat_arr = np.array([100.0])
        Q_cool_arr = np.array([80.0])
        penalty = 0.0

        def solve(self, **kwargs):
            captured.update(kwargs)

        def build_stream_collection(self, **kwargs):
            return _sc(_stream("HP", 90.0, 80.0, 10.0, is_process_stream=False))

    monkeypatch.setattr(
        hp_parallel_vapour,
        "ParallelVapourCompressionCycles",
        _FakeParallelSolved,
    )
    seq = iter([_pt_with_hnet(1.0, 1.0), _pt_with_hnet(1.0, 1.0)])
    monkeypatch.setattr(
        hp_shared, "get_process_heat_cascade", lambda **kwargs: next(seq)
    )

    out = hp_parallel_vapour._compute_parallel_hp_system_obj(np.array([0.2]), args)

    assert captured["is_heat_pump"] is False
    assert captured["Q_heat_base"] is None
    assert captured["Q_heat_available"] is None
    assert captured["Q_cool_base"] == pytest.approx(80.0)
    np.testing.assert_allclose(captured["x_cool_split"], np.array([1.0]))
    np.testing.assert_allclose(captured["Q_cool_available"], np.array([80.0]))
    assert out["cop_h"] == pytest.approx(4.0)


def test_parallel_optimiser_allows_missing_initial_seed(monkeypatch):
    captured = {}
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_cycle=False)
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp_parallel_vapour,
        "validate_vapour_hp_refrigerant_ls",
        lambda num_stages, args: ["R134A"] * num_stages,
    )

    def fake_solve_hpr_placement(*, f_obj, x0_ls, bnds, args):
        captured["x0_ls"] = x0_ls
        return {"success": True}

    monkeypatch.setattr(
        hp_parallel_vapour,
        "solve_hpr_placement",
        fake_solve_hpr_placement,
    )

    out = hp_parallel_vapour.optimise_parallel_heat_pump_placement(args)

    assert captured["x0_ls"] is None
    assert out["success"] is True
