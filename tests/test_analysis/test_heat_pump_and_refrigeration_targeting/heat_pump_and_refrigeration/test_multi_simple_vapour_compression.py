from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration.common import shared as hp_shared
from OpenPinch.services.heat_pump_integration.targeting_services import (
    multi_simple_vapour_compression as hp_multi_simple_vapour,
)

from ..helpers import _base_args, _patch_output_model_validate


def test_multi_single_hp_x0_and_bounds_shapes_are_consistent():
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

    x0, bnds = hp_multi_simple_vapour._get_multi_single_hp_opt_setup(init_results, args)

    assert x0.shape == (11,)
    assert len(bnds) == len(x0)
    assert np.all(
        (x0 >= np.array([b[0] for b in bnds])) & (x0 <= np.array([b[1] for b in bnds]))
    )


def test_multi_single_x0_round_trips_with_ambient_cooling_seed():
    args = _base_args(n_cond=2, n_evap=2)
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=20.0,
    )

    x0, _ = hp_multi_simple_vapour._get_multi_single_hp_opt_setup(
        init_res=init_res,
        args=args,
    )
    vars = hp_multi_simple_vapour._parse_multi_simple_hp_state_temperatures(x0, args)

    assert abs(x0[0]) < 1.0
    np.testing.assert_allclose(vars["T_evap"], init_res.T_evap)
    assert vars["Q_amb_hot"] == pytest.approx(init_res.Q_amb_hot)
    assert vars["Q_amb_cold"] == pytest.approx(init_res.Q_amb_cold)
    np.testing.assert_allclose(vars["Q_heat"], init_res.Q_cond)


def test_multi_single_x0_bounds_parse_and_performance(monkeypatch):
    init_res = SimpleNamespace(
        T_cond=np.array([120.0, 90.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_evap=np.array([90.0, 60.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    args = _base_args(n_cond=2, n_evap=2)
    x0, bnds = hp_multi_simple_vapour._get_multi_single_hp_opt_setup(
        init_res=init_res,
        args=args,
    )
    assert x0.shape[0] == 11

    monkeypatch.setattr(
        hp_shared,
        "PropsSI",
        lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0,
    )
    assert len(bnds) == 11

    vars = hp_multi_simple_vapour._parse_multi_simple_hp_state_temperatures(
        np.array([0.1] * 11), args
    )
    assert vars["T_cond"].shape == (2,)
    assert vars["dT_subcool"].shape == (2,)
    assert vars["Q_heat"].shape == (2,)
    np.testing.assert_allclose(vars["T_evap"], np.array([59.0, 50.0]))
    assert vars["dT_ihx_gas_side"].shape == (2,)
    assert vars["Q_amb_hot"] == pytest.approx(0.0)
    assert vars["Q_amb_cold"] == pytest.approx(
        max(args.Q_heat_max, args.Q_cool_max) * np.arctanh(0.1)
    )

    monkeypatch.setattr(
        hp_multi_simple_vapour,
        "_parse_multi_simple_hp_state_temperatures",
        lambda x, _args: (
            {
                "T_cond": np.array([100.0, 90.0]),
                "dT_subcool": np.array([10.0, 10.0]),
                "Q_heat": np.array([120.0, 80.0]),
                "T_evap": np.array([90.0, 80.0]),
                "dT_superheat": np.array([5.0, 5.0]),
                "Q_amb_hot": 0.0,
                "Q_amb_cold": 0.0,
            }
            if x[0] > 0.5
            else {
                "T_cond": np.array([120.0, 100.0]),
                "dT_subcool": np.array([5.0, 5.0]),
                "Q_heat": np.array([120.0, 80.0]),
                "T_evap": np.array([70.0, 50.0]),
                "dT_superheat": np.array([2.0, 2.0]),
                "Q_amb_hot": 0.0,
                "Q_amb_cold": 0.0,
            }
        ),
    )

    out_bad = hp_multi_simple_vapour._compute_multi_simple_hp_system_obj(
        np.array([0.9] * 10), args
    )
    assert np.isinf(out_bad["obj"])


def test_multi_single_optimiser_allows_missing_initial_seed(monkeypatch):
    captured = {}
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_cycle=False)
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp_multi_simple_vapour,
        "validate_vapour_hp_refrigerant_ls",
        lambda num_stages, args: ["R134A"] * num_stages,
    )

    def fake_solve_hpr_placement(*, f_obj, x0_ls, bnds, args):
        captured["x0_ls"] = x0_ls
        return {"success": True}

    monkeypatch.setattr(
        hp_multi_simple_vapour,
        "solve_hpr_placement",
        fake_solve_hpr_placement,
    )

    out = hp_multi_simple_vapour.optimise_multi_simple_heat_pump_placement(args)

    assert captured["x0_ls"] is None
    assert out["success"] is True
