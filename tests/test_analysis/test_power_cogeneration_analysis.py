"""Tests for power cogeneration helper routines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.analysis import power_cogeneration_analysis as pca


def _make_zone_config(
    *,
    model: str = "Fixed Isentropic Turbine",
    load: float = 0.8,
    mech_eff: float = 0.9,
) -> SimpleNamespace:
    return SimpleNamespace(
        P_TURBINE_BOX=20.0,
        T_TURBINE_BOX=300.0,
        MIN_EFF=0.5,
        COMBOBOX=model,
        LOAD=load,
        MECH_EFF=mech_eff,
        CONDESATE_FLASH_CORRECTION=False,
    )


def _make_zone(config: SimpleNamespace, utilities: StreamCollection | None = None):
    return SimpleNamespace(
        config=config,
        hot_utilities=utilities or StreamCollection(),
        work_target=0.0,
        turbine_efficiency_target=0.0,
    )


def _steam_utility(
    name: str, *, t_supply: float, t_target: float, q: float, dt_cont: float = 3.0
):
    return Stream(
        name=name,
        t_supply=t_supply,
        t_target=t_target,
        heat_flow=q,
        dt_cont=dt_cont,
        is_process_stream=False,
    )


def _patch_steam_properties(monkeypatch):
    monkeypatch.setattr(pca, "T_CRIT", 1_000.0)
    monkeypatch.setattr(pca, "h_pT", lambda p, t: 3_000.0 + p + t * 0.01)
    monkeypatch.setattr(pca, "hL_p", lambda p: 1_000.0 + p)
    monkeypatch.setattr(pca, "hV_p", lambda p: 2_500.0 + p)
    monkeypatch.setattr(pca, "psat_T", lambda t: max(1.0, t * 0.05))
    monkeypatch.setattr(pca, "s_ph", lambda p, h: 2.0 + p * 0.01 + h * 1e-5)
    monkeypatch.setattr(pca, "h_ps", lambda p, s: 2_000.0 + p * 0.1 + s * 5.0)
    monkeypatch.setattr(pca, "Tsat_p", lambda p: 100.0 + p * 0.5)


def test_prepare_turbine_parameters_clamps_inputs():
    cfg = _make_zone_config(load=2.0, mech_eff=-0.3)
    params = pca._prepare_turbine_parameters(cfg)

    assert params["P_in"] == 20.0
    assert params["T_in"] == 300.0
    assert params["load_frac"] == 1.0
    assert params["mech_eff"] == 0.0
    assert params["model"] == "Fixed Isentropic Turbine"


def test_preprocess_utilities_returns_none_when_no_viable_hot_utility(monkeypatch):
    _patch_steam_properties(monkeypatch)

    cfg = _make_zone_config()
    utilities = StreamCollection()
    utilities.add(_steam_utility("bad", t_supply=120.0, t_target=110.0, q=0.0))
    zone = _make_zone(cfg, utilities)

    out = pca._preprocess_utilities(zone, {"P_in": 20.0, "T_in": 300.0})
    assert out is None


def test_preprocess_utilities_builds_solver_inputs(monkeypatch):
    _patch_steam_properties(monkeypatch)

    cfg = _make_zone_config()
    utilities = StreamCollection()
    utilities.add(_steam_utility("HU1", t_supply=180.0, t_target=140.0, q=120.0))
    utilities.add(_steam_utility("HU2", t_supply=170.0, t_target=165.0, q=80.0))
    zone = _make_zone(cfg, utilities)

    out = pca._preprocess_utilities(zone, {"P_in": 20.0, "T_in": 300.0})

    assert out is not None
    assert out["s"] >= 2
    assert len(out["P_out"]) == len(out["Q_users"]) == len(out["m_k"])
    assert out["Q_boiler"] > 0.0
    assert out["m_in_est"] > 0.0


def test_segment_work_and_solver_helpers_cover_all_models(monkeypatch):
    _patch_steam_properties(monkeypatch)
    monkeypatch.setattr(pca, "Work_SunModel", lambda *args, **kwargs: 11.0)
    monkeypatch.setattr(pca, "Work_MedinaModel", lambda *args, **kwargs: 12.0)
    monkeypatch.setattr(pca, "Work_THM", lambda *args, **kwargs: 13.0)

    params = {
        "model": "Sun & Smith (2015)",
        "load_frac": 0.5,
        "mech_eff": 0.9,
        "min_eff": 0.4,
        "CONDESATE_FLASH_CORRECTION": False,
    }
    data = {
        "P_out": [20.0, 10.0],
        "Q_users": [0.0, 100.0],
        "w_k": [0.0, 0.0],
        "w_isen_k": [0.0, 0.0],
        "m_k": [1.0, 0.0],
        "eff_k": [0.0, 0.0],
        "dh_is_k": [0.0, 50.0],
        "h_out": [3_000.0, 2_600.0],
        "h_tar": [1_100.0, 1_000.0],
        "h_sat": [2_500.0, 2_300.0],
        "s": 2,
        "m_in_est": 5.0,
    }
    state = pca._TurbineState(params, data)

    assert state.max_mass_flow(10.0) == 20.0
    assert state._max_mass_flow(10.0) == 20.0

    assert pca._segment_work(state, index=1, mass_flow=2.0, mass_flow_max=3.0) == 11.0
    state.model = "Medina-Flores et al. (2010)"
    assert pca._segment_work(state, index=1, mass_flow=2.0, mass_flow_max=3.0) == 12.0
    state.model = "Varbanov et al. (2004)"
    assert pca._segment_work(state, index=1, mass_flow=2.0, mass_flow_max=3.0) == 13.0
    state.model = "Fixed Isentropic Turbine"
    assert pca._segment_work(
        state, index=1, mass_flow=2.0, mass_flow_max=3.0
    ) == pytest.approx(state.w_isen_k[1] * state.min_eff)
    state.model = "Unknown"
    assert pca._segment_work(state, index=1, mass_flow=2.0, mass_flow_max=3.0) == 0.0
    assert pca._segment_work(state, index=1, mass_flow=0.0, mass_flow_max=3.0) == 0.0

    assert pca._apply_efficiency_limits(2.0, -1.0, 0.6) == (0.0, 0.6)
    assert pca._apply_efficiency_limits(1.0, 10.0, 0.2) == (2.0, 0.2)
    assert pca._apply_efficiency_limits(6.0, 10.0, 0.2) == (6.0, 0.6)

    assert pca._segment_enthalpy(5.0, 2.0, 0.0, 0.9) == 5.0
    assert pca._segment_enthalpy(10.0, 2.0, 2.0, 0.5) == pytest.approx(8.0)

    state.flash_correction = True
    m_flash = pca._segment_mass_flow(state, index=1, mass_flow=2.0)
    assert np.isfinite(m_flash)
    state.flash_correction = False
    m_no_flash = pca._segment_mass_flow(state, index=1, mass_flow=2.0)
    assert np.isfinite(m_no_flash)
    assert pca._segment_mass_flow(state, index=1, mass_flow=0.0) == 0.0


def test_iterate_and_solve_turbine_work(monkeypatch):
    _patch_steam_properties(monkeypatch)
    params = {
        "model": "Fixed Isentropic Turbine",
        "load_frac": 0.8,
        "mech_eff": 0.9,
        "min_eff": 0.5,
        "CONDESATE_FLASH_CORRECTION": False,
    }
    data = {
        "P_out": [20.0, 10.0],
        "Q_users": [0.0, 120.0],
        "w_k": [0.0, 0.0],
        "w_isen_k": [0.0, 0.0],
        "m_k": [1.0, 1.0],
        "eff_k": [0.0, 0.0],
        "dh_is_k": [0.0, 30.0],
        "h_out": [3_200.0, 3_000.0],
        "h_tar": [1_000.0, 1_100.0],
        "h_sat": [2_700.0, 2_500.0],
        "s": 2,
        "m_in_est": 4.0,
    }

    w_total, w_max = pca._solve_turbine_work(params, data)
    assert np.isfinite(w_total)
    assert np.isfinite(w_max)


def test_get_power_cogeneration_above_pinch_handles_early_returns_and_success(
    monkeypatch,
):
    cfg = _make_zone_config()
    zone = _make_zone(cfg)

    monkeypatch.setattr(pca, "_prepare_turbine_parameters", lambda _cfg: None)
    assert pca.get_power_cogeneration_above_pinch(zone) is zone

    monkeypatch.setattr(pca, "_prepare_turbine_parameters", lambda _cfg: {"P_in": 10.0})
    monkeypatch.setattr(pca, "_preprocess_utilities", lambda _z, _p: None)
    assert pca.get_power_cogeneration_above_pinch(zone) is zone

    monkeypatch.setattr(pca, "_preprocess_utilities", lambda _z, _p: {"ok": True})
    monkeypatch.setattr(pca, "_solve_turbine_work", lambda _p, _u: (40.0, 80.0))
    out = pca.get_power_cogeneration_above_pinch(zone)
    assert out.work_target == 40.0
    assert out.turbine_efficiency_target == 0.5

    monkeypatch.setattr(pca, "_solve_turbine_work", lambda _p, _u: (0.0, 0.0))
    out = pca.get_power_cogeneration_above_pinch(zone)
    assert out.turbine_efficiency_target == 0.0


def test_work_models_and_coeff_setter(monkeypatch):
    _patch_steam_properties(monkeypatch)

    assert np.isfinite(pca.Work_MedinaModel(10.0, 4.0, 100.0))

    w_sun = pca.Work_SunModel(
        P_in=10.0,
        h_in=3_000.0,
        P_out=5.0,
        h_sat=10_000.0,
        m=2.0,
        m_max=3.0,
        dh_is=120.0,
        n_mech=0.9,
        t_type=1,
    )
    assert np.isfinite(w_sun)

    w_thm = pca.Work_THM(
        P_in=30.0,
        h_in=3_000.0,
        P_out=5.0,
        h_sat=5_000.0,
        m=50.0,
        dh_is=150.0,
        n_mech=0.9,
        t_size=1,
        t_type=1,
    )
    assert np.isfinite(w_thm)

    with pytest.raises(ValueError, match="Unsupported Sun model turbine type"):
        pca.Work_SunModel(10.0, 3_000.0, 5.0, 2_000.0, 2.0, 3.0, 120.0, 0.9, "bad")

    with pytest.raises(ValueError, match="Unsupported THM turbine type"):
        pca.Work_THM(10.0, 3_000.0, 5.0, 2_000.0, 2.0, 120.0, 0.9, 1, "bad")

    with pytest.raises(ValueError, match="Unsupported THM turbine size"):
        pca.Work_THM(10.0, 3_000.0, 5.0, 2_000.0, 2.0, 120.0, 0.9, "bad", 1)

    var_coef = [[[None for _ in range(4)] for _ in range(2)] for _ in range(2)]
    pca.Set_Coeff(VarCoef=var_coef)
    assert var_coef[0][0][2] == pytest.approx(1.097)
    assert var_coef[1][1][0] == pytest.approx(-0.463)


# ===== Merged from test_power_cogenertion_analysis.py =====
"""Regression tests for power cogenertion analysis analysis routines."""
