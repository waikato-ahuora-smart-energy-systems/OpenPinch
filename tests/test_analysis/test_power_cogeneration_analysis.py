"""Tests for power cogeneration analysis helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.classes.multi_stage_steam_turbine as turbine_mod
from OpenPinch.services.power_cogeneration_analysis import (
    power_cogeneration_analysis as pca,
)
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection


def _make_zone_config(
    *,
    model: str = "Fixed Isentropic Turbine",
    load: float = 0.8,
    mech_eff: float = 0.9,
):
    return SimpleNamespace(
        P_TURBINE_BOX=20.0,
        T_TURBINE_BOX=300.0,
        MIN_EFF=0.5,
        COMBOBOX=model,
        LOAD=load,
        MECH_EFF=mech_eff,
        CONDESATE_FLASH_CORRECTION=False,
        T_ENV=15.0,
    )


def _make_zone(config: SimpleNamespace, utilities: StreamCollection | None = None):
    return SimpleNamespace(
        config=config,
        hot_utilities=utilities or StreamCollection(),
        work_target=0.0,
        turbine_efficiency_target=0.0,
    )


def _steam_utility(
    name: str,
    *,
    t_supply: float,
    t_target: float,
    q: float,
    dt_cont: float = 3.0,
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
    monkeypatch.setattr(pca, "psat_T", lambda t: max(1.0, t * 0.05))

    monkeypatch.setattr(turbine_mod, "h_pT", lambda p, t: 3_000.0 + p + t * 0.01)
    monkeypatch.setattr(turbine_mod, "hL_p", lambda p: 1_000.0 + p)
    monkeypatch.setattr(turbine_mod, "hV_p", lambda p: 2_500.0 + p)
    monkeypatch.setattr(turbine_mod, "psat_T", lambda t: max(1.0, t * 0.05))
    monkeypatch.setattr(turbine_mod, "s_ph", lambda p, h: 2.0 + p * 0.01 + h * 1e-5)
    monkeypatch.setattr(turbine_mod, "h_ps", lambda p, s: 2_000.0 + p * 0.1 + s * 5.0)
    monkeypatch.setattr(turbine_mod, "Tsat_p", lambda p: 100.0 + p * 0.5)


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


def test_preprocess_utilities_builds_stage_arrays(monkeypatch):
    _patch_steam_properties(monkeypatch)

    cfg = _make_zone_config()
    utilities = StreamCollection()
    utilities.add(_steam_utility("HU1", t_supply=180.0, t_target=140.0, q=120.0))
    utilities.add(_steam_utility("HU2", t_supply=170.0, t_target=165.0, q=80.0))
    zone = _make_zone(cfg, utilities)

    out = pca._preprocess_utilities(zone, {"P_in": 20.0, "T_in": 300.0})

    assert out is not None
    assert np.allclose(out["stage_heat_flows"], [120.0, 80.0])
    assert np.allclose(out["stage_temperatures"], [146.0, 171.0])
    assert np.array_equal(out["source_indices"], np.array([0, 1]))


def test_get_power_cogeneration_above_pinch_updates_zone_targets(monkeypatch):
    _patch_steam_properties(monkeypatch)

    cfg = _make_zone_config()
    utilities = StreamCollection()
    utilities.add(_steam_utility("HU1", t_supply=180.0, t_target=140.0, q=120.0))
    utilities.add(_steam_utility("HU2", t_supply=170.0, t_target=165.0, q=80.0))
    zone = _make_zone(cfg, utilities)

    out = pca.get_power_cogeneration_above_pinch(zone)

    assert out is zone
    assert out.work_target > 0.0
    assert out.turbine_efficiency_target >= cfg.MIN_EFF


def test_get_power_cogeneration_above_pinch_returns_zone_when_no_viable_stage(
    monkeypatch,
):
    _patch_steam_properties(monkeypatch)

    cfg = _make_zone_config()
    zone = _make_zone(cfg, StreamCollection())

    out = pca.get_power_cogeneration_above_pinch(zone)
    assert out is zone
    assert out.work_target == 0.0
    assert out.turbine_efficiency_target == 0.0


def test_get_power_cogeneration_below_pinch_uses_environment_default(monkeypatch):
    _patch_steam_properties(monkeypatch)

    cfg = _make_zone_config()
    total_work, details = pca.get_power_cogeneration_below_pinch(
        np.array([180.0, 140.0]),
        np.array([200.0, 80.0]),
        zone_config=cfg,
    )

    assert total_work == pytest.approx(details["total_work"])
    assert details["sink_temperature"] == pytest.approx(cfg.T_ENV)
    assert len(details["stages"]) == 2


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
