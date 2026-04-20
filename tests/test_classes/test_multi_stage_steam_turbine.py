"""Regression tests for the multi-stage steam turbine solver."""

from __future__ import annotations

import numpy as np
import pytest

import OpenPinch.classes.multi_stage_steam_turbine as turbine_mod
from OpenPinch.classes.multi_stage_steam_turbine import MultiStageSteamTurbine


def _patch_steam_properties(monkeypatch):
    monkeypatch.setattr(turbine_mod, "h_pT", lambda p, t: 3_000.0 + p + t * 0.01)
    monkeypatch.setattr(turbine_mod, "hL_p", lambda p: 1_000.0 + p)
    monkeypatch.setattr(turbine_mod, "hV_p", lambda p: 2_500.0 + p)
    monkeypatch.setattr(turbine_mod, "psat_T", lambda t: max(1.0, t * 0.05))
    monkeypatch.setattr(turbine_mod, "s_ph", lambda p, h: 2.0 + p * 0.01 + h * 1e-5)
    monkeypatch.setattr(turbine_mod, "h_ps", lambda p, s: 2_000.0 + p * 0.1 + s * 5.0)
    monkeypatch.setattr(turbine_mod, "Tsat_p", lambda p: 100.0 + p * 0.5)


def test_turbine_requires_solution_before_result_access():
    turbine = MultiStageSteamTurbine()
    with pytest.raises(RuntimeError):
        _ = turbine.result
    with pytest.raises(RuntimeError):
        _ = turbine.stages
    with pytest.raises(RuntimeError):
        _ = turbine.total_work


def test_above_pinch_solve_returns_valid_stage_details(monkeypatch):
    _patch_steam_properties(monkeypatch)

    turbine = MultiStageSteamTurbine()
    total_work, details = turbine.solve(
        np.array([180.0, 165.0]),
        np.array([120.0, 80.0]),
        mode="above_pinch",
        T_in=300.0,
        P_in=20.0,
        model="Fixed Isentropic Turbine",
        min_eff=0.5,
        load_frac=0.8,
        mech_eff=0.9,
    )

    assert turbine.solved
    assert total_work == pytest.approx(details["total_work"])
    assert details["mode"] == "above_pinch"
    assert details["overall_efficiency"] >= 0.5
    assert len(details["stages"]) == 2
    assert details["stages"][0]["stage_type"] == "extraction"
    assert details["stages"][0]["pressure_in"] >= details["stages"][0]["pressure_out"]
    assert turbine.result.total_work == pytest.approx(total_work)


def test_below_pinch_solve_uses_environment_sink(monkeypatch):
    _patch_steam_properties(monkeypatch)

    turbine = MultiStageSteamTurbine()
    total_work, details = turbine.solve(
        np.array([180.0, 140.0]),
        np.array([200.0, 80.0]),
        mode="below_pinch",
        T_sink=25.0,
        model="Fixed Isentropic Turbine",
        min_eff=0.4,
        load_frac=0.9,
        mech_eff=0.95,
    )

    assert total_work == pytest.approx(details["total_work"])
    assert details["mode"] == "below_pinch"
    assert details["sink_temperature"] == pytest.approx(25.0)
    assert len(details["stages"]) == 2
    assert all(stage["stage_type"] == "condensing" for stage in details["stages"])
    assert all(
        stage["pressure_out"] == pytest.approx(details["sink_pressure"])
        for stage in details["stages"]
    )


def test_turbine_rejects_bad_inputs(monkeypatch):
    _patch_steam_properties(monkeypatch)

    turbine = MultiStageSteamTurbine()
    with pytest.raises(ValueError, match="align"):
        turbine.solve(
            np.array([180.0, 160.0]),
            np.array([100.0]),
            mode="above_pinch",
            T_in=300.0,
            P_in=20.0,
        )

    with pytest.raises(ValueError, match="requires T_in and P_in"):
        turbine.solve(
            np.array([180.0]),
            np.array([100.0]),
            mode="above_pinch",
        )

    with pytest.raises(ValueError, match="requires T_sink"):
        turbine.solve(
            np.array([180.0]),
            np.array([100.0]),
            mode="below_pinch",
        )
