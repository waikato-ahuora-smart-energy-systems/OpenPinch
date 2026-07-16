"""Regression tests for the multi-stage steam turbine solver."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.services.power_cogeneration.unit_models.multi_stage_steam_turbine as turbine_mod
from OpenPinch.services.power_cogeneration.unit_models.multi_stage_steam_turbine import (
    MultiStageSteamTurbine,
)


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


def test_segment_mass_flow_uses_all_higher_pressure_condensate_flash():
    state = SimpleNamespace(
        is_high_p_cond_flash=True,
        m_k=[0.0, 10.0, 5.0, 0.0],
        h_tar=[0.0, 200.0, 150.0, 100.0],
        Q_users=[0.0, 0.0, 0.0, 1000.0],
        h_out=[0.0, 0.0, 0.0, 300.0],
    )

    corrected = turbine_mod._segment_mass_flow(state, index=3, mass_flow=20.0)

    state.is_high_p_cond_flash = False
    uncorrected = turbine_mod._segment_mass_flow(state, index=3, mass_flow=20.0)

    assert corrected == pytest.approx(1.25)
    assert uncorrected == pytest.approx(5.0)


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

    with pytest.raises(ValueError, match="mode must"):
        turbine.solve(np.array([180.0]), np.array([100.0]), mode="sideways")


def test_turbine_core_helpers_cover_limit_and_model_branches():
    assert turbine_mod._apply_efficiency_limits(10.0, 0.0, 0.3) == (0.0, 0.3)
    assert turbine_mod._segment_enthalpy(3_000.0, 100.0, 0.0, 0.9) == 3_000.0
    assert (
        turbine_mod._predict_stage_work(
            model="unknown",
            pressure_in=10.0,
            enthalpy_in=3_000.0,
            pressure_out=5.0,
            saturation_enthalpy=2_500.0,
            mass_flow=1.0,
            mass_flow_max=1.0,
            dh_isentropic=100.0,
            mech_eff=0.9,
            min_eff=0.1,
        )
        == 0.0
    )
    assert (
        turbine_mod._predict_stage_work(
            model="Fixed Isentropic Turbine",
            pressure_in=10.0,
            enthalpy_in=3_000.0,
            pressure_out=5.0,
            saturation_enthalpy=2_500.0,
            mass_flow=0.0,
            mass_flow_max=1.0,
            dh_isentropic=100.0,
            mech_eff=0.9,
            min_eff=0.1,
        )
        == 0.0
    )
    assert (
        turbine_mod._predict_stage_work(
            model=turbine_mod.TurbineModel.SUN_SMITH.value,
            pressure_in=10.0,
            enthalpy_in=3_000.0,
            pressure_out=5.0,
            saturation_enthalpy=2_500.0,
            mass_flow=1.0,
            mass_flow_max=0.0,
            dh_isentropic=100.0,
            mech_eff=0.9,
            min_eff=0.1,
        )
        == 0.0
    )
    assert np.isfinite(
        turbine_mod._predict_stage_work(
            model=turbine_mod.TurbineModel.MEDINA_FLORES.value,
            pressure_in=10.0,
            enthalpy_in=3_000.0,
            pressure_out=5.0,
            saturation_enthalpy=2_500.0,
            mass_flow=1.0,
            mass_flow_max=1.0,
            dh_isentropic=100.0,
            mech_eff=0.9,
            min_eff=0.1,
        )
    )
    assert (
        turbine_mod._predict_stage_work(
            model=turbine_mod.TurbineModel.VARBANOV.value,
            pressure_in=10.0,
            enthalpy_in=3_000.0,
            pressure_out=5.0,
            saturation_enthalpy=2_500.0,
            mass_flow=1.0,
            mass_flow_max=1.0,
            dh_isentropic=100.0,
            mech_eff=0.9,
            min_eff=0.1,
        )
        >= 0.0
    )


def test_turbine_stage_input_normalisation_and_empty_results(monkeypatch):
    _patch_steam_properties(monkeypatch)
    turbine = MultiStageSteamTurbine()

    T_arr, Q_arr, source_idx = turbine._normalise_stage_inputs(180.0, 100.0)
    np.testing.assert_allclose(T_arr, [180.0])
    np.testing.assert_allclose(Q_arr, [100.0])
    np.testing.assert_array_equal(source_idx, [0])

    with pytest.raises(ValueError, match="1D"):
        turbine._normalise_stage_inputs(np.array([[180.0]]), np.array([[100.0]]))
    with pytest.raises(ValueError, match="finite"):
        turbine._normalise_stage_inputs(np.array([np.nan]), np.array([100.0]))

    T_arr, Q_arr, source_idx = turbine._normalise_stage_inputs(
        np.array([180.0]),
        np.array([0.0]),
    )
    assert T_arr.size == Q_arr.size == source_idx.size == 0

    above_work, above = turbine.solve(
        np.array([180.0]),
        np.array([100.0]),
        mode="above_pinch",
        T_in=300.0,
        P_in=0.1,
    )
    assert above_work == 0.0
    assert above["stages"] == []

    below_work, below = turbine.solve(
        np.array([20.0]),
        np.array([100.0]),
        mode="below_pinch",
        T_sink=25.0,
    )
    assert below_work == 0.0
    assert below["stages"] == []


def test_turbine_solved_accessors_and_segment_mass_flow_edges(monkeypatch):
    _patch_steam_properties(monkeypatch)

    turbine = MultiStageSteamTurbine()
    total_work, _details = turbine.solve(
        np.array([180.0]),
        np.array([100.0]),
        mode="below_pinch",
        T_sink=25.0,
        model="Fixed Isentropic Turbine",
    )

    assert turbine.stages == turbine.result.stages
    assert turbine.total_work == pytest.approx(total_work)
    assert (
        turbine_mod._segment_mass_flow(
            SimpleNamespace(),
            index=1,
            mass_flow=0.0,
        )
        == 0.0
    )
    assert (
        turbine_mod._segment_mass_flow(
            SimpleNamespace(
                is_high_p_cond_flash=False,
                Q_users=[0.0, 10.0],
                h_out=[0.0, 100.0],
                h_tar=[0.0, 100.0],
            ),
            index=1,
            mass_flow=1.0,
        )
        == 0.0
    )


def test_turbine_below_pinch_skips_stage_without_boiler_enthalpy(monkeypatch):
    monkeypatch.setattr(turbine_mod, "psat_T", lambda t: t)
    monkeypatch.setattr(turbine_mod, "hL_p", lambda p: 100.0)
    monkeypatch.setattr(turbine_mod, "hV_p", lambda p: 100.0)

    turbine = MultiStageSteamTurbine()
    result = turbine._solve_below_pinch(
        stage_temperatures=np.array([80.0]),
        stage_heat_flows=np.array([100.0]),
        source_indices=np.array([0]),
        T_sink=25.0,
        params={
            "model": "Fixed Isentropic Turbine",
            "min_eff": 0.1,
            "load_frac": 1.0,
            "mech_eff": 1.0,
            "is_high_p_cond_flash": False,
        },
    )

    assert result["stages"] == []
    assert result["total_work"] == 0.0
