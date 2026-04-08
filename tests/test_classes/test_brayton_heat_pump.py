"""Tests for the TESPy-backed Brayton heat pump wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.classes import brayton_heat_pump as br


class _FakeComponent:
    def __init__(self, name: str):
        self.name = name
        self.P = SimpleNamespace(val=10.0 if "compressor" in name else -2.0)
        self.Q = SimpleNamespace(val=30.0)

    def set_attr(self, **kwargs):
        if "Q" in kwargs:
            self.Q.val = kwargs["Q"]


class _FakeConnection:
    _state_map = {
        "s1": dict(T=15.0, p=1.0, h=100.0, m=2.0),
        "s2": dict(T=25.0, p=2.0, h=200.0, m=2.0),
        "s3": dict(T=35.0, p=3.0, h=300.0, m=2.0),
        "s4": dict(T=45.0, p=4.0, h=400.0, m=2.0),
        "s5": dict(T=10.0, p=1.0, h=90.0, m=2.0),
    }

    def __init__(self, *_args, label: str):
        init = self._state_map[label]
        self.label = label
        self.T = SimpleNamespace(val=init["T"])
        self.p = SimpleNamespace(val=init["p"])
        self.h = SimpleNamespace(val=init["h"])
        self.m = SimpleNamespace(val=init["m"])

    def set_attr(self, **kwargs):
        if "T" in kwargs:
            self.T.val = kwargs["T"]
        if "p" in kwargs:
            self.p.val = kwargs["p"]
        if "m" in kwargs:
            self.m.val = kwargs["m"]


class _FakeNetwork:
    def __init__(self, **_kwargs):
        self.attrs = {}
        self.connections = ()

    def set_attr(self, **kwargs):
        self.attrs.update(kwargs)

    def add_conns(self, *conns):
        self.connections = conns

    def solve(self, **_kwargs):
        return None


def _patch_tespy(monkeypatch, *, broken_connection: bool = False):
    monkeypatch.setattr(br, "Network", _FakeNetwork)
    monkeypatch.setattr(br, "CycleCloser", _FakeComponent)
    monkeypatch.setattr(br, "Compressor", _FakeComponent)
    monkeypatch.setattr(br, "Turbine", _FakeComponent)
    monkeypatch.setattr(br, "SimpleHeatExchanger", _FakeComponent)
    if broken_connection:

        class _BrokenConnection(_FakeConnection):
            def __init__(self, *_args, label: str):
                super().__init__(*_args, label=label)
                del self.h

        monkeypatch.setattr(br, "Connection", _BrokenConnection)
    else:
        monkeypatch.setattr(br, "Connection", _FakeConnection)


def test_brayton_cycle_requires_solution_before_property_access():
    hp = br.SimpleBraytonHeatPumpCycle()

    with pytest.raises(RuntimeError):
        _ = hp.Hs
    with pytest.raises(RuntimeError):
        _ = hp.Ts
    with pytest.raises(RuntimeError):
        _ = hp.Ps
    with pytest.raises(RuntimeError):
        _ = hp.Ss


def test_brayton_cycle_solve_profiles_and_stream_build(monkeypatch):
    _patch_tespy(monkeypatch)
    hp = br.SimpleBraytonHeatPumpCycle()

    with pytest.warns(UserWarning):
        work = hp.solve(
            T_comp_in=20.0,
            T_comp_out=120.0,
            dT_gc=20.0,
            Q_heat=500.0,
            eta_comp=0.8,
            eta_exp=0.75,
            is_recuperated=True,
            refrigerant="AIR",
        )

    assert work == hp.work_net
    assert hp.Q_heat == 500.0
    assert hp.Q_cool == 30.0
    assert len(hp.Hs) == 4
    assert len(hp.Ts) == 4
    assert len(hp.Ps) == 4
    assert len(hp.Ss) == 4

    hot_profile, cold_profile = hp.get_hp_th_profiles()
    assert hot_profile.shape == (2, 2)
    assert cold_profile.shape == (2, 2)

    hot_sc, cold_sc = hp.get_hp_hot_and_cold_streams()
    assert len(hot_sc) == 1
    assert len(cold_sc) == 1

    combined = hp.build_stream_collection(include_cond=True, include_evap=True)
    assert len(combined) == 2


def test_brayton_stream_build_handles_zero_temperature_span(monkeypatch):
    _patch_tespy(monkeypatch)
    hp = br.SimpleBraytonHeatPumpCycle()
    hp.solve(
        T_comp_in=20.0,
        T_comp_out=120.0,
        dT_gc=20.0,
        Q_heat=200.0,
        eta_comp=0.8,
        eta_exp=0.75,
        is_recuperated=False,
        refrigerant="AIR",
    )

    hp._states[1]["T"] = 55.0
    hp._states[2]["T"] = 55.0
    hp._states[3]["T"] = 40.0
    hp._states[0]["T"] = 40.0

    hot_sc, cold_sc = hp.get_hp_hot_and_cold_streams()
    assert hot_sc[0].t_supply != hot_sc[0].t_target
    assert cold_sc[0].t_supply != cold_sc[0].t_target


def test_brayton_cycle_raises_runtime_error_on_result_extraction_failure(monkeypatch):
    _patch_tespy(monkeypatch, broken_connection=True)
    hp = br.SimpleBraytonHeatPumpCycle()

    with pytest.raises(RuntimeError, match="Failed to extract results"):
        hp.solve(
            T_comp_in=20.0,
            T_comp_out=120.0,
            dT_gc=20.0,
            Q_heat=300.0,
            eta_comp=0.8,
            eta_exp=0.75,
            is_recuperated=False,
            refrigerant="AIR",
        )


# ===== Merged from test_brayton_heat_pump_extra.py =====
"""Additional branch coverage tests for Brayton heat pump wrappers."""

from OpenPinch.classes.brayton_heat_pump import SimpleBraytonHeatPumpCycle


def test_brayton_cycle_states_property_alias():
    cycle = SimpleBraytonHeatPumpCycle()
    assert cycle.cycle_states is cycle.states
