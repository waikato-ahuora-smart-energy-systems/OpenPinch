"""Extra property/helper branch coverage for ``SimpleHeatPumpCycle``."""

from __future__ import annotations

import pytest

from CoolProp.Plots.SimpleCycles import StateContainer
import CoolProp

from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle


def test_simple_heat_pump_state_and_cycle_state_setters():
    hp = SimpleHeatPumpCycle()

    assert hp.system is not None
    hp.state = "HEOS::Water"
    assert hp.state is not None
    assert hp.solved is False

    bad = StateContainer(unit_system=hp.system)
    bad[0, "H"] = 1.0
    with pytest.raises(ValueError, match="Expected exactly"):
        hp.cycle_states = bad

    good = StateContainer(unit_system=hp.system)
    for i in range(hp.STATECOUNT):
        good[i, "H"] = 1_000.0 + i
        good[i, "S"] = 10.0 + i
        good[i, "P"] = 100_000.0 + i
        good[i, "T"] = 300.0 + i
    hp.cycle_states = good

    assert hp.solved is True
    assert len(hp.state_points) == hp.STATECOUNT
    assert len(hp.states) == hp.STATECOUNT
    assert len(hp.Hs) == hp.STATECOUNT
    assert len(hp.Ss) == hp.STATECOUNT
    assert len(hp.Ts) == hp.STATECOUNT
    assert len(hp.Ps) == hp.STATECOUNT


def test_simple_heat_pump_cop_zero_division_and_misc_helpers(monkeypatch):
    hp = SimpleHeatPumpCycle()
    hp._solved = True
    hp._q_cond = 10.0
    hp._q_evap = 5.0
    hp._w_net = 0.0
    with pytest.raises(ZeroDivisionError):
        _ = hp.COP_h
    with pytest.raises(ZeroDivisionError):
        _ = hp.COP_r

    hp.dtcont = 3.0
    assert hp.dtcont == 3.0
    assert hp._convert_C_to_K(10.0) == pytest.approx(283.15)
    assert hp._convert_K_to_C(283.15) == pytest.approx(10.0)

    class _FakeState:
        def __init__(self):
            self.calls = []

        def update(self, mode, *vals):
            self.calls.append((mode, vals))
            if mode == CoolProp.PT_INPUTS:
                raise RuntimeError("force fallback")

        def hmass(self):
            return 10.0

        def smass(self):
            return 2.0

        def p(self):
            return 100_000.0

        def T(self):
            return 300.0

    fake_state = _FakeState()
    hp._state = fake_state
    hp._cycle_states = StateContainer(unit_system=hp.system)
    hp._compute_state_from_pressure_temperature(100_000.0, 300.0, phase=1)
    hp._save_cycle_state(0)

    assert fake_state.calls[0][0] == CoolProp.PT_INPUTS
    assert fake_state.calls[1][0] == CoolProp.PQ_INPUTS
    assert hp._cycle_states[0, "H"] == pytest.approx(10.0)
