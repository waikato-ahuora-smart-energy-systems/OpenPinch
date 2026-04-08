"""Additional branch coverage tests for SimpleHeatPumpCycle."""

import numpy as np
import pytest
import CoolProp

from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle
import OpenPinch.classes.simple_heat_pump as shp_mod


class _DummyState:
    def __init__(self):
        self._h = 1.0
        self._T = 300.0
        self.calls = []

    def update(self, *args):
        self.calls.append(args)
        input_key = args[0]
        if input_key == CoolProp.PQ_INPUTS and args[2] == 1.0:
            self._h = 4.0
        elif input_key == CoolProp.PQ_INPUTS and args[2] == 0.0:
            self._h = 2.0
        elif input_key == CoolProp.HmassP_INPUTS:
            self._h = float(args[1])
        self._T = 273.15 + self._h

    def keyed_output(self, _key):
        return 10.0

    def hmass(self):
        return self._h

    def smass(self):
        return 1.0

    def T(self):
        return self._T

    def p(self):
        return 1.0


class _CycleStore(dict):
    def __getitem__(self, key):
        return super().get(key, 0.0)


def test_simple_cycle_property_getters_and_require_solution():
    hp = SimpleHeatPumpCycle()
    hp._cycle_states = "cycle-states"
    hp._q_cas_cool = 1.0
    hp._q_cool = 2.0
    hp._q_cas_heat = 3.0
    hp._q_heat = 4.0
    hp._penalty = 0.5
    hp._dt_diff_max = 0.25
    hp._refrigerant = "water"
    hp._T_evap = 25.0
    hp._T_cond = 70.0
    hp._dT_superheat = 5.0
    hp._dT_subcool = 2.0
    hp._eta_comp = 0.8
    hp._ihx_gas_dt = 7.0

    assert hp.cycle_states == "cycle-states"
    assert hp.q_cas_cool == 1.0
    assert hp.q_cool == 2.0
    assert hp.q_cas_heat == 3.0
    assert hp.q_heat == 4.0
    assert hp.penalty == 0.5
    assert hp.dt_diff_max == 0.25
    assert hp.refrigerant == "water"
    assert hp.T_evap == 25.0
    assert hp.T_cond == 70.0
    assert hp.dT_superheat == 5.0
    assert hp.dT_subcool == 2.0
    assert hp.eta_comp == 0.8
    assert hp.dt_ihx_gas_side == 7.0

    hp._solved = False
    with pytest.raises(RuntimeError, match="Solve the cycle"):
        hp._require_solution()


def test_validate_solve_inputs_and_get_psat_high_temperature_branch():
    hp = SimpleHeatPumpCycle()
    with pytest.raises(ValueError, match="A fluid must be specified"):
        hp._validate_solve_inputs(refrigerant=None)

    hp._state = _DummyState()
    hp._t_crit = 350.0
    hp._d_crit = 1.0
    hp._get_P_sat_from_T(T=360.0)
    assert hp._state.calls[0][0] == CoolProp.DmassT_INPUTS


def test_build_evaporator_profile_subcooled_segment_branch(monkeypatch):
    hp = SimpleHeatPumpCycle()
    hp._solved = True
    hp._state = _DummyState()
    hp._p_crit = 10.0
    hp._dt_diff_max = 0.5

    monkeypatch.setattr(
        SimpleHeatPumpCycle, "Hs", property(lambda self: [0.0, 0.0, 0.0, 1.0, 0.0, 5.0])
    )
    monkeypatch.setattr(SimpleHeatPumpCycle, "Ps", property(lambda self: [1.0]))
    monkeypatch.setattr(
        shp_mod,
        "get_piecewise_data_points",
        lambda curve, is_hot_stream, dt_diff_max: curve,
    )

    out = hp._build_evaporator_profile()
    assert isinstance(out, np.ndarray)
    assert out.shape[0] > 0


def test_solve_raises_on_pressure_and_enthalpy_invalid_states(monkeypatch):
    hp = SimpleHeatPumpCycle()
    hp._state = _DummyState()

    monkeypatch.setattr(hp, "_validate_solve_inputs", lambda refrigerant=None: True)
    monkeypatch.setattr(
        hp, "_get_P_sat_from_T", lambda T, Q=1.0: 5.0 if T < 300 else 1.0
    )
    with pytest.raises(
        ValueError, match="Evaporator pressure must be below condenser pressure"
    ):
        hp.solve(T_evap=20.0, T_cond=40.0, refrigerant="water")

    hp2 = SimpleHeatPumpCycle()
    hp2._state = _DummyState()
    hp2._cycle_states = _CycleStore()
    monkeypatch.setattr(hp2, "_validate_solve_inputs", lambda refrigerant=None: True)
    monkeypatch.setattr(
        hp2, "_get_P_sat_from_T", lambda T, Q=1.0: 1.0 if T < 300 else 5.0
    )
    monkeypatch.setattr(
        hp2, "_compute_state_from_pressure_temperature", lambda **kwargs: hp2._state
    )
    monkeypatch.setattr(
        hp2, "_compute_compressor_outlet_state", lambda **kwargs: hp2._state
    )
    monkeypatch.setattr(
        hp2, "_compute_state_from_pressure_quality", lambda **kwargs: hp2._state
    )
    monkeypatch.setattr(
        hp2, "_compute_state_from_pressure_enthalpy", lambda **kwargs: hp2._state
    )

    def _fake_save(index):
        if index == 1:
            hp2._cycle_states[(1, "H")] = 10.0
            hp2._cycle_states[(1, "P")] = 5.0
        elif index == 2:
            hp2._cycle_states[(2, "H")] = 10.0
            hp2._cycle_states[(2, "P")] = 5.0
        else:
            hp2._cycle_states[(index, "H")] = 1.0
            hp2._cycle_states[(index, "P")] = 1.0

    monkeypatch.setattr(hp2, "_save_cycle_state", _fake_save)

    with pytest.raises(
        ValueError, match="Condenser cannot have a negative or zero enthalpy change"
    ):
        hp2.solve(T_evap=20.0, T_cond=40.0, refrigerant="water")
