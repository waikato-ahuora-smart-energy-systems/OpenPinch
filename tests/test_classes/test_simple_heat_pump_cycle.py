"""Regression tests for the simple heat pump cycle classes."""

import math
import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots.SimpleCycles import StateContainer
from OpenPinch.classes.vapour_compression_cycle import VapourCompressionCycle
from OpenPinch.lib.enums import *
import CoolProp
import OpenPinch.classes.vapour_compression_cycle as shp_mod


CoolProp = pytest.importorskip("CoolProp")


def _validate_results(
    cycle: VapourCompressionCycle,
    T_evap,
    T_cond,
    dT_superheat,
    dT_subcool,
):
    """Validate results for this test module."""
    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert len(cycle.Hs) == VapourCompressionCycle.STATECOUNT
    assert len(cycle.Ts) == VapourCompressionCycle.STATECOUNT
    assert len(cycle.Ps) == VapourCompressionCycle.STATECOUNT
    assert np.isclose(cycle.q_cond - cycle.q_evap - cycle.w_net, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_heat - cycle.Q_cas_heat, 0)
    assert np.isclose(sum([s.heat_flow for s in cond_streams]) - cycle.Q_heat, 0)
    assert np.isclose(
        min([s.t_target for s in cond_streams]) - (T_cond - dT_subcool), 0, atol=0.02
    )
    assert np.isclose(evap_streams[-1].t_target - (T_evap + dT_superheat), 0, atol=0.02)

    if evap_streams[0].type == StreamType.Cold.value:
        assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_cool, 0)
    else:
        assert np.isclose(
            sum([s.heat_flow for s in evap_streams]) - cycle.Q_cool * -1, 0
        )

    assert cycle.COP_h >= 1 if cycle.q_evap >= 0 else cycle.COP_h < 1
    assert cycle.COP_r >= 0 if cycle.q_evap >= 0 else cycle.COP_r < 0


def test_heat_pump_cycle_case_1():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 5.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_2():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dt_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_3():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dt_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=0,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_4():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_5():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=0,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_6():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=0,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_7():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_8():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle1 = VapourCompressionCycle()
    cycle1.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        is_heat_pump=True,
    )
    _validate_results(cycle1, T_evap, T_cond, dT_superheat, dT_subcool)

    cycle2 = VapourCompressionCycle()
    cycle2.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        is_heat_pump=True,
    )
    _validate_results(cycle2, T_evap, T_cond, dT_superheat, dT_subcool)

    assert np.isclose(cycle2.m_dot, cycle1.m_dot, 0.0)
    assert np.isclose(cycle2.Q_cond, cycle1.Q_cond, 0.0)
    assert np.isclose(cycle2.work, cycle1.work, 0.0)


def test_heat_pump_cycle_case_9():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        Q_cool=300,
        is_heat_pump=True,
    )
    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def test_heat_pump_cycle_case_10():
    T_evap = 0.0  # degC, evaporator saturation temperature
    T_cond = 24  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 2.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=10,
        eta_comp=0.7,
        refrigerant="R134A",
        Q_heat=0.0,
        Q_cas_heat=900,
        Q_cool=np.nan,
        is_heat_pump=True,
    )
    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def test_refrigeration_cycle_uses_q_cool_as_primary_duty():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 5.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_cool=1000.0,
        Q_cas_cool=150.0,
        Q_heat=np.nan,
        is_heat_pump=False,
    )

    assert len(cycle.Hs) == VapourCompressionCycle.STATECOUNT
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0.0)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_heat, cycle.Q_cond, 0.0)
    assert np.isclose(cycle.Q_cas_heat, 0.0, 0.0)
    assert cycle.COP_r > 0.0
    assert cycle.COP_h > 1.0

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def test_refrigeration_cycle_caps_q_heat_to_available_q_cond():
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=-10.0,
        T_cond=30.0,
        dT_superheat=3.0,
        dT_subcool=2.0,
        dt_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_cool=600.0,
        Q_cas_cool=0.0,
        Q_heat=1e9,
        is_heat_pump=False,
    )

    assert cycle.Q_heat <= cycle.Q_cond + 1e-6
    assert np.isclose(cycle.Q_cas_heat, 0.0, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)


# ===== Merged from test_simple_heat_pump_cycle_extra.py =====
"""Additional branch coverage tests for VapourCompressionCycle."""


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
    hp = VapourCompressionCycle()
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
    hp = VapourCompressionCycle()
    with pytest.raises(ValueError, match="A fluid must be specified"):
        hp._validate_solve_inputs(refrigerant=None)

    hp._state = _DummyState()
    hp._t_crit = 350.0
    hp._d_crit = 1.0
    hp._get_P_sat_from_T(T=360.0)
    assert hp._state.calls[0][0] == CoolProp.DmassT_INPUTS


def test_build_evaporator_profile_subcooled_segment_branch(monkeypatch):
    hp = VapourCompressionCycle()
    hp._solved = True
    hp._state = _DummyState()
    hp._p_crit = 10.0
    hp._dt_diff_max = 0.5

    monkeypatch.setattr(
        VapourCompressionCycle,
        "Hs",
        property(lambda self: [0.0, 0.0, 0.0, 1.0, 0.0, 5.0]),
    )
    monkeypatch.setattr(VapourCompressionCycle, "Ps", property(lambda self: [1.0]))
    monkeypatch.setattr(
        shp_mod,
        "get_piecewise_data_points",
        lambda curve, is_hot_stream, dt_diff_max: curve,
    )

    out = hp._build_evaporator_profile()
    assert isinstance(out, np.ndarray)
    assert out.shape[0] > 0


def test_solve_raises_on_pressure_and_enthalpy_invalid_states(monkeypatch):
    hp = VapourCompressionCycle()
    hp._state = _DummyState()

    monkeypatch.setattr(hp, "_validate_solve_inputs", lambda refrigerant=None: True)
    monkeypatch.setattr(
        hp, "_get_P_sat_from_T", lambda T, Q=1.0: 5.0 if T < 300 else 1.0
    )
    with pytest.raises(
        ValueError, match="Evaporator pressure must be below condenser pressure"
    ):
        hp.solve(T_evap=20.0, T_cond=40.0, refrigerant="water")

    hp2 = VapourCompressionCycle()
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


# ===== Merged from test_simple_heat_pump_properties_extra.py =====
"""Extra property/helper branch coverage for ``VapourCompressionCycle``."""


def test_simple_heat_pump_state_and_cycle_state_setters():
    hp = VapourCompressionCycle()

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
    hp._solved = True

    assert hp.solved is True
    assert len(hp.state_points) == hp.STATECOUNT
    assert len(hp.states) == hp.STATECOUNT
    assert len(hp.Hs) == hp.STATECOUNT
    assert len(hp.Ss) == hp.STATECOUNT
    assert len(hp.Ts) == hp.STATECOUNT
    assert len(hp.Ps) == hp.STATECOUNT


def test_simple_heat_pump_cop_zero_division_and_misc_helpers(monkeypatch):
    hp = VapourCompressionCycle()
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
