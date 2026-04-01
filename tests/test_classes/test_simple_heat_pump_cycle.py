"""Regression tests for the simple heat pump cycle classes."""

import math

import numpy as np
import pytest

CoolProp = pytest.importorskip("CoolProp")
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots.SimpleCycles import StateContainer

from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle
from OpenPinch.lib.enums import *

def _validate_results(
    cycle: SimpleHeatPumpCycle,
    T_evap,
    T_cond,
    dT_superheat,
    dT_subcool,
):
    """Validate results for this test module."""
    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert len(cycle.Hs) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ts) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ps) == SimpleHeatPumpCycle.STATECOUNT
    assert np.isclose(cycle.q_cond - cycle.q_evap - cycle.w_net, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_heat - cycle.Q_cas_heat, 0)
    assert np.isclose(sum([s.heat_flow for s in cond_streams]) - cycle.Q_heat, 0)
    assert np.isclose(min([s.t_target for s in cond_streams]) - (T_cond - dT_subcool), 0, atol=0.02)
    assert np.isclose(evap_streams[-1].t_target - (T_evap + dT_superheat), 0, atol=0.02)

    if evap_streams[0].type == StreamType.Cold.value:
        assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_cool, 0)
    else:
        assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_cool * -1, 0)

    assert cycle.COP_h >= 1 if cycle.q_evap >= 0 else cycle.COP_h < 1
    assert cycle.COP_r >= 0 if cycle.q_evap >= 0 else cycle.COP_r < 0


def test_heat_pump_cycle_case_1():
    T_evap = -15 # degC, evaporator saturation temperature
    T_cond = 35 # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 5.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,
        dt_ihx_gas_side=5.0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_heat=1000,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_2():
    T_evap = -15 # degC, evaporator saturation temperature
    T_cond = 35 # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dt_ihx_gas_side=5.0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_heat=1000,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_3():
    T_evap = -15 # degC, evaporator saturation temperature
    T_cond = 35 # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dt_ihx_gas_side=5.0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_heat=0,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_4():
    T_evap = -15 # degC, evaporator saturation temperature
    T_cond = 35 # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_heat=1000,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_5():
    T_evap = 5 # degC, evaporator saturation temperature
    T_cond = 170 # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=0,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_heat=1000,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_6():
    T_evap = 5 # degC, evaporator saturation temperature
    T_cond = 170 # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=0,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_heat=1000,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_7():
    T_evap = 5 # degC, evaporator saturation temperature
    T_cond = 170 # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=10,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_heat=1000,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_8():
    T_evap = 5 # degC, evaporator saturation temperature
    T_cond = 170 # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle1 = SimpleHeatPumpCycle()
    cycle1.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=10,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
    )
    _validate_results(cycle1, T_evap, T_cond, dT_superheat, dT_subcool)

    cycle2 = SimpleHeatPumpCycle()
    cycle2.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=10,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
    )
    _validate_results(cycle2, T_evap, T_cond, dT_superheat, dT_subcool)

    assert np.isclose(cycle2.m_dot, cycle1.m_dot, 0.0)
    assert np.isclose(cycle2.Q_cond, cycle1.Q_cond, 0.0)
    assert np.isclose(cycle2.work, cycle1.work, 0.0)


def test_heat_pump_cycle_case_9():
    T_evap = 5 # degC, evaporator saturation temperature
    T_cond = 170 # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=10,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        Q_cool=300,
    )
    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0)
    assert np.isclose(sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0)


def test_heat_pump_cycle_case_10():
    T_evap = 0.0 # degC, evaporator saturation temperature
    T_cond = 24 # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 2.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        T_evap = T_evap,
        T_cond = T_cond,
        dT_superheat = dT_superheat,
        dT_subcool = dT_subcool,        
        dt_ihx_gas_side=10,
        eta_comp = 0.7,
        refrigerant="R134A",
        Q_heat=0.0,
        Q_cas_heat=900,
        Q_cool=np.nan,
    )
    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0)
    assert np.isclose(sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0)
