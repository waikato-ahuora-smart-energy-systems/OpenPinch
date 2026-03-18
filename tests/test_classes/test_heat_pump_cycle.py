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
    Te,
    Tc,
    dT_sh,
    dT_sc,
):
    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert len(cycle.Hs) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ts) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ps) == SimpleHeatPumpCycle.STATECOUNT
    assert np.isclose(cycle.q_cond - cycle.q_evap - cycle.w_net, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0)
    assert np.isclose(sum([s.heat_flow for s in cond_streams]) - cycle.Q_cond, 0)
    assert np.isclose(min([s.t_target for s in cond_streams]) - (Tc - dT_sc), 0, atol=0.02)
    assert np.isclose(evap_streams[-1].t_target - (Te + dT_sh), 0, atol=0.02)

    if evap_streams[0].type == StreamType.Cold.value:
        assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_evap, 0)
    else:
        assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_evap * -1, 0)

    assert cycle.COP_h >= 1 if cycle.q_evap >= 0 else cycle.COP_h < 1
    assert cycle.COP_r >= 0 if cycle.q_evap >= 0 else cycle.COP_r < 0


def test_heat_pump_cycle_case_1():
    Te = -15 # degC, evaporator saturation temperature
    Tc = 35 # degC, condenser saturation temperature
    dT_sh = 5.0  # K superheat
    dT_sc = 5.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        dT_sh = dT_sh,
        dT_sc = dT_sc,
        ihx_gas_dt=5.0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_h_total=1000,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)


def test_heat_pump_cycle_case_2():
    Te = -15 # degC, evaporator saturation temperature
    Tc = 35 # degC, condenser saturation temperature
    dT_sh = 0.0  # K superheat
    dT_sc = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        ihx_gas_dt=5.0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_h_total=1000,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)


def test_heat_pump_cycle_case_3():
    Te = -15 # degC, evaporator saturation temperature
    Tc = 35 # degC, condenser saturation temperature
    dT_sh = 0.0  # K superheat
    dT_sc = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        ihx_gas_dt=5.0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_h_total=0,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)


def test_heat_pump_cycle_case_4():
    Te = -15 # degC, evaporator saturation temperature
    Tc = 35 # degC, condenser saturation temperature
    dT_sh = 5.0  # K superheat
    dT_sc = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        dT_sh = dT_sh,
        dT_sc = dT_sc,        
        ihx_gas_dt=0,
        eta_comp = 0.75,
        refrigerant="R134a",
        Q_h_total=1000,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)


def test_heat_pump_cycle_case_5():
    Te = 5 # degC, evaporator saturation temperature
    Tc = 170 # degC, condenser saturation temperature
    dT_sh = 10.0  # K superheat
    dT_sc = 10.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        dT_sh = dT_sh,
        dT_sc = dT_sc,        
        ihx_gas_dt=0,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_h_total=1000,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)


def test_heat_pump_cycle_case_6():
    Te = 5 # degC, evaporator saturation temperature
    Tc = 170 # degC, condenser saturation temperature
    dT_sh = 0.0  # K superheat
    dT_sc = 0.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        dT_sh = dT_sh,
        dT_sc = dT_sc,        
        ihx_gas_dt=0,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_h_total=1000,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)


def test_heat_pump_cycle_case_6():
    Te = 5 # degC, evaporator saturation temperature
    Tc = 170 # degC, condenser saturation temperature
    dT_sh = 10.0  # K superheat
    dT_sc = 10.0  # K subcooling

    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te = Te,
        Tc = Tc,
        dT_sh = dT_sh,
        dT_sc = dT_sc,        
        ihx_gas_dt=10,
        eta_comp = 0.75,
        refrigerant="R601",
        Q_h_total=1000,
    )

    _validate_results(cycle, Te, Tc, dT_sh, dT_sc)

