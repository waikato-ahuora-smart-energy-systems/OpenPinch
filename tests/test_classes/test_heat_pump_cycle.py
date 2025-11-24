import math

import numpy as np
import pytest

CoolProp = pytest.importorskip("CoolProp")
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots.SimpleCycles import StateContainer

from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle


@pytest.fixture(scope="module")
def cycle_inputs():
    fluid = "R134a"
    Te = 268.15  # K, evaporator saturation temperature (-5 °C)
    Tc = 308.15  # K, condenser saturation temperature (35 °C)
    dT_sh = 5.0  # K superheat
    dT_sc = 5.0  # K subcooling
    eta = 0.75

    p0 = PropsSI("P", "T", Te, "Q", 1.0, fluid)
    p2 = PropsSI("P", "T", Tc, "Q", 0.0, fluid)
    T0 = Te + dT_sh
    T2 = Tc - dT_sc

    return {
        "fluid": fluid,
        "Te": Te,
        "Tc": Tc,
        "dT_sh": dT_sh,
        "dT_sc": dT_sc,
        "eta": eta,
        "T0": T0,
        "p0": p0,
        "T2": T2,
        "p2": p2,
    }


def test_solve_t_dt_establishes_cycle_states(cycle_inputs):
    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        Te=cycle_inputs["Te"],
        Tc=cycle_inputs["Tc"],
        dT_sh=cycle_inputs["dT_sh"],
        dT_sc=cycle_inputs["dT_sc"],
        eta_comp=cycle_inputs["eta"],
        refrigerant=cycle_inputs["fluid"],
    )

    assert len(cycle.Hs) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ts) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ps) == SimpleHeatPumpCycle.STATECOUNT
    assert cycle.w_net > 0.0
    assert cycle.q_evap > 0.0
    assert cycle.COP_h > 1.0
    assert cycle.COP_r > 0.0


def test_system_property_updates_unit_container():
    cycle = SimpleHeatPumpCycle()
    cycle.system = "SI"
    assert cycle.system is SimpleHeatPumpCycle.UNIT_SYSTEMS["SI"]
    assert cycle.cycle_states.units is cycle.system
