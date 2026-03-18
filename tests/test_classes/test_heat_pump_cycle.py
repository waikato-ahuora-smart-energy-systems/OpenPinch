import math

import numpy as np
import pytest

CoolProp = pytest.importorskip("CoolProp")
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots.SimpleCycles import StateContainer

from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle


@pytest.fixture(scope="module")
def cycle_inputs():
    return {
        "refrigerant": "R134a",
        "Te": -15,  # degC, evaporator saturation temperature
        "Tc": 35, # degC, condenser saturation temperature
        "dT_sh": 5.0,  # K superheat
        "dT_sc": 5.0,  # K subcooling
        "eta_comp": 0.75,
    }


def test_solve_t_dt_establishes_cycle_states(cycle_inputs):
    cycle = SimpleHeatPumpCycle()
    cycle.solve(
        refrigerant=cycle_inputs["refrigerant"],
        Te=cycle_inputs["Te"],
        Tc=cycle_inputs["Tc"],
        dT_sh=cycle_inputs["dT_sh"],
        dT_sc=cycle_inputs["dT_sc"],
        eta_comp=cycle_inputs["eta_comp"],
    )

    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert len(cycle.Hs) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ts) == SimpleHeatPumpCycle.STATECOUNT
    assert len(cycle.Ps) == SimpleHeatPumpCycle.STATECOUNT
    assert np.isclose(cycle.q_cond - cycle.q_evap - cycle.w_net, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0)
    assert cycle.w_net > 0.0
    assert cycle.q_evap > 0.0
    assert cycle.COP_h > 1.0
    assert cycle.COP_r > 0.0
    assert np.isclose(sum([s.heat_flow for s in cond_streams]) - cycle.Q_cond, 0)
    assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_evap, 0)
    assert np.isclose(min([s.t_supply for s in evap_streams]) - cycle_inputs["Te"], 0)
    assert np.isclose(max([s.t_target for s in evap_streams]) - (cycle_inputs["Te"] + cycle_inputs["dT_sh"]), 0)
    assert np.isclose(min([s.t_target for s in cond_streams]) - (cycle_inputs["Tc"] - cycle_inputs["dT_sc"]), 0)

