import math

import numpy as np
import pytest

CoolProp = pytest.importorskip("CoolProp")
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots.SimpleCycles import StateContainer

from OpenPinch.classes.simple_heat_pump import HeatPumpCycle


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
    cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    cycle.solve_t_dt(
        cycle_inputs["Te"],
        cycle_inputs["Tc"],
        cycle_inputs["dT_sh"],
        cycle_inputs["dT_sc"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    assert len(cycle.Hs) == HeatPumpCycle.STATECOUNT
    assert len(cycle.Ts) == HeatPumpCycle.STATECOUNT
    assert len(cycle.Ps) == HeatPumpCycle.STATECOUNT
    assert cycle.w_net < 0.0
    assert cycle.q_evap > 0.0
    assert cycle.COP_heating() > 1.0
    assert cycle.COP_cooling() > 1.0


def test_direct_solve_matches_temperature_based_solution(cycle_inputs):
    ref_cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    ref_cycle.solve_t_dt(
        cycle_inputs["Te"],
        cycle_inputs["Tc"],
        cycle_inputs["dT_sh"],
        cycle_inputs["dT_sc"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    direct_cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    direct_cycle.solve(
        cycle_inputs["T0"],
        cycle_inputs["p0"],
        cycle_inputs["T2"],
        cycle_inputs["p2"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    np.testing.assert_allclose(direct_cycle.Hs, ref_cycle.Hs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(direct_cycle.Ts, ref_cycle.Ts, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(direct_cycle.Ps, ref_cycle.Ps, rtol=1e-6, atol=1e-6)
    assert math.isclose(direct_cycle.w_net, ref_cycle.w_net, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(direct_cycle.q_evap, ref_cycle.q_evap, rel_tol=1e-6, abs_tol=1e-6)


def test_pressure_based_solution_matches_reference(cycle_inputs):
    ref_cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    ref_cycle.solve(
        cycle_inputs["T0"],
        cycle_inputs["p0"],
        cycle_inputs["T2"],
        cycle_inputs["p2"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    pressure_cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    pressure_cycle.solve_p_dt(
        cycle_inputs["p0"],
        cycle_inputs["p2"],
        cycle_inputs["dT_sh"],
        cycle_inputs["dT_sc"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    np.testing.assert_allclose(pressure_cycle.Hs, ref_cycle.Hs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pressure_cycle.Ts, ref_cycle.Ts, rtol=1e-6, atol=1e-6)


def test_fill_states_populates_missing_properties(cycle_inputs):
    cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    cycle.solve_t_dt(
        cycle_inputs["Te"],
        cycle_inputs["Tc"],
        cycle_inputs["dT_sh"],
        cycle_inputs["dT_sc"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    container = StateContainer(unit_system=cycle.system)
    for idx in range(HeatPumpCycle.STATECOUNT):
        container[idx, CoolProp.iP] = cycle.states[idx, CoolProp.iP]
        container[idx, CoolProp.iT] = cycle.states[idx, CoolProp.iT]

    filled = cycle.fill_states(container)
    for idx in range(HeatPumpCycle.STATECOUNT):
        assert filled[idx, "H"] is not None
        assert filled[idx, "S"] is not None


def test_get_hp_th_profiles_returns_expected_shapes(cycle_inputs):
    cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    cycle.solve_t_dt(
        cycle_inputs["Te"],
        cycle_inputs["Tc"],
        cycle_inputs["dT_sh"],
        cycle_inputs["dT_sc"],
        cycle_inputs["eta"],
        fluid=cycle_inputs["fluid"],
        SI=True,
    )

    condenser_profile, evaporator_profile = cycle.get_hp_th_profiles()

    assert condenser_profile.shape == (4, 2)
    assert evaporator_profile.shape == (3, 2)
    assert np.all(np.isfinite(condenser_profile))
    assert np.all(np.isfinite(evaporator_profile))


def test_solve_p_dt_rejects_inverted_pressures(cycle_inputs):
    cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"], unit_system="SI")
    with pytest.raises(ValueError, match="Evaporator pressure must be below condenser pressure"):
        cycle.solve_p_dt(
            cycle_inputs["p2"],
            cycle_inputs["p0"],
            cycle_inputs["dT_sh"],
            cycle_inputs["dT_sc"],
            cycle_inputs["eta"],
            fluid=cycle_inputs["fluid"],
            SI=True,
        )


def test_system_property_updates_unit_container(cycle_inputs):
    cycle = HeatPumpCycle(refrigerant=cycle_inputs["fluid"])
    cycle.system = "SI"
    assert cycle.system is HeatPumpCycle.UNIT_SYSTEMS["SI"]
    assert cycle.cycle_states.units is cycle.system
