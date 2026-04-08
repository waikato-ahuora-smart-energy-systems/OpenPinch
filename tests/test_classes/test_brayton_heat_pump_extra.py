"""Additional branch coverage tests for Brayton heat pump wrappers."""

from OpenPinch.classes.brayton_heat_pump import SimpleBraytonHeatPumpCycle


def test_brayton_cycle_states_property_alias():
    cycle = SimpleBraytonHeatPumpCycle()
    assert cycle.cycle_states is cycle.states
