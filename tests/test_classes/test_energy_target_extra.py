"""Additional branch coverage tests for EnergyTarget properties."""

from OpenPinch.classes.energy_target import EnergyTarget
from OpenPinch.classes.value import Value


def test_energy_target_identifier_parent_active_and_cost_properties():
    target = EnergyTarget(name="T", identifier="id", parent_zone="Z")
    assert target.identifier == "id"
    assert target.parent_zone == "Z"

    target._active = True
    assert target.active is True
    target._active = Value(0.0)
    assert target.active == 0.0

    target.capital_cost = 123.0
    assert target.capital_cost == 123.0

    target.utility_heat_recovery_target = 44.0
    assert target.utility_heat_recovery_target == 44.0

    target.target_values = {"hot_utility_target": 10.0}
    assert target.target_values == {"hot_utility_target": 10.0}


def test_energy_target_capital_cost_descriptor_access():
    target = EnergyTarget(name="T", identifier="id", parent_zone="Z")
    EnergyTarget.capital_cost.fset(target, 321.0)
    assert EnergyTarget.capital_cost.fget(target) == 321.0
