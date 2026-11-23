"""A site that already has a zone of a heat pump zone's name is refused.

The heat pump's zones are named after the machine, ``"HP condenser"`` and
``"HP evaporator"`` by default. Two zones of one name under a site could
not be told apart, so the clash is refused by name; another ``machine``
name avoids it.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.analysis.heat_pumps.two_level_site import (
    two_level_site_source,
    utility_heat_pump_target,
)


def test_a_site_with_a_zone_named_as_a_heat_pump_zone_is_refused():
    source = two_level_site_source()
    for record in source["streams"]:
        if record["zone"] == "A":
            record["zone"] = "HP condenser"
    problem = PinchProblem(source=source, project_name="clash")
    target = utility_heat_pump_target(problem)

    with pytest.raises(ValueError, match="already has a zone named 'HP condenser'"):
        problem.with_heat_pump_streams(target)

    case = problem.with_heat_pump_streams(target, machine="Lift")
    zones = {zone["name"] for zone in case.to_problem_json()["zone_tree"]["children"]}
    assert zones == {"HP condenser", "B", "Lift condenser", "Lift evaporator"}
