"""A utility heat pump joins a site as two process zones.

A heat pump on the utility system takes heat from one utility level and
delivers it to a hotter one. In the new case its condensers are the hot
streams of one process zone and its evaporators the cold streams of
another, both under the site the target was solved on. Two zones, not
one: streams of one zone recover heat from each other, and the condenser
must not heat its own evaporator.
"""

from __future__ import annotations

from tests.analysis.heat_pumps.two_level_site import (
    two_level_site,
    utility_heat_pump_target,
)


def test_the_condenser_and_the_evaporator_each_get_a_process_zone_of_the_site():
    problem = two_level_site()
    target = utility_heat_pump_target(problem)

    case = problem.with_heat_pump_streams(target)

    source = case.to_problem_json()
    marked = {
        record["name"]: record
        for record in source["streams"]
        if record["heat_pump"] is not None
    }
    assert {name: record["zone"] for name, record in marked.items()} == {
        "HP_H1": "two level site/HP condenser",
        "HP_C1": "two level site/HP evaporator",
    }
    assert {name: record["heat_pump"] for name, record in marked.items()} == {
        "HP_H1": {"machine": "HP", "role": "condenser", "placement": "utility"},
        "HP_C1": {"machine": "HP", "role": "evaporator", "placement": "utility"},
    }
    zones = {zone["name"]: zone["type"] for zone in source["zone_tree"]["children"]}
    assert zones == {
        "A": "Process Zone",
        "B": "Process Zone",
        "HP condenser": "Process Zone",
        "HP evaporator": "Process Zone",
    }
