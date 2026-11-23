"""A utility heat pump's streams take none of the process match rules.

A heat pump on the process must exchange with process streams, so its
condenser may not reach a cold utility nor its evaporator a hot one. A
heat pump on the utility system is the reverse: its condenser raises a
utility level and its evaporator is heated from one. Streams marked with
``placement="utility"`` are therefore left out of the roles the rules
are written from.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    heat_pump_stream_roles,
)
from tests.analysis.heat_pumps.two_level_site import two_level_site_source


def _marked(zone: str, name: str, supply: float, target: float, role: str) -> dict:
    return {
        "zone": zone,
        "name": name,
        "t_supply": supply,
        "t_target": target,
        "heat_flow": 500.0,
        "heat_pump": {"machine": "HP", "role": role, "placement": "utility"},
    }


def test_streams_of_a_utility_heat_pump_have_no_process_role():
    source = two_level_site_source()
    source["streams"] += [
        _marked("HP condenser", "HP_H1", 181.0, 180.99, "condenser"),
        _marked("HP evaporator", "HP_C1", 138.89, 138.9, "evaporator"),
    ]
    problem = PinchProblem(source=source, project_name="site")

    assert heat_pump_stream_roles(problem) == {}
