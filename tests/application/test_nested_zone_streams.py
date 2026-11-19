"""A process zone that holds a unit operation keeps its own streams.

Streams can sit on a process zone and on a unit operation inside it. The
process zone and the site both see every one of them, so targeting the
process zone or the site counts the unit operation's heat and its own.
"""

from __future__ import annotations

import pytest

from OpenPinch.application.problem import PinchProblem


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(zone, name, supply, target, heat_flow):
    return {
        "zone": zone,
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(5.0, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


def _problem(*, unit_operation: bool) -> PinchProblem:
    process = {"name": "A", "type": "Process Zone"}
    streams = [
        _stream("Site/A", "H1", 200.0, 60.0, 140.0),
        _stream("Site/A", "C1", 50.0, 150.0, 100.0),
    ]
    if unit_operation:
        process["children"] = [{"name": "U", "type": "Unit Operation"}]
        streams.append(_stream("Site/A/U", "H2", 180.0, 90.0, 90.0))
    return PinchProblem(
        source={
            "streams": streams,
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {"name": "Site", "type": "Site", "children": [process]},
        }
    )


def _names(zone):
    return sorted(s.name for s in [*zone.hot_streams, *zone.cold_streams])


def test_a_process_zone_with_a_unit_operation_keeps_its_own_streams():
    problem = _problem(unit_operation=True)
    process = problem.master_zone.subzones["A"]

    assert _names(process) == ["C1", "H1", "H2"]
    assert _names(problem.master_zone) == ["C1", "H1", "H2"]


def test_the_unit_operations_heat_adds_to_its_process_zones_cold_utility():
    """H2 is a pure heat surplus below the process's hot end, so the process
    zone's cold utility target rises by its 90 kW."""

    def cold_utility(problem):
        target = problem.target.direct_heat_integration(zone="Site/A")
        return target.cold_utility_target

    assert cold_utility(_problem(unit_operation=False)) == pytest.approx(40.0, abs=1e-6)
    assert cold_utility(_problem(unit_operation=True)) == pytest.approx(130.0, abs=1e-6)
