"""Cases and readers shared by the heat pump stream case tests."""

from __future__ import annotations

import json
from pathlib import Path

import OpenPinch
from OpenPinch import PinchProblem

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)


def basic_source() -> dict:
    return json.loads(BASIC_PINCH.read_text(encoding="utf-8"))


def two_zone_problem() -> PinchProblem:
    """The basic pinch case with its streams dealt to process zones A and B."""

    source = basic_source()
    for n, record in enumerate(source["streams"]):
        record["zone"] = "A" if n % 2 == 0 else "B"
    source["zone_tree"] = {
        "name": "Site",
        "type": "Site",
        "children": [
            {"name": "A", "type": "Process Zone"},
            {"name": "B", "type": "Process Zone"},
        ],
    }
    return PinchProblem(source=source, project_name="two")


def full_load_target(problem: PinchProblem, **options):
    """The problem's Carnot heat pump target at full load, on a small search.

    One start and 100 evaluations: 0.15 s against 17 s for the default
    search, and a few seconds against minutes in Pyodide. The target stops
    short of the best (715 kW of condenser on the basic pinch case, where
    the full search finds 750 kW), which the builder does not mind: its
    case reproduces whatever target it is given.
    """

    return problem.target.carnot_heat_pump(
        **{
            "load_fraction": 1.0,
            "condensers": 1,
            "evaporators": 1,
            "maximum_restarts": 1,
            "maximum_evaluations": 100,
        }
        | options
    )


def heat_pump_zones(case: PinchProblem) -> set[str]:
    """The zones holding the case's heat pump streams."""

    return {
        record["zone"]
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
