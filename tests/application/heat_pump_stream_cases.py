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
    """The problem's Carnot heat pump target at full load, with no search.

    The first start alone, at one evaluation: 0.02 s against 17 s for the
    default search. In Pyodide the search runs about a hundred times
    slower, and 100 evaluations in each of these tests added 3.5 minutes
    to a lane with a 20 minute limit. The first start is the target's own
    initial design, which meets the full load on the cases here (750 kW of
    condenser on the basic pinch case); the builder's case reproduces
    whatever target it is given.
    """

    return problem.target.carnot_heat_pump(
        **{
            "load_fraction": 1.0,
            "condensers": 1,
            "evaporators": 1,
            "maximum_restarts": 0,
            "maximum_evaluations": 1,
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
