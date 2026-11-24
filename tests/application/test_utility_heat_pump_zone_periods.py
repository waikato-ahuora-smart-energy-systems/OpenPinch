"""A utility heat pump's zones over several periods.

A shared-design utility target (``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``)
holds its condenser and evaporator streams in every period. The zones the
site design takes write each stream with its temperatures and duty in each
period, as a process heat pump's streams are written, so a site with
several periods can be designed with its heat pump.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _case_with_utility_heat_pump_zones
from tests.analysis.heat_pumps.two_level_site import two_level_site_source

PERIODS = {"PROBLEM_PERIOD_IDS": ["base", "peak"], "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5]}


def _shared_target(problem):
    return problem.target.heat_pump_and_refrigeration(
        "carnot",
        placement="utility",
        load={"fraction": 1.0},
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 50},
        options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True},
    )


def test_the_zones_write_each_heat_pump_stream_in_every_period():
    problem = PinchProblem(
        source=two_level_site_source(PERIODS), project_name="periods"
    )

    case = _case_with_utility_heat_pump_zones(
        problem, _shared_target(problem), dt_cont=1.0
    )

    records = {
        record["name"]: record
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    assert len(records["HP_H1"]["heat_flow"].get("values", ())) == 2
    assert len(records["HP_C1"]["t_supply"].get("values", ())) == 2


def test_with_heat_pump_streams_takes_a_shared_design_on_a_site_of_periods():
    problem = PinchProblem(
        source=two_level_site_source(PERIODS), project_name="periods"
    )

    case = problem.with_heat_pump_streams(_shared_target(problem))

    zones = {child["name"] for child in case.to_problem_json()["zone_tree"]["children"]}
    assert {"HP condenser", "HP evaporator"} <= zones
