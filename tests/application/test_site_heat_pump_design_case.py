"""A site design's heat pump heats and cools the site's utility levels.

The user chose on 2026-11-24 that site design takes ``heat_pump=`` with
heat pumps that heat and cool utility streams, not process streams. Its
case at one dt_cont multiplier is the utility heat pump target's, placed
on the site, with its condenser and evaporator as zones of the site.
"""

from __future__ import annotations

from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.analysis.heat_pumps.two_level_site import recovering_two_level_site

ONE_EVALUATION = {"restarts": 0, "evaluations": 1}


def test_a_utility_heat_pump_case_holds_the_machine_as_zones_of_the_site():
    run = heat_pump_design_case(
        recovering_two_level_site(),
        1.0,
        placement="utility",
        load={"fraction": 1.0},
        levels={"condensers": 1, "evaporators": 1},
        search=ONE_EVALUATION,
    )

    marked = {
        record["name"]: (record["zone"], record["heat_pump"]["placement"])
        for record in run.case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    assert marked == {
        "HP_H1": ("two level site/HP condenser", "utility"),
        "HP_C1": ("two level site/HP evaporator", "utility"),
    }
