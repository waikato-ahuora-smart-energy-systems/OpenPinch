"""A site design's heat pump heats and cools the site's utility levels.

The user chose on 2026-11-24 that site design takes ``heat_pump=`` with
heat pumps that heat and cool utility streams, not process streams. Its
case at one dt_cont multiplier is the utility heat pump target's, placed
on the site, with its condenser and evaporator as zones of the site.
"""

from __future__ import annotations

import pytest

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


def test_a_site_design_with_several_periods_and_a_heat_pump_is_refused():
    # The utility heat pump's zones are not built over several periods
    # (with_heat_pump_streams refuses them too); refused before any design.
    from OpenPinch import PinchProblem
    from tests.analysis.heat_pumps.two_level_site import two_level_site_source

    source = two_level_site_source(
        {"PROBLEM_PERIOD_IDS": ["base", "peak"], "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5]}
    )
    problem = PinchProblem(source=source, project_name="periods")

    with pytest.raises(ValueError, match="utility heat pump.*2 periods"):
        problem.design.total_site_heat_exchanger_network(heat_pump="carnot")


def test_the_exchanger_cap_counts_a_utility_heat_pumps_streams():
    # HENS_HEAT_PUMP_STREAM_EXCHANGERS caps the exchangers on a heat pump's
    # condensers and evaporators wherever the machine is placed.
    from OpenPinch.application.heat_pump_design import capped_heat_pump_streams

    run = heat_pump_design_case(
        recovering_two_level_site(),
        1.0,
        placement="utility",
        load={"fraction": 1.0},
        levels={"condensers": 1, "evaporators": 1},
        search=ONE_EVALUATION,
    )

    capped = capped_heat_pump_streams(run.case)

    assert {key.rsplit(".", 1)[-1] for key in capped} == {"HP_H1", "HP_C1"}


def test_a_utility_case_that_would_not_target_as_the_heat_pump_did_is_refused():
    # MPS is a hot utility alone, so the condenser cannot raise it: the case
    # would look complete and target wrongly, as with_heat_pump_streams
    # refuses too.
    from OpenPinch import PinchProblem
    from tests.analysis.heat_pumps.two_level_site import two_level_site_source

    source = two_level_site_source()
    mps = next(u for u in source["utilities"] if u["name"] == "MPS")
    mps["type"] = "Hot"
    problem = PinchProblem(source=source, project_name="one sided")

    with pytest.raises(ValueError, match="type Both"):
        heat_pump_design_case(
            problem,
            1.0,
            placement="utility",
            load={"fraction": 1.0},
            levels={"condensers": 1, "evaporators": 1},
            search=ONE_EVALUATION,
        )
