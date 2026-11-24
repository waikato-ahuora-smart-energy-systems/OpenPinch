"""Site design ranks a utility heat pump's designs with the plain site's.

``design.total_site_heat_exchanger_network(heat_pump=...)`` designs the
site without a heat pump and, at each dt_cont multiplier, the site with
a utility heat pump's zones, and ranks every network by its site cost,
the machine's electricity and compressor included. On the two-level
site, a 50-evaluation target (COP 2.65) designs to a site cost of
76,000 $/y but its machine costs 273,000 $/y, so the plain site
(315,000 $/y) is selected: the heat pump networks rank, and lose.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import ranking_cost
from OpenPinch.analysis.heat_exchanger_networks.results.selection import (
    ranked_networks,
)
from tests.analysis.heat_pumps.two_level_site import recovering_two_level_site
from tests.application.test_utility_heat_pump_site_design import _solvers_available


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_design_ranks_its_utility_heat_pump_networks_with_the_plain_ones():
    view = recovering_two_level_site().design.total_site_heat_exchanger_network(
        width=0,
        depth=1,
        heat_pump={
            "cycle": "carnot",
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 50},
        },
        heat_pump_cost=300.0,
    )

    networks = [outcome.network for outcome in ranked_networks(view.result)]
    pumped = [n for n in networks if n.heat_pumps]
    assert pumped
    assert all(n.heat_pumps[0].machine == "HP" for n in pumped)
    costs = [ranking_cost(n, n.total_annual_cost) for n in networks]
    selected = view.selected_network
    assert ranking_cost(selected, selected.total_annual_cost) == min(costs)
    assert not selected.heat_pumps


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_heat_pump_case_the_site_cannot_hold_is_left_out_with_a_warning():
    # With MPS a hot utility alone the condenser cannot raise it, so the
    # case is refused; the site is still designed without the heat pump.
    from OpenPinch import PinchProblem

    source = recovering_two_level_site().to_problem_json()
    mps = next(u for u in source["utilities"] if u["name"] == "MPS")
    mps["type"] = "Hot"
    problem = PinchProblem(source=source, project_name="one sided")

    with pytest.warns(UserWarning, match="No heat pump network.*type Both"):
        view = problem.design.total_site_heat_exchanger_network(
            width=0,
            depth=1,
            heat_pump={
                "cycle": "carnot",
                "load": {"fraction": 1.0},
                "levels": {"condensers": 1, "evaporators": 1},
                "search": {"restarts": 0, "evaluations": 50},
            },
            heat_pump_cost=300.0,
        )

    assert not any(o.network.heat_pumps for o in ranked_networks(view.result))


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_heat_pump_is_tried_at_several_approaches(monkeypatch):
    # Standing the machine HPR_DT_CONT (1 K) off the steam made its
    # exchangers 2,804 m2: with a full search it lost to the plain site
    # (330,740 against 315,299 $/y), where at 5 K it won (259,717 $/y).
    # Without a given approach the site design tries HPR_DT_CONT times 1, 5
    # and 10, one machine each, and the ranking chooses.
    from OpenPinch.application import heat_pump_design

    built = heat_pump_design.heat_pump_design_case
    approaches = []

    def recording(problem, multiplier, **arguments):
        approaches.append(arguments["levels"].minimum_approach_temperature)
        return built(problem, multiplier, **arguments)

    monkeypatch.setattr(heat_pump_design, "heat_pump_design_case", recording)

    recovering_two_level_site().design.total_site_heat_exchanger_network(
        width=0,
        depth=1,
        heat_pump={
            "cycle": "carnot",
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 50},
        },
        heat_pump_cost=300.0,
    )

    assert approaches == [1.0, 5.0, 10.0]


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_of_several_periods_is_designed_with_its_heat_pump():
    # The zones are written per period from a shared-design target, so a
    # site of several periods is designed with its heat pump, not refused.
    from OpenPinch import PinchProblem

    source = recovering_two_level_site().to_problem_json()
    source["options"] |= {
        "PROBLEM_PERIOD_IDS": ["base", "peak"],
        "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
    }
    problem = PinchProblem(source=source, project_name="periods")

    view = problem.design.total_site_heat_exchanger_network(
        width=0,
        depth=1,
        heat_pump={
            "cycle": "carnot",
            "load": {"fraction": 1.0},
            "levels": {
                "condensers": 1,
                "evaporators": 1,
                "minimum_approach_temperature": 5.0,
            },
            "search": {"restarts": 0, "evaluations": 50},
        },
        heat_pump_cost=300.0,
    )

    assert any(o.network.heat_pumps for o in ranked_networks(view.result))
