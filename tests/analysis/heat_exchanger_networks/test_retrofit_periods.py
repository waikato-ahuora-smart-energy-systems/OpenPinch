"""A retrofit must not depend on the order the periods are listed in.

The screens that rank new matches, relocations, heat pumps and loops read a
network and its streams over every period, weighted, not from the first
period alone.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
    NewMatch,
    utility_relief,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    _stream_ends,
    run_retrofit_search,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.application.problem import PinchProblem


def _quantity(value, unit):
    if isinstance(value, list):
        return {"values": value, "unit": unit}
    return {"value": value, "unit": unit}


def _stream(name, t_supply, t_target, heat_flow):
    return {
        "zone": "Site/A",
        "name": name,
        "t_supply": _quantity(t_supply, "degC"),
        "t_target": _quantity(t_target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(5.0, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, t_supply, t_target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(t_supply, "degC"),
        "t_target": _quantity(t_target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _problem(*, idle_first: bool) -> PinchProblem:
    """A hot stream to cool and a cold stream that is off 10 % of the time."""
    ids = ["idle", "run"] if idle_first else ["run", "idle"]
    weights = [0.1, 0.9] if idle_first else [0.9, 0.1]
    cold_heat = [0.0, 100.0] if idle_first else [100.0, 0.0]
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 200.0, 120.0, 160.0),
                _stream("C1", 40.0, 90.0, cold_heat),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 10.0, 15.0, 5.0),
            ],
            "options": {
                "PROBLEM_PERIOD_IDS": ids,
                "PROBLEM_PERIOD_WEIGHTS": weights,
            },
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "A", "type": "Process Zone"}],
            },
        }
    )


@pytest.mark.parametrize("idle_first", [True, False])
def test_stream_ends_weight_the_flowrate_over_the_periods(idle_first):
    problem = _problem(idle_first=idle_first)
    problem.target.direct_heat_integration()

    ends = _stream_ends(problem_to_solver_arrays(problem, 10.0))

    # C1 runs at 2 kW/K for 90 % of the time, whichever period is first.
    (cold,) = [value for name, value in ends.items() if name.endswith("C1")]
    assert cold[2] == pytest.approx(0.9 * 2.0, rel=1e-9)
    assert cold[0] == pytest.approx(40.0 + 273.15, abs=1e-9)
    assert cold[1] == pytest.approx(90.0 + 273.15, abs=1e-9)


EXISTING = {
    "recovery": [],
    "heaters": [{"stream": "C1", "utility": "HPS", "area": 20.0, "id": "E1"}],
    "coolers": [{"stream": "H1", "utility": "CW", "area": 40.0, "id": "E2"}],
    "stage_count": 1,
}


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.parametrize("idle_first", [True, False])
def test_a_retrofit_finds_the_match_whichever_period_is_first(idle_first):
    result = _problem(idle_first=idle_first).design.retrofit(
        EXISTING,
        max_modifications=1,
        new_unit_cost=0.0,
        min_approach_temperature=10.0,
    )

    # A new match takes the heater's duty in the period the cold stream runs.
    assert [m.kind for m in result.selected.modifications] == ["new_match"]


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.parametrize("idle_first", [True, False])
def test_utility_relief_weights_the_networks_duties_over_the_periods(idle_first):
    problem = _problem(idle_first=idle_first)
    problem.target.direct_heat_integration()
    rated = run_retrofit_search(
        problem,
        problem.design._network_from_structure(EXISTING),
        RetrofitSettings(max_modifications=0),
    )
    network = rated.frontier[0].network
    weights = [0.1, 0.9] if idle_first else [0.9, 0.1]

    relief = utility_relief(
        network, NewMatch("A.H1", "A.C1", 1, new_stage=False), 1000.0, weights=weights
    )

    # The heater runs at 100 kW for 90 % of the time; the cooler always runs.
    assert relief == pytest.approx(90.0, abs=1e-3)


@pytest.mark.synthesis
@pytest.mark.solver
def test_the_search_screens_new_matches_with_the_period_weights(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    seen = []
    screen = search.screen_new_matches

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights"))
        return screen(*args, **kwargs)

    monkeypatch.setattr(search, "screen_new_matches", spy)

    _problem(idle_first=True).design.retrofit(
        EXISTING, max_modifications=1, new_unit_cost=0.0, min_approach_temperature=10.0
    )

    assert seen
    assert all(weights == pytest.approx([0.1, 0.9]) for weights in seen)


@pytest.mark.synthesis
@pytest.mark.solver
def test_the_search_scores_relocations_with_the_period_weights(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    seen = []
    potential = search.match_potential

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights"))
        return potential(*args, **kwargs)

    monkeypatch.setattr(search, "match_potential", spy)
    existing = {
        "recovery": [{"hot": "H1", "cold": "C1", "stage": 1, "area": 5.0, "id": "E3"}],
        "heaters": EXISTING["heaters"],
        "coolers": EXISTING["coolers"],
        "stage_count": 1,
    }

    _problem(idle_first=True).design.retrofit(
        existing,
        max_modifications=1,
        new_unit_cost=0.0,
        min_approach_temperature=10.0,
        relocation_cost=1.0,
    )

    # Relocation scoring is the search's own call to ``match_potential``.
    assert seen
    assert all(weights == pytest.approx([0.1, 0.9]) for weights in seen)


@pytest.mark.synthesis
@pytest.mark.solver
def test_the_heat_pump_screen_weights_the_networks_duties_over_the_periods():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        HeatPumpCandidate,
        screen_heat_pumps,
    )

    problem = _problem(idle_first=True)
    problem.target.direct_heat_integration()
    network = (
        run_retrofit_search(
            problem,
            problem.design._network_from_structure(EXISTING),
            RetrofitSettings(max_modifications=0),
        )
        .frontier[0]
        .network
    )
    candidate = HeatPumpCandidate("A.H1", "A.C1", 383.15, 373.15, cop=3.0)

    # The heater is off in the first period, so a first-period reading finds
    # nothing to relieve; over the periods it runs 90 % of the time.
    kept = screen_heat_pumps(network, [candidate], limit=1, weights=[0.1, 0.9])

    assert kept == [candidate]


@pytest.mark.synthesis
@pytest.mark.solver
def test_the_search_screens_heat_pumps_with_the_period_weights(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    seen = []
    screen = search.screen_heat_pumps

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights"))
        return screen(*args, **kwargs)

    monkeypatch.setattr(search, "screen_heat_pumps", spy)

    _problem(idle_first=True).design.retrofit(
        EXISTING,
        max_modifications=1,
        new_unit_cost=0.0,
        min_approach_temperature=10.0,
        heat_pump_cycle="carnot",
    )

    assert seen
    assert all(weights == pytest.approx([0.1, 0.9]) for weights in seen)
