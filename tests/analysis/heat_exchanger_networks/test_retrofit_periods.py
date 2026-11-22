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


@pytest.mark.parametrize("idle_first", [True, False])
def test_period_stream_ends_give_each_periods_own_flowrate(idle_first):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        _period_stream_ends,
    )

    problem = _problem(idle_first=idle_first)
    problem.target.direct_heat_integration()

    periods = _period_stream_ends(problem_to_solver_arrays(problem, 10.0))

    # C1 runs at 2 kW/K in the "run" period and is off in the "idle" one.
    idle, run = (0, 1) if idle_first else (1, 0)
    cold = [
        next(value for name, value in ends.items() if name.endswith("C1"))
        for ends in periods
    ]
    assert cold[idle][2] == 0.0
    assert cold[run] == pytest.approx((40.0 + 273.15, 90.0 + 273.15, 2.0), rel=1e-9)
    hot = [
        next(value for name, value in ends.items() if name.endswith("H1"))
        for ends in periods
    ]
    assert hot[idle] == hot[run] == pytest.approx((473.15, 393.15, 2.0), rel=1e-9)


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


@pytest.mark.synthesis
@pytest.mark.solver
def test_the_loop_screen_weights_the_networks_duties_over_the_periods():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        HeatRecoveryLoop,
        screen_heat_recovery_loops,
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
    loop = HeatRecoveryLoop("A.H1", "A.C1", 378.15)

    # The heater the loop would relieve is off in the first period.
    kept = screen_heat_recovery_loops(network, [loop], limit=1, weights=[0.1, 0.9])

    assert kept == [loop]


@pytest.mark.synthesis
@pytest.mark.solver
def test_the_search_screens_loops_with_the_period_weights(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    seen = []
    screen = search.screen_heat_recovery_loops

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights"))
        return screen(*args, **kwargs)

    monkeypatch.setattr(search, "screen_heat_recovery_loops", spy)
    # The same two streams in two processes, so only a loop can link them.
    source = _problem(idle_first=True).to_problem_json()
    for stream in source["streams"]:
        if stream["name"] == "C1":
            stream["zone"] = "Site/B"
    source["zone_tree"]["children"].append({"name": "B", "type": "Process Zone"})
    existing = {
        "recovery": [],
        "heaters": [{"stream": "B.C1", "utility": "HPS", "area": 20.0, "id": "E1"}],
        "coolers": [{"stream": "A.H1", "utility": "CW", "area": 40.0, "id": "E2"}],
        "stage_count": 1,
    }

    with pytest.warns(UserWarning):
        PinchProblem(source=source).design.retrofit(
            existing,
            max_modifications=1,
            new_unit_cost=0.0,
            min_approach_temperature=10.0,
        )

    assert seen
    assert all(weights == pytest.approx([0.1, 0.9]) for weights in seen)


def _two_period_network(duties, heaters=("C1",), inlets=None):
    """A cooler on H1 and a heater on C1 (or on each of ``heaters``) with the
    given duty in each period and, with ``inlets``, the process stream's
    temperature (K) entering it in each period (``None`` when it is off)."""
    from OpenPinch.domain._heat_exchanger.period_state import (
        HeatExchangerPeriodState,
    )
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    structure = HeatExchangerNetwork.from_structure(
        recovery=[],
        heaters=list(heaters),
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )
    return structure.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(
                    update={
                        "period_states": tuple(
                            HeatExchangerPeriodState(
                                period_id=str(index),
                                period_idx=index,
                                duty=duty,
                                **_process_inlet(e, inlets, index),
                            )
                            for index, duty in enumerate(
                                duties[f"{e.source_stream}->{e.sink_stream}"]
                            )
                        )
                    }
                )
                for e in structure.exchangers
            )
        }
    )


def _process_inlet(exchanger, inlets, index):
    """The process side's inlet temperature of a cooler or heater."""
    if inlets is None:
        return {}
    field = (
        "source_inlet_temperature"
        if exchanger.source_stream == "H1"
        else "sink_inlet_temperature"
    )
    key = f"{exchanger.source_stream}->{exchanger.sink_stream}"
    return {field: inlets[key][index]}


def test_a_match_relieves_nothing_when_its_streams_never_run_together():
    # H1 is cooled only in the first period and C1 heated only in the second,
    # so a match between them can move no heat in either.
    network = _two_period_network({"H1->CU": [100.0, 0.0], "HU->C1": [0.0, 100.0]})

    relief = utility_relief(
        network, NewMatch("H1", "C1", 1, new_stage=False), 1000.0, weights=[0.5, 0.5]
    )

    assert relief == 0.0


def test_a_heat_pump_is_left_out_when_its_streams_never_run_together():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        HeatPumpCandidate,
        screen_heat_pumps,
    )

    network = _two_period_network({"H1->CU": [100.0, 0.0], "HU->C1": [0.0, 100.0]})
    candidate = HeatPumpCandidate("H1", "C1", 325.0, 365.0, cop=4.0)

    kept = screen_heat_pumps(network, [candidate], limit=1, weights=[0.5, 0.5])

    assert kept == []


def test_a_loop_is_left_out_when_its_streams_never_run_together():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        HeatRecoveryLoop,
        screen_heat_recovery_loops,
    )

    network = _two_period_network({"H1->CU": [100.0, 0.0], "HU->C1": [0.0, 100.0]})
    loop = HeatRecoveryLoop("H1", "C1", 380.0)

    kept = screen_heat_recovery_loops(network, [loop], limit=1, weights=[0.5, 0.5])

    assert kept == []


# H1 (400 to 350 K) runs at 2 kW/K in the first period only and C1 (300 to
# 340 K) at 2 kW/K in the second only: on average each runs at 1 kW/K.
_NEVER_TOGETHER_ENDS = {"H1": (400.0, 350.0, 1.0), "C1": (300.0, 340.0, 1.0)}
_NEVER_TOGETHER_PERIOD_ENDS = (
    {"H1": (400.0, 350.0, 2.0), "C1": (300.0, 340.0, 0.0)},
    {"H1": (400.0, 350.0, 0.0), "C1": (300.0, 340.0, 2.0)},
)


def test_a_match_has_no_potential_when_its_streams_never_run_together():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
        match_potential,
    )

    network = _two_period_network(
        {"H1->CU": [100.0, 0.0], "HU->C1": [0.0, 80.0]},
        inlets={"H1->CU": [400.0, None], "HU->C1": [None, 300.0]},
    )

    # The period means give both streams 1 kW/K and so a potential; period
    # by period the smaller flowrate is 0.
    potential = match_potential(
        network,
        _NEVER_TOGETHER_ENDS,
        NewMatch("H1", "C1", 1, new_stage=True),
        approach=10.0,
        weights=[0.5, 0.5],
        period_ends=_NEVER_TOGETHER_PERIOD_ENDS,
    )

    assert potential == 0.0


def test_the_screen_proposes_no_match_between_streams_that_never_run_together():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
        screen_new_matches,
    )

    network = _two_period_network(
        {"H1->CU": [100.0, 0.0], "HU->C1": [0.0, 80.0]},
        inlets={"H1->CU": [400.0, None], "HU->C1": [None, 300.0]},
    )

    proposed = screen_new_matches(
        network,
        _NEVER_TOGETHER_ENDS,
        approach=10.0,
        limit=5,
        weights=[0.5, 0.5],
        period_ends=_NEVER_TOGETHER_PERIOD_ENDS,
    )

    assert proposed == ()


def test_the_screen_caps_relief_by_each_periods_potential():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
        screen_new_matches,
    )

    # Every stream runs at 2 kW/K. H1 (400 to 350 K) is cooled 100 kW in both
    # periods. C2 (300 to 325 K) is heated 50 kW in both and could take
    # 2 * (400 - 300 - 10) = 180 kW from H1 in each, so a match relieves
    # 50 kW. C1 is heated 80 kW then 50 kW, but in the second period it runs
    # from 395 to 420 K, too close to H1: its potential is 180 kW in the first
    # period and none in the second, so a match relieves min(180, 100, 80) =
    # 80 kW then nothing, 40 kW on average, and C2 leads. Capped by its mean
    # potential (90 kW) in both periods, C1 looked like 65 kW and led.
    network = _two_period_network(
        {"H1->CU": [100.0, 100.0], "HU->C1": [80.0, 50.0], "HU->C2": [50.0, 50.0]},
        heaters=("C1", "C2"),
        inlets={
            "H1->CU": [400.0, 400.0],
            "HU->C1": [300.0, 395.0],
            "HU->C2": [300.0, 300.0],
        },
    )
    ends = {
        "H1": (400.0, 350.0, 2.0),
        "C1": (347.5, 380.0, 2.0),
        "C2": (300.0, 325.0, 2.0),
    }
    period_ends = (
        {**ends, "C1": (300.0, 340.0, 2.0)},
        {**ends, "C1": (395.0, 420.0, 2.0)},
    )

    proposed = screen_new_matches(
        network,
        ends,
        approach=10.0,
        limit=1,
        weights=[0.5, 0.5],
        period_ends=period_ends,
    )

    assert [move.cold for move in proposed] == ["C2"]
