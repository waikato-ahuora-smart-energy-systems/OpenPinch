"""Heat recovery loops for a multi-process retrofit (no solver but the
SciPy stand-in, except the last test).

A loop is an intermediate fluid at one temperature: a hot stream of one
process heats it in a loop cooler at the stream's end, and a cold stream of
another process takes the heat in a loop heater at its end. It is the only
link between processes that a site retrofit proposes.
"""

from __future__ import annotations

import pytest

from tests.application.test_site_heat_exchanger_network import (
    _solvers_available,
    _stream,
    _utility,
)


def test_a_loop_runs_midway_between_its_targets_keeping_the_approach_each_side():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        LOOP_SPAN,
        HeatRecoveryLoop,
        heat_recovery_loop_candidates,
    )

    # Supply, target (K) and heat capacity flowrate (kW/K).
    ends = {"H1": (473.0, 393.0, 2.0), "C1": (313.0, 363.0, 2.0)}
    processes = {"H1": "Site/A", "C1": "Site/B"}

    (loop,) = heat_recovery_loop_candidates(ends, processes, approach=10.0)

    assert loop == HeatRecoveryLoop("H1", "C1", 378.0)
    # The loop circulates over LOOP_SPAN about its temperature; the hot
    # stream at its target still clears the loop's warm end, and the cold
    # stream at its target stays below its cool end, by the approach.
    assert LOOP_SPAN == pytest.approx(1.0)
    assert 393.0 - (loop.temperature + LOOP_SPAN / 2) >= 10.0
    assert (loop.temperature - LOOP_SPAN / 2) - 363.0 >= 10.0


def test_no_loop_within_one_process_or_between_targets_too_close_for_the_approach():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        heat_recovery_loop_candidates,
    )

    ends = {
        "H1": (473.0, 393.0, 2.0),
        "H2": (420.0, 384.0, 1.0),
        "C1": (313.0, 363.0, 2.0),
        "C2": (300.0, 330.0, 1.0),
        "H9": (450.0, 400.0, 1.0),
    }
    # H2 and C1 are 21 K apart: 2 x 10 K and the 1 K span need 21 K.
    # C2 shares H1's process; H9 is outside every process.
    processes = {"H1": "Site/A", "H2": "Site/A", "C1": "Site/B", "C2": "Site/A"}

    loops = heat_recovery_loop_candidates(ends, processes, approach=10.0)

    assert [(loop.hot, loop.cold) for loop in loops] == [
        ("H1", "C1"),
        ("H2", "C1"),
    ]
    assert heat_recovery_loop_candidates(ends, processes, approach=10.5) == [loops[0]]


def _network(duties):
    from OpenPinch.domain._heat_exchanger.period_state import (
        HeatExchangerPeriodState,
    )
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    structure = HeatExchangerNetwork.from_structure(
        recovery=[],
        heaters=["C1", "C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    return structure.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(
                    update={
                        "period_states": (
                            HeatExchangerPeriodState(
                                period_id="0",
                                period_idx=0,
                                duty=duties[f"{e.source_stream}->{e.sink_stream}"],
                            ),
                        )
                    }
                )
                for e in structure.exchangers
            )
        }
    )


def test_the_screen_ranks_loops_by_the_utility_they_could_relieve():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        HeatRecoveryLoop,
        screen_heat_recovery_loops,
    )

    network = _network({"H1->CU": 90.0, "H2->CU": 0.0, "HU->C1": 200.0, "HU->C2": 30.0})
    to_c1 = HeatRecoveryLoop("H1", "C1", 380.0)
    to_c2 = HeatRecoveryLoop("H1", "C2", 380.0)
    from_h2 = HeatRecoveryLoop("H2", "C1", 380.0)

    # A loop relieves at most its hot stream's cooler duty and its cold
    # stream's heater duty: 90 kW to C1, 30 kW to C2, none from H2.
    candidates = [to_c2, from_h2, to_c1]
    assert screen_heat_recovery_loops(network, candidates, limit=5) == [to_c1, to_c2]
    assert screen_heat_recovery_loops(network, candidates, limit=1) == [to_c1]


def test_loop_utilities_join_the_solver_arrays_free_at_the_loop_temperature():
    import json

    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        HeatRecoveryLoop,
        loop_cooler_name,
        loop_heater_name,
        with_loop_utilities,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )
    from OpenPinch.application.problem import PinchProblem
    from OpenPinch.resources import read_sample_case

    problem = PinchProblem(
        json.loads(read_sample_case("Five-stream-Ciric-and-Floudas-1989.json"))
    )
    ensure_target_results(problem, {}, None)
    arrays = problem_to_solver_arrays(problem, 10.0)
    loop = HeatRecoveryLoop("H1", "C1", 380.0)

    extended = with_loop_utilities(arrays, [loop])

    data = extended.arrays
    axes = extended.axis_maps
    heater = axes["hot_utilities"][loop_heater_name(loop)]
    cooler = axes["cold_utilities"][loop_cooler_name(loop)]
    assert (heater, cooler) == (1, 1)
    # The loop gives heat from its warm end to its cool end and takes it
    # back the other way, over 1 K about its temperature, at no price.
    assert data["T_hu_in_period"][0][heater] == pytest.approx(380.5)
    assert data["T_hu_out_period"][0][heater] == pytest.approx(379.5)
    assert data["T_cu_in_period"][0][cooler] == pytest.approx(379.5)
    assert data["T_cu_out_period"][0][cooler] == pytest.approx(380.5)
    assert data["hu_cost_period"][0][heater] == pytest.approx(0.0)
    assert data["cu_cost_period"][0][cooler] == pytest.approx(0.0)
    assert len(arrays.axis_maps["hot_utilities"]) == 1


def test_a_loop_delivers_the_heat_it_collects():
    """H1 (process A) and C1 (process B) share no recovery match. Unlinked,
    the two free loop utilities would each take what they can (160 kW from
    H1, 100 kW to C1); linked, the loop heater delivers exactly what the loop
    cooler collects. (The SciPy stand-in stalls short of the 100 kW optimum
    that Ipopt reaches; the last test checks that one.)"""
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )
    from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
        build_duty_optimisation_request,
        fixed_network_structure,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        HeatRecoveryLoop,
        loop_cooler_name,
        loop_heater_name,
        with_loop_utilities,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
    from tests.support.scipy_nlp import scipy_fixed_structure_model

    problem = _two_processes()
    ensure_target_results(problem, {}, None)
    arrays = problem_to_solver_arrays(problem, 10.0)
    loop = HeatRecoveryLoop("A.H1", "B.C1", 378.15)
    arrays = with_loop_utilities(arrays, [loop])
    names = {
        name.rsplit(".", 1)[-1]: name
        for axis in ("hot_utilities", "cold_utilities")
        for name in arrays.axis_maps[axis]
    }
    network = HeatExchangerNetwork.from_structure(
        recovery=[],
        heaters=[("B.C1", names["HPS"]), ("B.C1", loop_heater_name(loop))],
        coolers=[("A.H1", names["CW"]), ("A.H1", loop_cooler_name(loop))],
    )
    request = build_duty_optimisation_request(
        network, objective="cost", min_approach_temperature=10.0
    )
    structure = fixed_network_structure(request, arrays.axis_maps)
    utilities = structure.spec.utilities

    def slot(side, index):
        return next(
            e for e, m in enumerate(utilities) if (m.side, m.utility) == (side, index)
        )

    cooler, heater, steam = slot("cold", 1), slot("hot", 1), slot("hot", 0)
    spec = replace(structure.spec, loops=((cooler, heater),))
    model = scipy_fixed_structure_model()(
        name="loop",
        solver="ipopt-pyomo",
        solver_arrays=arrays,
        spec=spec,
        minimisation_goal="utility cost",
        default_approach=10.0,
    )
    model.optimise()

    assert model.mSuccess == 1
    collected = model.utility_results[cooler].periods[0].duty
    delivered = model.utility_results[heater].periods[0].duty
    steam_duty = model.utility_results[steam].periods[0].duty
    assert delivered > 50.0
    assert delivered == pytest.approx(collected, rel=1e-6)
    assert steam_duty + delivered == pytest.approx(100.0, rel=1e-6)


def _two_processes():
    """Process A cools H1 (200 to 120 C, 160 kW) on cooling water; process
    B heats C1 (40 to 90 C, 100 kW) on steam. Neither process has a stream
    the other can use directly within itself."""
    from OpenPinch.application.problem import PinchProblem

    return PinchProblem(
        source={
            "streams": [
                _stream("A", "H1", 200.0, 120.0, 160.0),
                _stream("B", "C1", 40.0, 90.0, 100.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {"name": "A", "type": "Process Zone"},
                    {"name": "B", "type": "Process Zone"},
                ],
            },
        }
    )


TWO_PROCESS_EXISTING = {
    "recovery": [],
    "heaters": [{"stream": "B.C1", "utility": "HPS", "area": 2.0, "id": "E1"}],
    "coolers": [{"stream": "A.H1", "utility": "CW", "area": 4.0, "id": "E2"}],
    "stage_count": 1,
}


@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_two_process_retrofit_links_the_processes_through_a_balanced_loop():
    """With Ipopt: the loop takes C1's whole heating from H1, so the
    retrofit pays; its two exchangers carry equal duty and keep the
    approach on both sides."""
    import warnings

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        LOOP_SPAN,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = _two_processes().design.retrofit(
            TWO_PROCESS_EXISTING,
            max_modifications=1,
            new_unit_cost=0.0,
            min_approach_temperature=10.0,
        )

    design = result.selected
    (loop,) = [m for m in design.modifications if m.kind == "heat_recovery_loop"]
    assert design.net_annual_benefit > result.frontier[0].net_annual_benefit
    assert (loop.hot_stream, loop.cold_stream) == ("A.H1", "B.C1")
    assert loop.loop_temperature == pytest.approx(105.0)
    units = {e.exchanger_id: e for e in design.network.exchangers}
    cooler = units[loop.exchanger_id].period_states[0]
    heater = units[loop.utility_exchanger_id].period_states[0]
    assert heater.duty == pytest.approx(100.0, rel=1e-3)
    assert heater.duty == pytest.approx(cooler.duty, rel=1e-4)
    assert loop.loop_duty == pytest.approx(heater.duty, rel=1e-6)
    # Steam falls by the loop's duty; the loop's own heat is not utility.
    assert design.hot_utility == pytest.approx(0.0, abs=1e-2)
    # The hot stream leaves the loop cooler at least the approach above
    # the loop's warm end... and the cold stream leaves the loop heater
    # at least the approach below its cool end.
    warm = loop.loop_temperature + 273.15 + LOOP_SPAN / 2
    cool = loop.loop_temperature + 273.15 - LOOP_SPAN / 2
    assert cooler.source_outlet_temperature - cool >= 10.0 - 1e-3
    assert warm - heater.sink_outlet_temperature >= 10.0 - 1e-3
