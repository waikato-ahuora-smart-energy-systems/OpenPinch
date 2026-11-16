"""Heat exchanger network retrofit on the Ciric and Floudas (1989) case.

Case data (Example 1 of Hallberg and Scribner, 2009, "Heat Exchanger Network
Retrofit Comparison", University of Oklahoma, adapted from Ciric and Floudas,
1989): three hot and two cold streams, a 500 C hot utility at $0.0113/MJ and
20-40 C cooling water at $0.00238/MJ, 350 days a year. The existing network
has seven exchangers; its hot utility is 17,759 kW and cold utility 15,510 kW.
"""

from __future__ import annotations

import functools
import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import REPOSITORY_ROOT

CASE_JSON = (
    REPOSITORY_ROOT / "tests/fixtures/retrofit/Five-stream-Ciric-and-Floudas-1989.json"
)

# Existing exchangers (Figure 1 and Table 2 of the report): E1 H3-C2 and
# E3 H1-C1 in stage 1, E4 H2-C2 and E2 H3-C1 in stage 2, coolers E5 on H1 and
# E6 on H2, heater E7 on C2. Areas in m2.
EXISTING_AREAS = {
    "E1": 609.7,
    "E2": 579.2,
    "E3": 1008.5,
    "E4": 117.96,
    "E5": 787.5,
    "E6": 104.6,
    "E7": 246.75,
}
REPORTED_HOT_UTILITY = 17_759.0
REPORTED_COLD_UTILITY = 15_510.0


def _live_solver_available() -> bool:
    try:
        require_synthesis_dependency("pyomo.environ", purpose="retrofit test")
        require_solver_binary("ipopt", purpose="retrofit test")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
_needs_live_solver = pytest.mark.skipif(
    "not _live_solver_available()",
    reason="needs the ipopt binary and the pyomo package",
)


def _case_problem() -> PinchProblem:
    return PinchProblem(json.loads(CASE_JSON.read_text(encoding="utf-8")))


def _existing_network() -> HeatExchangerNetwork:
    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H3", "C2", 1), ("H1", "C1", 1), ("H2", "C2", 2), ("H3", "C1", 2)],
        heaters=["C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    names = {
        "recovery:H3->C2:S1": "E1",
        "recovery:H3->C1:S2": "E2",
        "recovery:H1->C1:S1": "E3",
        "recovery:H2->C2:S2": "E4",
        "cold-utility:H1->CU": "E5",
        "cold-utility:H2->CU": "E6",
        "hot-utility:HU->C2": "E7",
    }
    return structure.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(
                    update={
                        "exchanger_id": names[exchanger.exchanger_id],
                        "area": EXISTING_AREAS[names[exchanger.exchanger_id]],
                    }
                )
                for exchanger in structure.exchangers
            )
        }
    )


def _retrofit_model(problem, network, **spec_changes):
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
        build_duty_optimisation_request,
        fixed_network_structure,
    )
    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        FixedStructureModel,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )

    problem.target.direct_heat_integration()
    arrays = problem_to_solver_arrays(problem, 10.0)
    request = build_duty_optimisation_request(network, objective="cost")
    structure = fixed_network_structure(request, arrays.axis_maps)
    model = FixedStructureModel(
        name="retrofit",
        solver="ipopt-pyomo",
        solver_arrays=arrays,
        spec=replace(structure.spec, **spec_changes),
        minimisation_goal="retrofit",
        default_approach=1.0,
    )
    model.optimise(print_output=False)
    return structure, model


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_goal_rates_the_existing_network_within_the_area_tolerance():
    problem = _case_problem()
    structure, model = _retrofit_model(
        problem, _existing_network(), area_tolerance=0.02
    )

    assert model.mSuccess == 1
    hot = sum(
        result.periods[0].duty
        for (kind, index), result in zip(structure.slots, structure.results(model))
        if kind == "utility" and structure.spec.utilities[index].side == "hot"
    )
    # Reported areas are rounded and the solver uses Chen's LMTD, so the
    # existing network rates within 2 % of the reported hot utility.
    assert hot == pytest.approx(REPORTED_HOT_UTILITY, rel=0.02)
    # Existing units keep their installed areas when none may be enlarged.
    areas = [result.area for result in structure.results(model)]
    assert areas == pytest.approx(
        [EXISTING_AREAS[e.exchanger_id] for e in _existing_network().exchangers]
    )


def _with_new_match(network, hot, cold, stage, exchanger_id):
    extra = HeatExchangerNetwork.from_structure(recovery=[(hot, cold, stage)])
    (new,) = extra.exchangers
    return network.model_copy(
        update={
            "exchangers": network.exchangers
            + (new.model_copy(update={"exchanger_id": exchanger_id}),)
        }
    )


def _hot_utility(structure, model) -> float:
    return sum(
        result.periods[0].duty
        for (kind, index), result in zip(structure.slots, structure.results(model))
        if kind == "utility" and structure.spec.utilities[index].side == "hot"
    )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_goal_sizes_a_new_match_and_charges_only_its_area():
    problem = _case_problem()
    base_structure, base = _retrofit_model(
        problem, _existing_network(), area_tolerance=0.02
    )
    network = _with_new_match(_existing_network(), "H1", "C2", 3, "N1")
    structure, model = _retrofit_model(problem, network, area_tolerance=0.02)

    assert model.mSuccess == 1
    # H3 has no cooler and E2 stays at its installed area, so the new match
    # recovers only some hundreds of kW.
    assert _hot_utility(structure, model) < _hot_utility(base_structure, base) - 100.0
    results = structure.results(model)
    new = results[-1]
    assert new.area > 0.0
    assert new.added_area == pytest.approx(new.area)
    assert [r.added_area for r in results[:-1]] == [0.0] * (len(results) - 1)
    # Area cost law exponent 1: the retrofit capital is the annualised
    # coefficient times the added area.
    assert model.retrofit_capital_value == pytest.approx(
        float(model.A_coeff[0]) * new.area, rel=1e-6
    )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_goal_lets_an_enlargeable_unit_grow_at_a_cost():
    problem = _case_problem()
    base_structure, base = _retrofit_model(
        problem, _existing_network(), area_tolerance=0.02
    )
    # E1 (H3-C2) is the first recovery match.
    structure, model = _retrofit_model(
        problem,
        _existing_network(),
        area_tolerance=0.02,
        enlargeable_recovery=frozenset({0}),
    )

    assert model.mSuccess == 1
    e1 = structure.results(model)[0]
    assert e1.added_area > 1.0
    assert e1.area == pytest.approx(EXISTING_AREAS["E1"] + e1.added_area)
    assert _hot_utility(structure, model) < _hot_utility(base_structure, base)
    assert model.retrofit_capital_value == pytest.approx(
        float(model.A_coeff[0]) * e1.added_area, rel=1e-6
    )


# Report costs (pages 13-14): a new exchanger $34,600 plus $857/m2, added area
# $8,650 per unit (existing shell) plus $857/m2, a split $10,000. The fixture
# sets the area law ($857/m2, exponent 1), the new-unit cost and a 5-year life
# with no discount, so the capital recovery factor is 0.2 per year.
REPORT_COSTS = {
    "new_unit_cost": 34_600.0,
    "area_addition_cost": 8_650.0,
    "split_cost": 10_000.0,
}


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_search_finds_one_best_design_per_modification_count():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    problem = _case_problem()
    result = run_retrofit_search(
        problem,
        _existing_network(),
        RetrofitSettings(
            max_modifications=2,
            beam_width=2,
            new_match_candidates=3,
            area_tolerance=0.02,
            **REPORT_COSTS,
        ),
    )

    base = result.frontier[0]
    assert base.modifications == ()
    assert base.hot_utility == pytest.approx(REPORTED_HOT_UTILITY, rel=0.02)
    assert base.annual_savings == 0.0
    assert [d.modification_count for d in result.frontier] == [0, 1, 2]
    for design in result.frontier[1:]:
        assert design.hot_utility < base.hot_utility
        assert design.annual_savings > 0.0
        fixed = sum(m.fixed_cost for m in design.modifications)
        area = sum(m.area_cost for m in design.modifications)
        assert design.capital_cost == pytest.approx(fixed + area)
        assert design.annualised_capital_cost == pytest.approx(
            0.2 * design.capital_cost
        )
        assert design.net_annual_benefit == pytest.approx(
            design.annual_savings - design.annualised_capital_cost
        )
        assert design.payback_years == pytest.approx(
            design.capital_cost / design.annual_savings
        )
    assert result.selected.net_annual_benefit > 0.0


def test_retrofit_search_needs_the_installed_area_of_every_existing_unit():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    network = _existing_network()
    network = network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": None}) if e.exchanger_id == "E4" else e
                for e in network.exchangers
            )
        }
    )

    with pytest.raises(ValueError, match="installed area.*E4"):
        run_retrofit_search(_case_problem(), network, RetrofitSettings())


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_search_rejects_an_existing_network_short_of_area():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # H3 has no cooler: with half its area E2 cannot bring H3 to its target.
    network = _existing_network()
    network = network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": 0.5 * e.area})
                if e.exchanger_id == "E2"
                else e
                for e in network.exchangers
            )
        }
    )

    with pytest.raises(ValueError, match="existing network.*installed areas"):
        run_retrofit_search(_case_problem(), network, RetrofitSettings())


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_every_frontier_modification_does_something():
    # The deep, narrow search runs out of useful moves before its last depth.
    for design in _deep_search().frontier[1:]:
        duties = {
            e.exchanger_id: max(s.duty for s in e.period_states)
            for e in design.network.exchangers
        }
        for modification in design.modifications:
            assert modification.added_area >= 1e-3
            assert duties[modification.exchanger_id] > 0.0


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_frontier_marks_designs_no_better_than_fewer_modifications():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # Costly added area: one modification loses money against the existing
    # network, and a fourth adds less than it costs.
    result = run_retrofit_search(
        _case_problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=4,
            beam_width=2,
            new_match_candidates=4,
            **(REPORT_COSTS | {"area_addition_cost": 2e6}),
        ),
    )

    expected = [
        any(
            earlier.net_annual_benefit >= design.net_annual_benefit
            for earlier in result.frontier[:n]
        )
        for n, design in enumerate(result.frontier)
    ]
    assert any(expected)
    assert [d.dominated for d in result.frontier] == expected


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_result_counts_the_designs_the_search_solved():
    result = _deep_search()

    # The base, plus at least one child per frontier depth.
    assert result.evaluated_designs >= len(result.frontier)


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_a_new_match_reports_its_stage_in_the_final_network():
    result = _case_problem().design.retrofit(
        EXISTING_STRUCTURE,
        max_modifications=4,
        area_addition_cost=8_650.0,
        split_cost=10_000.0,
    )

    checked = 0
    for design in result.frontier:
        stages = {e.exchanger_id: e.stage for e in design.network.exchangers}
        for modification in design.modifications:
            if modification.kind == "new_match":
                assert modification.stage == stages[modification.exchanger_id]
                checked += 1
    assert checked


@functools.cache
def _deep_search():
    """A deep, narrow search: six modifications, a beam of one."""

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    return run_retrofit_search(
        _case_problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=6, beam_width=1, new_match_candidates=2, **REPORT_COSTS
        ),
    )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_new_matches_use_the_existing_network_stream_names():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # Added area priced out, so the frontier is made of new matches.
    result = run_retrofit_search(
        _case_problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=2,
            beam_width=2,
            new_match_candidates=4,
            **(REPORT_COSTS | {"area_addition_cost": 1e9}),
        ),
    )
    new_matches = [
        m
        for design in result.frontier
        for m in design.modifications
        if m.kind == "new_match"
    ]

    assert new_matches
    for match in new_matches:
        assert match.hot_stream in {"H1", "H2", "H3"}
        assert match.cold_stream in {"C1", "C2"}
        assert match.exchanger_id.startswith(
            f"new:{match.hot_stream}->{match.cold_stream}"
        )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_search_prices_each_modification_with_the_report_costs():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    result = run_retrofit_search(
        _case_problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=1, beam_width=8, new_match_candidates=3, **REPORT_COSTS
        ),
    )

    (design,) = result.frontier[1:]
    (modification,) = design.modifications
    assert modification.added_area > 0.0
    assert modification.area_cost == pytest.approx(857.0 * modification.added_area)
    if modification.kind == "new_match":
        assert modification.fixed_cost == pytest.approx(
            34_600.0 + 10_000.0 * len(modification.new_splits)
        )
    else:
        assert modification.fixed_cost == pytest.approx(8_650.0)
    assert design.capital_cost == pytest.approx(
        modification.fixed_cost + modification.area_cost
    )


# The existing network as a mapping with installed areas and plant tags.
EXISTING_STRUCTURE = {
    "recovery": [
        {"hot": "H3", "cold": "C2", "stage": 1, "area": 609.7, "id": "E1"},
        {"hot": "H3", "cold": "C1", "stage": 2, "area": 579.2, "id": "E2"},
        {"hot": "H1", "cold": "C1", "stage": 1, "area": 1008.5, "id": "E3"},
        {"hot": "H2", "cold": "C2", "stage": 2, "area": 117.96, "id": "E4"},
    ],
    "heaters": [{"stream": "C2", "area": 246.75, "id": "E7"}],
    "coolers": [
        {"stream": "H1", "area": 787.5, "id": "E5"},
        {"stream": "H2", "area": 104.6, "id": "E6"},
    ],
}


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_design_retrofit_takes_the_existing_network_as_a_mapping():
    result = _case_problem().design.retrofit(
        EXISTING_STRUCTURE,
        max_modifications=1,
        beam_width=4,
        new_match_candidates=2,
        area_addition_cost=8_650.0,
        split_cost=10_000.0,
    )

    base = result.frontier[0]
    assert base.hot_utility == pytest.approx(REPORTED_HOT_UTILITY, rel=0.02)
    assert {e.exchanger_id for e in base.network.exchangers} == set(EXISTING_AREAS)
    assert {e.exchanger_id: e.area for e in base.network.exchangers} == (
        pytest.approx(EXISTING_AREAS)
    )
    (modification,) = result.frontier[1].modifications
    if modification.kind == "new_match":
        # The new-unit cost defaults to COSTING_HX_UNIT_COST ($34,600).
        assert modification.fixed_cost == pytest.approx(
            34_600.0 + 10_000.0 * len(modification.new_splits)
        )
    else:
        assert modification.exchanger_id in EXISTING_AREAS
        assert modification.fixed_cost == pytest.approx(8_650.0)
