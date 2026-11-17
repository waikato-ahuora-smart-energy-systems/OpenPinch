"""Retrofit search mechanics on a fake fixed-structure solver (no solver).

The fake keeps the real request building, structure and screen, and stands
in for the NLP only: a design's utility cost falls by $100,000/y per
recovery match and $20,000/y per enlarged or enhanced unit, every installed unit needs
its whole widened area (so it is at its cap), and every new unit gets 25 m2.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.resources import read_sample_case

CASE = "Five-stream-Ciric-and-Floudas-1989.json"
COSTS = {"new_unit_cost": 34_600.0, "area_addition_cost": 8_650.0, "split_cost": 1e4}
# The case's area law ($857/m2, exponent 1) and capital recovery (0.2 /y).
AREA_COST = 857.0
RECOVERY_FACTOR = 0.2
NEW_UNIT_AREA = 25.0
ADDED_AREA = 10.0


def _existing_network() -> HeatExchangerNetwork:
    areas = {"E1": 609.7, "E2": 579.2, "E3": 1008.5, "E4": 117.96}
    areas |= {"E5": 787.5, "E6": 104.6, "E7": 246.75}
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
                e.model_copy(
                    update={
                        "exchanger_id": names[e.exchanger_id],
                        "area": areas[names[e.exchanger_id]],
                    }
                )
                for e in structure.exchangers
            )
        }
    )


class _Backend:
    """Switches a test can flip, and what the fake saw."""

    def __init__(self) -> None:
        self.fail = lambda spec: False
        self.reject = lambda network: ()
        self.cleaned = 0
        self.solved = 0
        self.goals: dict[str, set[str]] = {}
        self.area_values = lambda spec: None


@pytest.fixture
def fake_solver(monkeypatch) -> _Backend:
    from OpenPinch.analysis.heat_exchanger_networks import duty_optimisation
    from OpenPinch.analysis.heat_exchanger_networks.models import fixed_structure
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
    from OpenPinch.domain._heat_exchanger.period_state import (
        HeatExchangerPeriodState,
    )

    backend = _Backend()
    ExchangerResult = fixed_structure.ExchangerResult
    ExchangerPeriodResult = fixed_structure.ExchangerPeriodResult

    class FakeModel:
        def __init__(self, *, spec, name, minimisation_goal, **_settings) -> None:
            self.spec = spec
            backend.goals.setdefault(name, set()).add(minimisation_goal)

        def optimise(self, print_output: bool) -> None:
            backend.solved += 1
            spec = self.spec
            enlarged = len(spec.enlargeable_recovery) + len(spec.enlargeable_utilities)
            enhanced = len(spec.enhanced_recovery) + len(spec.enhanced_utilities)
            self.utility_cost_value = (
                1_000_000.0
                - 100_000.0 * len(spec.recovery)
                - 20_000.0 * (enlarged + enhanced)
            )
            self.area_values = backend.area_values(spec)
            self.mSuccess = 0 if backend.fail(spec) else 1

        def cleanup(self) -> None:
            backend.cleaned += 1

    def results(structure, model) -> list:
        spec = structure.spec
        out = []
        for kind, index in structure.slots:
            recovery = kind == "recovery"
            installed = (spec.recovery_areas if recovery else spec.utility_areas).get(
                index
            )
            enlargeable = (
                spec.enlargeable_recovery if recovery else spec.enlargeable_utilities
            )
            if installed is None:
                area = added = NEW_UNIT_AREA
            else:
                added = ADDED_AREA if index in enlargeable else 0.0
                area = installed + added
            period = ExchangerPeriodResult(
                duty=100.0,
                active=True,
                approach=(10.0, 10.0),
                source_inlet=400.0,
                source_outlet=300.0,
                sink_inlet=50.0,
                sink_outlet=150.0,
                required_area=area * (1.0 + spec.area_tolerance),
            )
            out.append(
                ExchangerResult(
                    periods=[period], area=area, capital_cost=0.0, added_area=added
                )
            )
        return out

    def solution(request, structure, model, arrays) -> HeatExchangerNetwork:
        state = HeatExchangerPeriodState(
            period_id="0",
            period_idx=0,
            duty=100.0,
            active=True,
            source_inlet_temperature=400.0,
            source_outlet_temperature=300.0,
            sink_inlet_temperature=50.0,
            sink_outlet_temperature=150.0,
        )
        network = request.network
        return network.model_copy(
            update={
                "exchangers": tuple(
                    e.model_copy(update={"period_states": (state,)})
                    for e in network.exchangers
                ),
                "summary_metrics": {
                    "hot_utility_load": model.utility_cost_value / 100.0,
                    "cold_utility_load": model.utility_cost_value / 200.0,
                },
            }
        )

    monkeypatch.setattr(fixed_structure, "FixedStructureModel", FakeModel)
    monkeypatch.setattr(duty_optimisation.FixedNetworkStructure, "results", results)
    monkeypatch.setattr(search, "build_solution_network", solution)
    monkeypatch.setattr(
        search,
        "verify_solution",
        lambda structure, model, network: backend.reject(network),
    )
    return backend


def _problem() -> PinchProblem:
    return PinchProblem(json.loads(read_sample_case(CASE)))


def test_every_frontier_design_is_priced_from_its_modifications(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    result = run_retrofit_search(
        _problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=2,
            beam_width=2,
            new_match_candidates=2,
            relocation_cost=5_000.0,
            relocation_candidates=2,
            new_utility_units=True,
            **COSTS,
        ),
    )

    base = result.frontier[0]
    assert base.modifications == ()
    assert base.utility_cost == pytest.approx(1_000_000.0 - 4 * 100_000.0)
    assert len(result.frontier) == 3
    for design in result.frontier[1:]:
        capital = sum(m.fixed_cost + m.area_cost for m in design.modifications)
        assert design.capital_cost == pytest.approx(capital)
        assert design.annual_savings == pytest.approx(
            base.utility_cost - design.utility_cost
        )
        assert design.net_annual_benefit == pytest.approx(
            design.annual_savings - RECOVERY_FACTOR * capital
        )
        for m in design.modifications:
            expected_area = {
                "new_match": NEW_UNIT_AREA,
                "area_addition": ADDED_AREA,
                "relocation": 0.0,
            }[m.kind]
            assert m.added_area == pytest.approx(expected_area)
            assert m.area_cost == pytest.approx(
                AREA_COST * (m.added_area + m.utility_area)
            )
    assert fake_solver.cleaned == fake_solver.solved == result.evaluated_designs


def test_an_existing_network_that_does_not_solve_is_an_error(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    fake_solver.fail = lambda spec: True

    with pytest.raises(ValueError, match="does not meet its targets"):
        run_retrofit_search(_problem(), _existing_network(), RetrofitSettings())
    assert fake_solver.cleaned == fake_solver.solved == 1


def test_designs_that_fail_to_solve_or_verify_stay_off_the_frontier(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # A new match in a new stage does not solve; any other new unit fails
    # verification. Area additions remain.
    fake_solver.fail = lambda spec: spec.stage_count > 2
    fake_solver.reject = lambda network: tuple(
        e.exchanger_id for e in network.exchangers if e.exchanger_id.startswith("new:")
    )

    result = run_retrofit_search(
        _problem(),
        _existing_network(),
        RetrofitSettings(
            strategy="stepwise", max_modifications=2, new_match_candidates=4, **COSTS
        ),
    )

    kinds = {m.kind for d in result.frontier for m in d.modifications}
    assert kinds == {"area_addition"}
    assert result.evaluated_designs > len(result.frontier)


def test_a_relocation_never_leaves_a_stream_without_an_exchanger(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # Without its cooler E6, H2 is served by E4 alone: moving E4 to a match
    # without H2 would leave H2 with no exchanger, which is no network.
    network = _existing_network()
    network = network.model_copy(
        update={
            "exchangers": tuple(e for e in network.exchangers if e.exchanger_id != "E6")
        }
    )

    result = run_retrofit_search(
        _problem(),
        network,
        RetrofitSettings(
            max_modifications=1,
            beam_width=50,
            new_match_candidates=0,
            relocation_cost=5_000.0,
            relocation_candidates=100,
            **COSTS,
        ),
    )

    assert result.evaluated_designs > 1
    for design in result.frontier:
        e4 = next(e for e in design.network.exchangers if e.exchanger_id == "E4")
        assert e4.source_stream.endswith("H2")


def test_design_retrofit_runs_the_search_from_a_mapping(fake_solver):
    existing = {
        "recovery": [
            {"hot": "H3", "cold": "C2", "stage": 1, "area": 609.7, "id": "E1"},
            {"hot": "H3", "cold": "C1", "stage": 2, "area": 579.2, "id": "E2"},
            {"hot": "H1", "cold": "C1", "stage": 1, "area": 1008.5, "id": "E3"},
            {"hot": "H2", "cold": "C2", "stage": 2, "area": 117.96, "id": "E4"},
        ],
        "heaters": [{"stream": "C2", "utility": "HU", "area": 246.75, "id": "E7"}],
        "coolers": [
            {"stream": "H1", "area": 787.5, "id": "E5"},
            {"stream": "H2", "area": 104.6, "id": "E6"},
        ],
    }
    problem = _problem()
    stored = dict(problem.config)

    result = problem.design.retrofit(
        existing,
        max_modifications=1,
        new_match_candidates=1,
        new_utility_units=True,
        relocation_cost=5_000.0,
        relocation_candidates=1,
        max_added_area_fraction=0.5,
    )

    base = result.frontier[0]
    assert {e.exchanger_id: e.area for e in base.network.exchangers} == {
        "E1": 609.7,
        "E2": 579.2,
        "E3": 1008.5,
        "E4": 117.96,
        "E5": 787.5,
        "E6": 104.6,
        "E7": 246.75,
    }
    new_matches = [
        m for d in result.frontier for m in d.modifications if m.kind == "new_match"
    ]
    # A new match ($100,000/y on the fake) beats added area ($20,000/y), and
    # its unit cost defaults to the case's COSTING_HX_UNIT_COST.
    (match,) = new_matches
    assert match.fixed_cost >= 34_600.0
    # The search runs on a snapshot under a temporary configuration.
    assert dict(problem.config) == stored


INSTALLED = {
    "E1": 609.7,
    "E2": 579.2,
    "E3": 1008.5,
    "E4": 117.96,
    "E5": 787.5,
    "E6": 104.6,
    "E7": 246.75,
}


def test_the_structure_strategy_steps_by_structure_and_sizes_area(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    result = run_retrofit_search(
        _problem(),
        _existing_network(),
        RetrofitSettings(
            strategy="structure",
            max_modifications=2,
            new_match_candidates=2,
            **COSTS,
        ),
    )

    assert len(result.frontier) == 3
    for depth, design in enumerate(result.frontier):
        structural = [m for m in design.modifications if m.kind != "area_addition"]
        assert len(structural) == depth
    additions = [
        m
        for design in result.frontier[1:]
        for m in design.modifications
        if m.kind == "area_addition"
    ]
    # Every existing unit may grow on the fake; a unit keeps its added area
    # only when it is at least 1 % of its installed area (E3's 10 m2 on
    # 1,008.5 m2 is not), and each pays the area-addition cost once.
    assert additions
    for m in additions:
        assert m.added_area >= 0.01 * INSTALLED[m.exchanger_id]
        assert m.fixed_cost == pytest.approx(COSTS["area_addition_cost"])
        assert m.area_cost == pytest.approx(AREA_COST * m.added_area)
    assert "E3" not in {m.exchanger_id for m in additions}


def test_design_retrofit_takes_the_search_strategy(fake_solver):
    problem = _problem()

    result = problem.design.retrofit(
        _existing_network(),
        strategy="structure",
        ceiling_candidates=3,
        max_modifications=1,
        new_match_candidates=1,
    )

    (design,) = result.frontier[1:]
    assert [m.kind for m in design.modifications if m.kind != "area_addition"] == [
        "new_match"
    ]
    with pytest.raises(ValueError, match="strategy"):
        problem.design.retrofit(_existing_network(), strategy="greedy")


def test_design_retrofit_steps_by_structure_by_default(fake_solver):
    result = _problem().design.retrofit(
        _existing_network(), max_modifications=2, new_match_candidates=2
    )

    for depth, design in enumerate(result.frontier):
        structural = [m for m in design.modifications if m.kind != "area_addition"]
        assert len(structural) == depth
    assert any(
        m.kind == "area_addition" for d in result.frontier for m in d.modifications
    )


def test_the_ceiling_ranks_relocations_that_give_up_more_than_they_gain(
    fake_solver, monkeypatch
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # Every relocation now scores below zero: the move gives up more duty
    # than its new position's potential. The stepwise search drops such
    # moves; the structure search lets the ceiling judge them.
    monkeypatch.setattr(search, "match_potential", lambda *args, **kwargs: 0.0)
    ranked = []
    ceiling = search._Evaluator.ceiling

    def spy(self, state):
        ranked.append(state.moves[-1].kind)
        return ceiling(self, state)

    monkeypatch.setattr(search._Evaluator, "ceiling", spy)

    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            new_match_candidates=1,
            relocation_cost=5_000.0,
            **COSTS,
        ),
    )

    assert ranked.count("relocation") > 0


def test_relocations_and_new_matches_each_get_their_best_solved(
    fake_solver, monkeypatch
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # On the fake a new match always has the lower ceiling (one more
    # recovery match), so one shared quota would solve new matches only.
    solved = []
    sized = search._Evaluator.sized

    def spy(self, state):
        solved.append(state.moves[-1].kind)
        return sized(self, state)

    monkeypatch.setattr(search._Evaluator, "sized", spy)

    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            new_match_candidates=2,
            relocation_cost=5_000.0,
            relocation_candidates=4,
            **COSTS,
        ),
    )

    assert solved.count("new_match") == 2
    assert solved.count("relocation") == 2


def test_the_structure_search_ranks_every_relocation_by_its_ceiling(
    fake_solver, monkeypatch
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # The pre-score barely tracks the ceiling (the Smith case's best move
    # scores 127th of 395), so the structure search caps nothing before
    # the ceiling ranking: relocation_candidates bounds the stepwise search.
    ranked = []
    ceiling = search._Evaluator.ceiling

    def spy(self, state):
        ranked.append(state.moves[-1].kind)
        return ceiling(self, state)

    monkeypatch.setattr(search._Evaluator, "ceiling", spy)

    def relocations_ranked(candidates: int) -> int:
        ranked.clear()
        search.run_retrofit_search(
            _problem(),
            _existing_network(),
            search.RetrofitSettings(
                max_modifications=1,
                new_match_candidates=1,
                relocation_cost=5_000.0,
                relocation_candidates=candidates,
                **COSTS,
            ),
        )
        return ranked.count("relocation")

    assert relocations_ranked(1) == relocations_ranked(1_000) > 1


def test_hens_max_parallel_ranks_ceilings_on_threads_with_the_same_result(
    fake_solver, monkeypatch
):
    import threading

    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
    from OpenPinch.analysis.parallelism import supported_workers

    threads = set()
    ceiling = search._Evaluator.ceiling

    def spy(self, state):
        threads.add(threading.current_thread().name)
        return ceiling(self, state)

    monkeypatch.setattr(search._Evaluator, "ceiling", spy)

    def run(max_parallel: int):
        case = json.loads(read_sample_case(CASE))
        case["options"]["HENS_MAX_PARALLEL"] = max_parallel
        result = search.run_retrofit_search(
            PinchProblem(case),
            _existing_network(),
            search.RetrofitSettings(
                max_modifications=2, relocation_cost=5_000.0, **COSTS
            ),
        )
        return result.evaluated_designs, [
            (d.hot_utility, d.net_annual_benefit) for d in result.frontier
        ]

    serial = run(1)
    assert threads == {threading.main_thread().name}
    threads.clear()
    assert run(4) == serial
    # Pyodide cannot start threads, so the search stays serial there.
    if supported_workers(4) > 1:
        assert threading.main_thread().name not in threads
    else:
        assert threads == {threading.main_thread().name}


def test_the_ceiling_minimises_utility_cost_not_duty(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # The ranking reads the ceiling's utility cost, so the ceiling must
    # minimise it: with several utilities at different prices, minimising
    # duty leaves the split between them arbitrary.
    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(max_modifications=1, **COSTS),
    )

    assert fake_solver.goals["retrofit-ceiling"] == {"utility cost"}


@pytest.mark.parametrize("strategy", ["structure", "stepwise"])
def test_hens_max_parallel_solves_each_step_on_threads_with_the_same_result(
    fake_solver, monkeypatch, strategy
):
    import threading

    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
    from OpenPinch.analysis.parallelism import supported_workers

    threads = set()
    evaluate = search._Evaluator.evaluate

    def spy(self, state):
        if state.moves:
            threads.add(threading.current_thread().name)
        return evaluate(self, state)

    monkeypatch.setattr(search._Evaluator, "evaluate", spy)

    def run(max_parallel: int):
        case = json.loads(read_sample_case(CASE))
        case["options"]["HENS_MAX_PARALLEL"] = max_parallel
        result = search.run_retrofit_search(
            PinchProblem(case),
            _existing_network(),
            search.RetrofitSettings(max_modifications=2, strategy=strategy, **COSTS),
        )
        return result.evaluated_designs, [
            (d.net_annual_benefit, [m.exchanger_id for m in d.modifications])
            for d in result.frontier
        ]

    serial = run(1)
    assert threads == {threading.main_thread().name}
    threads.clear()
    assert run(4) == serial
    if supported_workers(4) > 1:
        assert threading.main_thread().name not in threads


def test_the_stepwise_search_adds_area_only_where_more_area_is_worth_something(
    fake_solver, monkeypatch
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # Every installed unit is at its cap on the fake, but the solver's
    # duals say only the first recovery unit's area is worth anything.
    fake_solver.area_values = lambda spec: {
        **{("recovery", r): 0.0 for r in spec.recovery_areas},
        **{("utility", e): 0.0 for e in spec.utility_areas},
        ("recovery", min(spec.recovery_areas)): 50.0,
    }
    tried = []
    evaluate = search._Evaluator.evaluate

    def spy(self, state):
        if state.moves and state.moves[-1].kind == "area_addition":
            tried.append(state.moves[-1].exchanger_id)
        return evaluate(self, state)

    monkeypatch.setattr(search._Evaluator, "evaluate", spy)

    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(max_modifications=1, strategy="stepwise", **COSTS),
    )

    assert len(tried) == 1


def test_the_stepwise_search_can_enhance_an_existing_unit_at_its_cap(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # Area additions and new units are priced out, so the best first step
    # is enhancing E3, the only unit with an enhancement factor.
    result = search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            strategy="stepwise",
            new_unit_cost=1e9,
            area_addition_cost=1e9,
            split_cost=1e4,
            enhancement_factors={"E3": 1.3},
            enhancement_cost=500.0,
            enhancement_area_cost=10.0,
        ),
    )

    (step,) = result.frontier[1].modifications
    assert step.kind == "enhancement"
    assert step.exchanger_id == "E3"
    assert step.added_area == 0.0
    # The paper's insert cost: $500 plus $10 per m2 of the unit's area.
    assert step.fixed_cost == pytest.approx(500.0 + 10.0 * 1008.5)
    assert step.area_cost == 0.0
    assert result.frontier[1].annual_savings == pytest.approx(20_000.0)


def test_an_enhanced_unit_is_verified_at_its_enhanced_capacity(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # Verification sizes each unit at the base heat transfer coefficient,
    # so an enhanced unit must be judged as its widened area times the
    # factor, as the model rates it.
    verified = []

    def record(network):
        areas = {e.exchanger_id: e.area for e in network.exchangers}
        verified.append(areas["E3"])
        return ()

    fake_solver.reject = record
    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            strategy="stepwise",
            area_tolerance=0.02,
            enhancement_factors={"E3": 1.3},
            **COSTS,
        ),
    )

    assert verified[0] == pytest.approx(1008.5 * 1.02)
    assert pytest.approx(1008.5 * 1.02 * 1.3) in verified


def test_the_structure_search_enhances_units_it_would_enlarge(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # On the fake an enhanced unit saves $20,000/y; at $500 an insert is
    # worth it, so each sized structure keeps E1 enhanced as well as
    # enlarged (the fake still adds its area).
    result = search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            enhancement_factors={"E1": 1.3},
            enhancement_cost=500.0,
            **COSTS,
        ),
    )

    kinds = {(m.kind, m.exchanger_id) for m in result.frontier[1].modifications}
    assert ("enhancement", "E1") in kinds
    assert ("area_addition", "E1") in kinds
    structural = [
        m
        for m in result.frontier[1].modifications
        if m.kind not in {"area_addition", "enhancement"}
    ]
    assert len(structural) == 1


def test_a_unit_given_area_in_one_step_can_be_enhanced_in_the_next(
    fake_solver, monkeypatch
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # New units and inserts are priced so the first step's beam holds area
    # additions only; the second step must still offer each enlarged unit
    # an insert (it may replace part of the added area).
    states = []
    evaluate = search._Evaluator.evaluate

    def spy(self, state):
        states.append(state)
        return evaluate(self, state)

    monkeypatch.setattr(search._Evaluator, "evaluate", spy)
    units = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=2,
            strategy="stepwise",
            new_unit_cost=1e9,
            area_addition_cost=8_650.0,
            split_cost=1e4,
            enhancement_factors=dict.fromkeys(units, 1.3),
            enhancement_cost=1e6,
        ),
    )

    assert any(state.enlarged & state.enhanced for state in states)


def test_enhancement_factors_for_unknown_exchangers_are_an_error(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    with pytest.raises(ValueError, match="enhancement_factors.*E9"):
        search.run_retrofit_search(
            _problem(),
            _existing_network(),
            search.RetrofitSettings(
                max_modifications=1,
                enhancement_factors={"E1": 1.3, "E9": 1.2},
                **COSTS,
            ),
        )
