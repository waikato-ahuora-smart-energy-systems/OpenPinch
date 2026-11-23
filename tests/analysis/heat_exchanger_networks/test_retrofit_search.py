"""Retrofit search mechanics on a fake fixed-structure solver (no solver).

The fake keeps the real request building, structure and screen, and stands
in for the NLP only: a design's utility cost falls by $100,000/y per
recovery match, $150,000/y per heat pump or heat recovery loop and
$20,000/y per enlarged or
enhanced unit, every exchanger carries 100 kW, every installed unit needs
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
        self.unbuildable = lambda spec: False
        self.rating_allowances = lambda spec: None
        self.heat_pumps: list = []
        self.loops: list = []


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
            if backend.unbuildable(spec):
                raise ValueError("streams H1 have no heater or cooler")
            self.spec = spec
            backend.goals.setdefault(name, set()).add(minimisation_goal)

        def optimise(self, print_output: bool) -> None:
            backend.solved += 1
            spec = self.spec
            enlarged = len(spec.enlargeable_recovery) + len(spec.enlargeable_utilities)
            enhanced = len(spec.enhanced_recovery) + len(spec.enhanced_utilities)
            backend.heat_pumps.append((spec.utilities, spec.heat_pumps))
            backend.loops.append(spec.loops)
            self.utility_cost_value = (
                1_000_000.0
                - 100_000.0 * len(spec.recovery)
                - 20_000.0 * (enlarged + enhanced)
                - 150_000.0 * len(spec.heat_pumps)
                - 150_000.0 * len(spec.loops)
            )
            self.area_values = backend.area_values(spec)
            self.rating_allowances = backend.rating_allowances(spec)
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


def test_the_structure_search_drops_area_additions_that_do_not_pay(fake_solver):
    """The sizing solve does not see an area addition's fixed cost, so it
    grows every unit that earns its area cost. Here each addition saves
    $20,000/y and costs (200,000 + 857 x 10) x 0.2 = $41,714/y: none pays,
    and the design keeps the new match alone."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    result = run_retrofit_search(
        _problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=1,
            new_unit_cost=34_600.0,
            area_addition_cost=200_000.0,
            split_cost=1e4,
        ),
    )

    kinds = [m.kind for m in result.frontier[1].modifications]
    assert kinds == ["new_match"]
    assert fake_solver.cleaned == fake_solver.solved == result.evaluated_designs


def test_the_result_counts_the_candidate_solves_that_failed_at_each_step(fake_solver):
    """A candidate whose solve does not converge is dropped. The result
    says how many were at each step, and the search warns when over a
    quarter of its solves went that way: a frontier built on the few that
    solved may miss the better designs."""
    import warnings

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    def run():
        return run_retrofit_search(
            _problem(),
            _existing_network(),
            RetrofitSettings(max_modifications=2, **COSTS),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run().failed_solves == (0, 0)

    # No structure with a fifth recovery match solves, so step 1 ends the
    # search with every candidate lost.
    fake_solver.fail = lambda spec: len(spec.recovery) > 4
    with pytest.warns(UserWarning, match="retrofit candidate solves failed"):
        result = run()

    assert len(result.frontier) == 1
    assert result.failed_solves == (result.evaluated_designs - 1,)
    assert result.failed_solves[0] > 0


def test_an_existing_unit_reports_its_installed_area_and_what_was_added(fake_solver):
    """The design's network gave an existing unit the area its duty needs
    (for a unit on a segmented stream, the sum of its slices), not the area
    it has. On the Hallberg and Scribner crude case 12 of 18 units differed:
    E1 read 3,262 m2 against 4,303 m2 installed, and a unit reading 5 % of
    its area made the design's rating allowance 2.33."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    result = run_retrofit_search(
        _problem(), _existing_network(), RetrofitSettings(max_modifications=1, **COSTS)
    )

    design = result.frontier[1]
    added = {
        m.exchanger_id: m.added_area
        for m in design.modifications
        if m.kind == "area_addition"
    }
    assert added
    reported = {e.exchanger_id: e.area for e in design.network.exchangers}
    for unit in _existing_network().exchangers:
        assert reported[unit.exchanger_id] == pytest.approx(
            unit.area + added.get(unit.exchanger_id, 0.0)
        )


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


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (
            {"enhancement_pressure_drops": {"E9": ("C1", 5.0)}},
            "enhancement_pressure_drops.*E9",
        ),
        (
            {"enhancement_pressure_drops": {"E1": ("C9", 5.0)}},
            "enhancement_pressure_drops.*C9",
        ),
        ({"pressure_drop_allowances": {"C9": 30.0}}, "pressure_drop_allowances.*C9"),
    ],
)
def test_pressure_drop_settings_must_name_existing_exchangers_and_streams(
    fake_solver, extra, match
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    with pytest.raises(ValueError, match=match):
        search.run_retrofit_search(
            _problem(),
            _existing_network(),
            search.RetrofitSettings(
                max_modifications=1,
                enhancement_factors={"E1": 1.3},
                **extra,
                **COSTS,
            ),
        )


def test_the_structure_search_keeps_inserts_within_the_pressure_drop_allowance(
    fake_solver,
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # E1 and E2 are both enlarged on the fake and both worth an insert, but
    # C1 can take only one insert's pressure drop.
    result = search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            enhancement_factors={"E1": 1.3, "E2": 1.3},
            enhancement_cost=500.0,
            enhancement_pressure_drops={"E1": ("C1", 10.0), "E2": ("C1", 10.0)},
            pressure_drop_allowances={"C1": 15.0},
            **COSTS,
        ),
    )

    enhanced = [
        m.exchanger_id
        for m in result.frontier[1].modifications
        if m.kind == "enhancement"
    ]
    assert len(enhanced) == 1


def test_competing_inserts_are_chosen_by_their_return_per_unit_of_pressure_drop(
    fake_solver,
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # E1 and E2 are both enlarged by the same area on the fake and each
    # insert saves the same; E2's costs a little less and adds a third of
    # E1's pressure drop, and C1 takes only one of them.
    result = search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1,
            enhancement_factors={"E1": 1.3, "E2": 1.3},
            enhancement_cost=500.0,
            enhancement_area_cost=10.0,
            enhancement_pressure_drops={"E1": ("C1", 12.0), "E2": ("C1", 4.0)},
            pressure_drop_allowances={"C1": 15.0},
            **COSTS,
        ),
    )

    enhanced = [
        m.exchanger_id
        for m in result.frontier[1].modifications
        if m.kind == "enhancement"
    ]
    assert enhanced == ["E2"]


def test_an_insert_is_rejected_once_a_relocation_takes_its_stream_away():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
        NewMatch,
        apply_relocation,
    )

    # E4 (H2-C2) has its insert on C2; moved to H2-C1 it no longer serves
    # C2, so its insert's pressure drop has no stream to be checked against.
    settings = search.RetrofitSettings(
        enhancement_factors={"E4": 1.3},
        enhancement_pressure_drops={"E4": ("C2", 5.0)},
        pressure_drop_allowances={"C2": 30.0},
    )
    network = _existing_network()
    moved = apply_relocation(network, "E4", NewMatch("H2", "C1", 1, False))

    assert search._within_pressure_drop({"E4"}, settings, network)
    assert not search._within_pressure_drop({"E4"}, settings, moved)


@pytest.mark.parametrize("strategy", ["structure", "stepwise"])
def test_a_structure_the_model_cannot_build_is_dropped(fake_solver, strategy):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # A move can leave streams with no utility exchanger whose loads cannot
    # balance; the model refuses to build it. The search drops the move.
    fake_solver.unbuildable = lambda spec: len(spec.recovery) > 4
    result = search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(
            max_modifications=1, strategy=strategy, relocation_cost=5_000.0, **COSTS
        ),
    )

    assert not any(
        m.kind == "new_match" for d in result.frontier for m in d.modifications
    )


def test_ceiling_relocations_caps_the_relocations_ranked_per_design(
    fake_solver, monkeypatch
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # A large network can offer thousands of relocations (14,617 on the
    # pulp mill); the structure search ranks at most ceiling_relocations
    # of them per design, the best pre-scored first.
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
            ceiling_relocations=2,
            **COSTS,
        ),
    )

    assert ranked.count("relocation") == 2


def test_verification_uses_the_models_rating_allowance_for_installed_units(
    fake_solver,
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    # The model reports how much more area verification needs than its cap
    # for each installed unit (its design calibration and Chen's LMTD);
    # verification widens the unit by it, so the two judge alike.
    fake_solver.rating_allowances = lambda spec: {
        **{("recovery", r): 1.1 for r in spec.recovery_areas},
        **{("utility", e): 1.1 for e in spec.utility_areas},
    }
    verified = []

    def record(network):
        verified.append({e.exchanger_id: e.area for e in network.exchangers})
        return ()

    fake_solver.reject = record
    search.run_retrofit_search(
        _problem(),
        _existing_network(),
        search.RetrofitSettings(max_modifications=0, area_tolerance=0.02, **COSTS),
    )

    assert verified[0]["E3"] == pytest.approx(1008.5 * 1.02 * 1.1)
    assert verified[0]["E7"] == pytest.approx(246.75 * 1.02 * 1.1)


@pytest.mark.parametrize("strategy", ["stepwise", "structure"])
@pytest.mark.parametrize("kind", ["new_match", "relocation"])
def test_repiping_cost_prices_every_stream_a_move_repipes(fake_solver, strategy, kind):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )
    from OpenPinch.domain.enums import HeatExchangerKind

    # A new match repipes both its streams, and a paired end heater or
    # cooler its stream. A relocation repipes the streams it leaves and
    # joins; a stream it stays on, in the same stage, keeps its piping.
    if kind == "relocation":
        # New matches and added area fail, so the best step is a relocation.
        fake_solver.fail = lambda spec: (
            len(spec.recovery) > 4
            or bool(spec.enlargeable_recovery or spec.enlargeable_utilities)
        )
    existing = _existing_network()
    before = {e.exchanger_id: e for e in existing.exchangers}
    repiping = 2_000.0
    result = run_retrofit_search(
        _problem(),
        existing,
        RetrofitSettings(
            strategy=strategy,
            max_modifications=1,
            relocation_cost=5_000.0,
            new_utility_units=True,
            repiping_cost=repiping,
            **COSTS,
        ),
    )

    design = result.frontier[1]
    moves = [m for m in design.modifications if m.kind == kind]
    assert len(moves) == 1
    for m in design.modifications:
        if m.kind in {"area_addition", "enhancement"}:
            assert m.repiped_streams == ()
    m = moves[0]
    expected = {m.hot_stream, m.cold_stream}
    if kind == "relocation":
        old = before[m.exchanger_id]
        kept = (
            {old.source_stream, old.sink_stream}.intersection(expected)
            if not m.new_stage and m.stage == old.stage
            else set()
        )
        expected = (expected | {old.source_stream, old.sink_stream}) - kept
    fixed = {"new_match": COSTS["new_unit_cost"], "relocation": 5_000.0}[kind]
    fixed += COSTS["split_cost"] * len(m.new_splits)
    if m.utility_exchanger_id is not None:
        utility = next(
            e
            for e in design.network.exchangers
            if e.exchanger_id == m.utility_exchanger_id
        )
        heater = utility.kind is HeatExchangerKind.HOT_UTILITY
        expected.add(utility.sink_stream if heater else utility.source_stream)
        fixed += COSTS["new_unit_cost"]
    assert set(m.repiped_streams) == expected
    assert len(m.repiped_streams) == len(expected)
    assert m.fixed_cost == pytest.approx(fixed + repiping * len(expected))


def test_a_paired_utility_on_a_stream_the_match_repipes_is_charged_once(fake_solver):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # Only a new match paired with a new end utility on one of its own
    # streams solves (a new cooler on H3, which has none, with an H3 match).
    def shares_a_stream(spec) -> bool:
        match, utility = spec.recovery[-1], spec.utilities[-1]
        stream = match.hot if utility.side == "cold" else match.cold
        return len(spec.utilities) == 4 and utility.stream == stream

    fake_solver.fail = lambda spec: len(spec.recovery) > 4 and not shares_a_stream(spec)
    result = run_retrofit_search(
        _problem(),
        _existing_network(),
        RetrofitSettings(
            max_modifications=1,
            new_match_candidates=50,
            ceiling_candidates=50,
            new_utility_units=True,
            repiping_cost=1_000.0,
            **COSTS,
        ),
    )

    (match,) = [m for m in result.frontier[1].modifications if m.kind == "new_match"]
    assert match.utility_exchanger_id is not None
    assert sorted(match.repiped_streams) == sorted(
        {match.hot_stream, match.cold_stream}
    )
    fixed = 2 * COSTS["new_unit_cost"] + COSTS["split_cost"] * len(match.new_splits)
    assert match.fixed_cost == pytest.approx(fixed + 2 * 1_000.0)


def test_design_retrofit_takes_the_repiping_cost(fake_solver):
    result = _problem().design.retrofit(
        _existing_network(),
        max_modifications=1,
        new_match_candidates=1,
        new_unit_cost=0.0,
        repiping_cost=3_000.0,
    )

    (match,) = [m for m in result.frontier[1].modifications if m.kind == "new_match"]
    assert len(match.repiped_streams) == 2
    assert match.fixed_cost == pytest.approx(6_000.0)


@pytest.mark.parametrize("strategy", ["stepwise", "structure"])
def test_a_heat_pump_step_is_priced_by_its_capacity_and_exchanger_area(
    fake_solver, strategy
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    # The existing network heats only C2, so a heat pump delivers there,
    # from H1 or H2, which have coolers. Its evaporator and condenser carry
    # the fake's 100 kW and 25 m2 each. Either lifts about 200 K, so the
    # lift limit is taken off.
    result = run_retrofit_search(
        _problem(),
        _existing_network(),
        RetrofitSettings(
            strategy=strategy,
            max_modifications=1,
            heat_pump_cycle="carnot",
            heat_pump_max_lift=None,
            heat_pump_cost=300.0,
            repiping_cost=1_000.0,
            **COSTS,
        ),
    )

    design = result.frontier[1]
    (pump,) = [m for m in design.modifications if m.kind == "heat_pump"]
    assert (pump.hot_stream, pump.cold_stream) in {("H1", "C2"), ("H2", "C2")}
    assert pump.repiped_streams == (pump.hot_stream, pump.cold_stream)
    assert pump.heat_pump_capacity == pytest.approx(100.0)
    assert pump.heat_pump_cop > 1.0
    # The compressor at its capacity, the evaporator and condenser as new
    # exchangers, and the two repiped streams.
    assert pump.fixed_cost == pytest.approx(
        300.0 * 100.0 + 2 * COSTS["new_unit_cost"] + 2 * 1_000.0
    )
    assert (pump.added_area, pump.utility_area) == (NEW_UNIT_AREA, NEW_UNIT_AREA)
    assert pump.area_cost == pytest.approx(AREA_COST * 2 * NEW_UNIT_AREA)
    exchangers = {e.exchanger_id: e for e in design.network.exchangers}
    evaporator = exchangers[pump.exchanger_id]
    condenser = exchangers[pump.utility_exchanger_id]
    assert (evaporator.source_stream, condenser.sink_stream) == (
        pump.hot_stream,
        pump.cold_stream,
    )
    # The heat pump's condenser and evaporator are not hot and cold utility.
    assert design.hot_utility == pytest.approx(design.utility_cost / 100.0 - 100.0)
    assert design.cold_utility == pytest.approx(design.utility_cost / 200.0 - 100.0)
    # The model links the pair's duties by the COP.
    linked = {
        (utilities[evaporator].side, utilities[condenser].side, cop)
        for utilities, pumps in fake_solver.heat_pumps
        for evaporator, condenser, cop in pumps
    }
    assert ("cold", "hot", pump.heat_pump_cop) in linked


def test_a_carnot_heat_pump_is_offered_only_within_the_lift_limit(fake_solver):
    """A Carnot cycle has a COP at any lift. On this case a heat pump to C2
    (heated to 265 degC) lifts 208 K from H1 and 197 K from H2, which no
    machine does, so by default (140 K, a two-stage cascade) none is
    offered. ``heat_pump_max_lift`` moves the limit; ``None`` removes it."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    def pumps(**limit):
        result = run_retrofit_search(
            _problem(),
            _existing_network(),
            RetrofitSettings(
                max_modifications=1,
                heat_pump_cycle="carnot",
                heat_pump_cost=300.0,
                **COSTS,
                **limit,
            ),
        )
        return [
            (m.hot_stream, m.cold_stream)
            for design in result.frontier
            for m in design.modifications
            if m.kind == "heat_pump"
        ]

    assert pumps() == []
    assert pumps(heat_pump_max_lift=200.0) == [("H2", "C2")]
    assert len(pumps(heat_pump_max_lift=None)) == 1


def test_a_carnot_heat_pump_with_a_free_compressor_is_warned_about(fake_solver):
    """``heat_pump_cost`` is 0 by default, which prices a Carnot heat pump
    as its two exchangers alone: on this case a 15 MW machine cost $69,200,
    two exchanger unit costs, and the search installed it."""
    import warnings

    def retrofit(**heat_pump):
        return _problem().design.retrofit(
            _existing_network(), max_modifications=0, **COSTS, **heat_pump
        )

    with pytest.warns(UserWarning, match="heat_pump_cost"):
        retrofit(heat_pump_cycle="carnot")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        retrofit(heat_pump_cycle="carnot", heat_pump_cost=300.0)
        retrofit()


def test_design_retrofit_rates_carnot_heat_pumps_with_the_hpr_efficiency(fake_solver):
    def pump(eta_ii: float):
        result = _problem().design.retrofit(
            _existing_network(),
            max_modifications=1,
            heat_pump_cycle="carnot",
            heat_pump_max_lift=None,
            heat_pump_cost=300.0,
            new_unit_cost=0.0,
            options={"HPR_ETA_II_CARNOT": eta_ii},
        )
        (pump,) = [m for m in result.frontier[1].modifications if m.kind == "heat_pump"]
        return pump

    low, high = pump(0.5), pump(0.6)

    # The HPR Carnot heating COP is 1 + eta_II T_evap / lift.
    assert (low.hot_stream, low.cold_stream) == (high.hot_stream, high.cold_stream)
    assert high.heat_pump_cop - 1.0 == pytest.approx(
        1.2 * (low.heat_pump_cop - 1.0), rel=1e-12
    )
    assert low.fixed_cost == pytest.approx(300.0 * 100.0)


def test_an_unknown_heat_pump_cycle_is_an_error(fake_solver):
    with pytest.raises(ValueError, match="heat_pump_cycle.*brayton"):
        _problem().design.retrofit(_existing_network(), heat_pump_cycle="brayton")
    assert fake_solver.solved == 0


def test_a_real_fluid_heat_pump_is_priced_by_the_hpr_machine_capital(fake_solver):
    from OpenPinch.analysis.heat_pumps.common.capital import (
        CompressorDuty,
        MachineComponents,
        installed_capital,
    )
    from OpenPinch.analysis.heat_pumps.cycles.limits import CycleLimits
    from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
        VapourCompressionCycle,
    )

    # Ciric and Floudas heats only C2, to 538 K: a single water stage lifts
    # that far only with the pressure ratio and suction limits relaxed. New
    # matches fail, so the step is a heat pump, however dear its machine.
    fake_solver.fail = lambda spec: len(spec.recovery) > 4
    options = {
        "HPR_REFRIGERANTS": ["water"],
        "HPR_MAX_PRESSURE_RATIO": 200.0,
        "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.1,
    }
    result = _problem().design.retrofit(
        _existing_network(),
        max_modifications=1,
        heat_pump_cycle="vapour_compression",
        heat_pump_cost=300.0,
        repiping_cost=1_000.0,
        # The cycle below is built at a 1 K approach; the retrofit default
        # is now the case's minimum approach.
        min_approach_temperature=1.0,
        options=options,
    )

    design = result.frontier[1]
    (pump,) = [m for m in design.modifications if m.kind == "heat_pump"]
    # The HPR service's spelling of the fluid.
    assert pump.heat_pump_refrigerant == "WATER"
    # The machine (compressor, motor, evaporator and condenser) at its 100 kW
    # capacity, as the HPR service prices it; heat_pump_cost is for Carnot.
    hot = {"H1": 350.15, "H2": 361.15}[pump.hot_stream] - 1.0
    cycle = VapourCompressionCycle()
    cycle.solve(
        hot - 273.15,
        538.15 + 1.0 - 273.15,
        dtcont=1.0,
        eta_comp=0.7,
        eta_motor=0.95,
        refrigerant="water",
        Q_heat=100.0,
        limits=CycleLimits(
            max_pressure_ratio=200.0,
            max_discharge_superheat=50.0,
            max_discharge_pressure=60e5,
            max_discharge_pressure_co2=140e5,
            min_suction_pressure=0.1e5,
            min_lift=5.0,
        ),
    )
    machine = MachineComponents(
        compressors=(
            CompressorDuty("water", cycle.suction_volume_flow, cycle.shaft_power),
        ),
        exchanger_areas=(NEW_UNIT_AREA, NEW_UNIT_AREA),
    )
    assert pump.heat_pump_cop == pytest.approx(cycle.COP_h, rel=1e-9)
    assert pump.fixed_cost == pytest.approx(installed_capital(machine) + 2_000.0)
    assert pump.area_cost == 0.0


def test_a_utility_named_like_a_heat_pump_keeps_its_own_properties(fake_solver):
    # The case's hot utility carries the name the H1 -> C2 heat pump's
    # condenser would take; the existing heater must still draw on it.
    name = "HP H1->C2:condenser"
    data = json.loads(read_sample_case(CASE))
    for utility in data["utilities"]:
        if utility["type"] == "Hot":
            utility["name"] = name
    existing = _existing_network()
    existing = existing.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"source_stream": name})
                if e.source_stream == "HU"
                else e
                for e in existing.exchangers
            )
        }
    )

    PinchProblem(data).design.retrofit(
        existing, max_modifications=1, heat_pump_cycle="carnot"
    )

    baseline, _pumps = fake_solver.heat_pumps[0]
    assert [u.utility for u in baseline if u.side == "hot"] == [0]


def test_a_period_named_like_a_heat_pump_leaves_the_heat_pump_offered(fake_solver):
    # Only stream and utility names can clash with a heat pump's utilities;
    # a period, stage or segment of the same name does not.
    def pump(problem: PinchProblem):
        result = problem.design.retrofit(
            _existing_network(),
            max_modifications=1,
            heat_pump_cycle="carnot",
            heat_pump_max_lift=None,
        )
        (pump,) = [m for m in result.frontier[1].modifications if m.kind == "heat_pump"]
        return pump.hot_stream, pump.cold_stream

    hot, cold = pump(_problem())
    named = _problem()
    named.update_options({"PROBLEM_PERIOD_IDS": [f"HP {hot}->{cold}:condenser"]})

    assert pump(named) == (hot, cold)


def test_a_bare_stream_name_two_zones_share_is_refused():
    """An existing network naming ``H1`` when ``A.H1`` and ``B.H1`` both
    exist cannot say which stream it means; mapping both to ``H1`` would
    let one stream's ends and CP stand for the other's."""
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        _existing_names,
    )

    existing = SimpleNamespace(
        exchangers=(SimpleNamespace(source_stream="H1", sink_stream="A.C1"),)
    )
    axes = {
        "hot_process_streams": {"A.H1": 0, "B.H1": 1},
        "cold_process_streams": {"A.C1": 0},
    }

    with pytest.raises(ValueError, match="'H1'.*A.H1.*B.H1"):
        _existing_names(axes, existing)


def test_a_bare_name_beside_its_qualified_name_is_still_refused():
    """``A.H1`` named in full beside a bare ``H1`` leaves ``B.H1`` the only
    key the bare name could map to; the bare name still fits both zones."""
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        _existing_names,
    )

    existing = SimpleNamespace(
        exchangers=(
            SimpleNamespace(source_stream="A.H1", sink_stream="A.C1"),
            SimpleNamespace(source_stream="H1", sink_stream="A.C1"),
        )
    )
    axes = {
        "hot_process_streams": {"A.H1": 0, "B.H1": 1},
        "cold_process_streams": {"A.C1": 0},
    }

    with pytest.raises(ValueError, match="'H1'.*A.H1.*B.H1"):
        _existing_names(axes, existing)


def test_a_name_one_stream_has_exactly_is_not_a_tail_of_another():
    """``H1`` names the stream keyed ``H1``; ``B.H1`` keeps its own name."""
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        _existing_names,
    )

    existing = SimpleNamespace(
        exchangers=(SimpleNamespace(source_stream="H1", sink_stream="C1"),)
    )
    axes = {
        "hot_process_streams": {"H1": 0, "B.H1": 1},
        "cold_process_streams": {"C1": 0},
    }

    assert _existing_names(axes, existing) == {
        "H1": "H1",
        "B.H1": "B.H1",
        "C1": "C1",
    }


def test_a_bare_stream_name_one_zone_has_maps_to_it():
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        _existing_names,
    )

    existing = SimpleNamespace(
        exchangers=(SimpleNamespace(source_stream="H1", sink_stream="C1"),)
    )
    axes = {
        "hot_process_streams": {"A.H1": 0, "B.H2": 1},
        "cold_process_streams": {"A.C1": 0},
    }

    assert _existing_names(axes, existing) == {
        "A.H1": "H1",
        "B.H2": "B.H2",
        "A.C1": "C1",
    }


def test_a_heat_pump_whose_machine_cannot_be_priced_is_dropped(
    fake_solver, monkeypatch
):
    """The HPR cycle can fail at the heat pump's solved capacity; that
    candidate is dropped and the search goes on, rather than stopping."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    def unpriceable(*args, **kwargs):
        raise ValueError("the cycle does not solve at this capacity")

    monkeypatch.setattr(search, "vapour_compression_capital", unpriceable)
    result = _problem().design.retrofit(
        _existing_network(),
        max_modifications=1,
        heat_pump_cycle="vapour_compression",
        heat_pump_cost=300.0,
        repiping_cost=1_000.0,
        min_approach_temperature=1.0,
        options={
            "HPR_REFRIGERANTS": ["water"],
            "HPR_MAX_PRESSURE_RATIO": 200.0,
            "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.1,
        },
    )

    assert not [
        modification
        for design in result.frontier
        for modification in design.modifications
        if modification.kind == "heat_pump"
    ]


# A two-process site's network, given as the documented mapping (the
# selected total site network of tests/application's ``_site``).
SITE_EXISTING = {
    "recovery": [
        {
            "hot": "Heating.H1",
            "cold": "Heating.C1",
            "stage": 2,
            "area": 2.33,
            "id": "E1",
        },
        {
            "hot": "Heating.H2",
            "cold": "Heating.C1",
            "stage": 4,
            "area": 6.0,
            "id": "E2",
        },
        {
            "hot": "Cooling.H3",
            "cold": "Cooling.C2",
            "stage": 5,
            "area": 2.8,
            "id": "E3",
        },
        {
            "hot": "Cooling.H3",
            "cold": "Cooling.C3",
            "stage": 3,
            "area": 6.0,
            "id": "E4",
        },
    ],
    "heaters": [{"stream": "Heating.C1", "utility": "HPS", "area": 1.18, "id": "E5"}],
    "coolers": [{"stream": "Cooling.H3", "utility": "CW", "area": 1.44, "id": "E6"}],
    "stage_count": 5,
}


def _tried_matches(monkeypatch) -> list[tuple[str, str]]:
    """Record the stream pair of every new match and relocation tried."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    tried: list[tuple[str, str]] = []
    new_match, relocation = search.apply_new_match, search.apply_relocation

    def spy_new_match(network, match, exchanger_id):
        tried.append((match.hot, match.cold))
        return new_match(network, match, exchanger_id)

    def spy_relocation(network, exchanger_id, match):
        tried.append((match.hot, match.cold))
        return relocation(network, exchanger_id, match)

    monkeypatch.setattr(search, "apply_new_match", spy_new_match)
    monkeypatch.setattr(search, "apply_relocation", spy_relocation)
    return tried


@pytest.mark.parametrize("strategy", ["structure", "stepwise"])
def test_a_multi_process_retrofit_matches_streams_only_within_a_process(
    fake_solver, monkeypatch, strategy
):
    from tests.application.test_site_heat_exchanger_network import _site

    tried = _tried_matches(monkeypatch)

    with pytest.warns(UserWarning, match="site utility balance"):
        result = _site().design.retrofit(
            SITE_EXISTING,
            strategy=strategy,
            max_modifications=1,
            new_match_candidates=20,
            relocation_cost=0.0,
            relocation_candidates=20,
        )

    assert tried
    assert all(hot.split(".")[0] == cold.split(".")[0] for hot, cold in tried)
    assert len(result.frontier) == 2


def test_a_single_process_retrofit_screens_every_pair_without_a_site_warning(
    fake_solver, monkeypatch, recwarn
):
    tried = _tried_matches(monkeypatch)

    _problem().design.retrofit(
        _existing_network(), max_modifications=1, new_match_candidates=20
    )

    assert not [w for w in recwarn if "site utility balance" in str(w.message)]
    # H1 is matched with both cold streams, as before.
    assert {("H1", "C1"), ("H1", "C2")} <= set(tried)


def test_a_two_process_retrofit_links_the_processes_through_a_heat_recovery_loop(
    fake_solver,
):
    """Process A has only H1 and process B only C1, so no direct match is
    allowed; a loop between them is proposed, priced as its two new
    exchangers, the loop cost and two repiped streams, and it lowers the
    retrofit objective. Its two utilities are linked as a balanced pair."""
    from tests.analysis.heat_exchanger_networks.test_retrofit_loops import (
        TWO_PROCESS_EXISTING,
        _two_processes,
    )

    problem = _two_processes()
    costing = problem.master_zone.config.costing
    unit_area_cost = costing.hx_area_coeff * NEW_UNIT_AREA**costing.hx_area_exp
    with pytest.warns(UserWarning, match="heat recovery loop"):
        result = problem.design.retrofit(
            TWO_PROCESS_EXISTING,
            max_modifications=1,
            new_unit_cost=34_600.0,
            area_addition_cost=1.0,
            repiping_cost=2_000.0,
            heat_recovery_loop_cost=50_000.0,
        )

    base, design = result.frontier
    # The fake also enlarges both existing units, as area additions.
    (loop,) = [m for m in design.modifications if m.kind != "area_addition"]
    assert loop.kind == "heat_recovery_loop"
    assert (loop.hot_stream, loop.cold_stream) == ("A.H1", "B.C1")
    # Midway between H1's target (120 C) and C1's (90 C).
    assert loop.loop_temperature == pytest.approx(105.0)
    assert loop.loop_duty == pytest.approx(100.0)
    assert loop.exchanger_id == "new:A.H1->Loop A.H1->B.C1:cooler"
    assert loop.utility_exchanger_id == "new:Loop A.H1->B.C1:heater->B.C1"
    assert loop.repiped_streams == ("A.H1", "B.C1")
    assert loop.fixed_cost == pytest.approx(2 * 34_600.0 + 50_000.0 + 2 * 2_000.0)
    assert loop.area_cost == pytest.approx(2 * unit_area_cost)
    assert design.net_annual_benefit > base.net_annual_benefit
    # The loop's own heat is neither hot nor cold utility.
    assert design.hot_utility == pytest.approx(design.utility_cost / 100.0 - 100.0)
    (utilities, _pumps), loops = fake_solver.heat_pumps[-1], fake_solver.loops[-1]
    ((cooler, heater),) = loops
    assert (utilities[cooler].side, utilities[heater].side) == ("cold", "hot")


def test_the_loop_option_turns_heat_recovery_loops_off(fake_solver):
    from tests.analysis.heat_exchanger_networks.test_retrofit_loops import (
        TWO_PROCESS_EXISTING,
        _two_processes,
    )

    with pytest.warns(UserWarning, match="site utility balance") as caught:
        result = _two_processes().design.retrofit(
            TWO_PROCESS_EXISTING,
            max_modifications=1,
            new_unit_cost=0.0,
            options={"HENS_RETROFIT_LOOPS": False},
        )

    assert len(result.frontier) == 1
    assert not fake_solver.loops or not any(fake_solver.loops)
    assert not [w for w in caught if "heat recovery loop" in str(w.message)]


def test_a_single_process_retrofit_proposes_no_heat_recovery_loop(fake_solver):
    result = _problem().design.retrofit(
        _existing_network(), max_modifications=1, new_match_candidates=20
    )

    assert not any(fake_solver.loops)
    assert all(
        m.kind != "heat_recovery_loop"
        for design in result.frontier
        for m in design.modifications
    )
