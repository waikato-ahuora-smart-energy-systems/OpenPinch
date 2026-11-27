"""A site network over a sweep of utility level counts.

With ``levels="placed"``, ``design.total_site_heat_exchanger_network`` places
the site's utilities once per level count, designs the site with each placed
set, records a count whose placement is infeasible or whose design fails, and
selects the cheapest design over the counts.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.errors import WorkflowContractError
from OpenPinch.analysis.utility_placement.errors import NoFeasiblePlacementError
from OpenPinch.application import site_levels
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind
from tests.application.test_site_heat_exchanger_network import (
    _process,
    _site,
    _solvers_available,
    _steam_main_site,
)

# A short placement search: these tests check the sweep, not the placement.
SHORT_PLACEMENT = {"evaluation_limit": 300, "iteration_limit": 50}


def _zonal_site() -> PinchProblem:
    from importlib.resources import files

    source = files("OpenPinch.tutorials.sample_cases") / "zonal_site.json"
    return PinchProblem(json.loads(source.read_text()))


def _designed(cost, *, site_cost=None):
    metrics = {} if site_cost is None else {"site_total_annual_cost": site_cost}
    return SimpleNamespace(
        selected_network=SimpleNamespace(
            total_annual_cost=cost, summary_metrics=metrics
        )
    )


def _fake_place(infeasible=()):
    def place(problem, count, *, placement=None):
        if count in infeasible:
            raise NoFeasiblePlacementError(
                code="no_feasible_placement", message="no placement"
            )
        return SimpleNamespace(count=count)

    return place


@pytest.mark.parametrize(
    ("count", "split"), [(2, (2, 0)), (3, (2, 1)), (4, (3, 1)), (5, (4, 1))]
)
def test_a_count_keeps_the_cases_sensible_levels_while_two_isothermal_remain(
    count, split
):
    """zonal_site has one sensible utility (hot water) among its levels."""
    assert site_levels.level_split(_zonal_site(), count) == split


@pytest.mark.parametrize("count", [2, 3, 4])
def test_cooling_water_over_5_k_is_the_sensible_level_of_the_unit_site(count):
    assert site_levels.level_split(_site(), count) == (
        count - min(1, count - 2),
        min(1, count - 2),
    )


def test_sensible_levels_are_counted_per_side_as_placement_infers_them():
    """Each placed level serves both sides, so hot water (hot) and cooling
    water (cold) are one sensible level on each side, not two."""
    from tests.application.test_site_heat_exchanger_network import _utility

    source = _site().to_problem_json()
    source["utilities"].append(_utility("HW", "Hot", 90.0, 70.0, 30.0))
    problem = PinchProblem(source)

    assert site_levels.level_split(problem, 4) == (3, 1)


def test_a_count_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        site_levels.level_split(_site(), 1)


def test_each_counts_placed_levels_reach_its_design():
    """The design for a count sees the site's streams with that count's
    placed levels. Each level serves both heating and cooling: its hot and
    cold utility share its name, so the site accounts it as one main. The
    steam-main site's placed levels lie apart and need no fallback; beside
    them the case keeps its own cooling water, which rejects to ambient."""
    problem = _steam_main_site()
    seen = {}

    def design(case):
        seen[len(seen) + 2] = case.to_problem_json()
        return _designed(100.0)

    site_levels.run_level_counts(
        problem, (2, 3), design=design, placement=SHORT_PLACEMENT
    )

    streams = sorted(s["name"] for s in problem.to_problem_json()["streams"])
    for count, source in seen.items():
        assert sorted(s["name"] for s in source["streams"]) == streams
        isothermal, sensible = site_levels.level_split(problem, count)
        placed = {(u["name"], u["type"]) for u in source["utilities"]}
        assert placed == {("CW", "Cold")} | {
            (f"{kind}_{level}", side)
            for side in ("Hot", "Cold")
            for kind, levels in (("iso", isothermal), ("sensible", sensible))
            for level in range(1, levels + 1)
        }


def test_a_placed_level_is_a_main_of_the_site_utility_system():
    from OpenPinch.analysis.heat_exchanger_networks.site.levels import (
        site_utility_system,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )

    case = site_levels.placed_case(_steam_main_site(), 2, placement=SHORT_PLACEMENT)
    levels = site_utility_system(problem_to_solver_arrays(case, 10.0)).levels

    kinds = {level.name: level.kind for level in levels.levels}
    assert kinds["iso_1"] == kinds["iso_2"] == "both"


def _site_levels(case):
    from OpenPinch.analysis.heat_exchanger_networks.site.levels import (
        site_utility_system,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )

    return site_utility_system(problem_to_solver_arrays(case, 10.0)).levels.levels


def test_a_count_cools_on_the_cases_own_utility_where_placement_fell_back():
    """With two levels, placement puts both near 140 C and cools the rest
    on its fallback CU at -30 C, priced as cooling water. The placed case
    cools on the case's own cooling water instead: no -30 C cooler at the
    cooling-water price."""
    problem = _site()

    case = site_levels.placed_case(problem, 2, placement=SHORT_PLACEMENT)

    utilities = case.to_problem_json()["utilities"]
    assert {u["name"] for u in utilities}.isdisjoint({"HU", "CU"})
    (water,) = (u for u in problem.to_problem_json()["utilities"] if u["name"] == "CW")
    assert water in utilities
    assert min(u["t_supply"]["value"] for u in utilities) == 20.0


def test_a_placed_case_keeps_the_utility_that_rejects_to_ambient(monkeypatch):
    """Every rejected heat ends at ambient (user rule, 2026-11-27). When the
    placed levels need no fallback cooler, the case dropped its cooling
    water, and a placed steam main became the coldest sink: on the
    reviewer's three-level steam main the site surplus was dumped from a
    main priced as LP steam, 41,432 against 21,348 $/y at two levels
    (review finding 15). The placement is stubbed: two placed mains."""
    from OpenPinch.application._problem.accessors import target
    from tests.application.test_site_heat_exchanger_network import _utility

    problem = _site()
    source = problem.to_problem_json()
    source["utilities"] = [
        _utility(f"{side}_iso_{level}", kind, t, t + step, 60.0)
        for level, t in ((1, 180.0), (2, 100.0))
        for side, kind, step in (("hot", "Hot", -0.01), ("cold", "Cold", 0.01))
    ]
    placed = PinchProblem(source)
    monkeypatch.setattr(
        target._TargetAccessor, "utility_placement", lambda self, **_: placed
    )
    monkeypatch.setattr(site_levels, "_fallback_utilities", lambda _placed: {})

    case = site_levels.placed_case(problem, 2)

    (water,) = (u for u in problem.to_problem_json()["utilities"] if u["name"] == "CW")
    assert water in case.to_problem_json()["utilities"]


def test_a_placed_case_keeps_each_periods_ambient_sink(monkeypatch):
    """Cooling water is at 20 C in the first period and 10 C (below the
    15 C ambient) in the second, where river water at 25 C rejects to
    ambient instead. The case keeps both (Codex review of #553)."""
    from OpenPinch.application._problem.accessors import target
    from tests.application.test_site_heat_exchanger_network import _utility

    source = _site(periods=True).to_problem_json()
    water = _utility("CW", "Cold", 20.0, 25.0, 5.0)
    water["t_supply"] = {"values": [20.0, 10.0], "unit": "degC"}
    water["t_target"] = {"values": [25.0, 15.0], "unit": "degC"}
    others = [u for u in source["utilities"] if u["name"] != "CW"]
    source["utilities"] = [*others, water, _utility("RW", "Cold", 25.0, 30.0, 6.0)]
    problem = PinchProblem(source)
    placed_source = problem.to_problem_json()
    placed_source["utilities"] = [
        _utility(f"{side}_iso_{level}", kind, t, t + step, 60.0)
        for level, t in ((1, 180.0), (2, 100.0))
        for side, kind, step in (("hot", "Hot", -0.01), ("cold", "Cold", 0.01))
    ]
    placed = PinchProblem(placed_source)
    monkeypatch.setattr(
        target._TargetAccessor, "utility_placement", lambda self, **_: placed
    )
    monkeypatch.setattr(site_levels, "_fallback_utilities", lambda _placed: {})

    case = site_levels.placed_case(problem, 2)

    names = {u["name"] for u in case.to_problem_json()["utilities"]}

    assert {"CW", "RW"} <= names


def test_a_count_heats_on_the_cases_own_utility_where_placement_fell_back():
    """With three levels, placement heats above them on its fallback HU
    at 300 C; the placed case takes the case's hottest utility, HPS."""
    problem = _site()

    case = site_levels.placed_case(problem, 3, placement=SHORT_PLACEMENT)

    utilities = case.to_problem_json()["utilities"]
    assert {u["name"] for u in utilities}.isdisjoint({"HU", "CU"})
    (steam,) = (u for u in problem.to_problem_json()["utilities"] if u["name"] == "HPS")
    assert steam in utilities


@pytest.mark.parametrize("count", [2, 3])
def test_placed_levels_closer_than_the_minimum_gap_are_one_level(count):
    """Placement puts the unit site's levels within 0.3 K of each other
    near 140 C (or 120 C): one main in effect. The case keeps one of them,
    so the site's levels lie at least the minimum gap apart. Beside it the
    case keeps its own cooling water (it rejects to ambient) and any
    stand-in for a fallback."""
    case = site_levels.placed_case(_site(), count, placement=SHORT_PLACEMENT)

    levels = _site_levels(case)
    temperatures = sorted(level.t_high[0] for level in levels)

    placed = [level for level in levels if level.name.startswith(("iso_", "sens"))]
    assert len(placed) == 1
    assert all(
        above - below >= site_levels.MINIMUM_LEVEL_GAP
        for below, above in zip(temperatures, temperatures[1:])
    )


def test_an_infeasible_count_is_recorded_and_skipped():
    designed = []

    def design(case):
        designed.append(case.count)
        return _designed(100.0 + case.count)

    runs = site_levels.run_level_counts(
        _site(), (2, 3, 4), design=design, place=_fake_place(infeasible=(3,))
    )

    assert designed == [2, 4]
    assert [run.status for run in runs] == ["designed", "infeasible", "designed"]
    assert runs[1].view is None
    assert "no placement" in runs[1].message


def test_a_count_whose_design_fails_is_recorded():
    def design(case):
        if case.count == 2:
            raise WorkflowContractError("no successful task outcomes")
        return _designed(90.0)

    runs = site_levels.run_level_counts(
        _site(), (2, 3), design=design, place=_fake_place()
    )

    assert [run.status for run in runs] == ["failed", "designed"]
    assert "no successful task outcomes" in runs[0].message


def test_the_cheapest_count_is_selected():
    costs = {2: 120.0, 3: 100.0, 4: 110.0}
    runs = site_levels.run_level_counts(
        _site(),
        (2, 3, 4),
        design=lambda case: _designed(costs[case.count]),
        place=_fake_place(),
    )

    assert site_levels.select_level_count(runs).count == 3
    assert [run.total_annual_cost for run in runs] == [120.0, 100.0, 110.0]


def test_counts_compare_on_the_site_cost_when_the_network_has_one():
    designs = {2: _designed(100.0, site_cost=150.0), 3: _designed(120.0)}
    runs = site_levels.run_level_counts(
        _site(), (2, 3), design=lambda case: designs[case.count], place=_fake_place()
    )

    assert [run.total_annual_cost for run in runs] == [150.0, 120.0]
    assert site_levels.select_level_count(runs).count == 3


def test_the_early_stop_ends_the_sweep_after_two_counts_that_cost_more():
    costs = {2: 100.0, 3: 110.0, 4: 120.0, 5: 90.0}
    designed = []

    def design(case):
        designed.append(case.count)
        return _designed(costs[case.count])

    runs = site_levels.run_level_counts(
        _site(), (2, 3, 4, 5), design=design, place=_fake_place(), early_stop=True
    )

    assert designed == [2, 3, 4]
    assert runs[-1].status == "not run"
    assert site_levels.select_level_count(runs).count == 2


def test_an_infeasible_count_does_not_count_toward_the_early_stop():
    costs = {2: 100.0, 4: 110.0, 5: 90.0}
    runs = site_levels.run_level_counts(
        _site(),
        (2, 3, 4, 5),
        design=lambda case: _designed(costs[case.count]),
        place=_fake_place(infeasible=(3,)),
        early_stop=True,
    )

    assert [run.status for run in runs] == [
        "designed",
        "infeasible",
        "designed",
        "designed",
    ]
    assert site_levels.select_level_count(runs).count == 5


def test_counts_after_the_time_limit_are_not_run():
    designed = []

    def design(case):
        designed.append(case.count)
        return _designed(100.0)

    runs = site_levels.run_level_counts(
        _site(), (2, 3, 4), design=design, place=_fake_place(), expired=lambda: True
    )

    # The first count always runs, so the sweep has a design to return.
    assert designed == [2]
    assert [(run.status, run.message) for run in runs[1:]] == [
        ("not run", "time limit"),
        ("not run", "time limit"),
    ]


def test_counts_run_until_one_is_designed_whatever_the_time():
    runs = site_levels.run_level_counts(
        _site(),
        (2, 3, 4),
        design=lambda case: _designed(100.0),
        place=_fake_place(infeasible=(2,)),
        expired=lambda: True,
    )

    assert [run.status for run in runs] == ["infeasible", "designed", "not run"]


def test_no_designed_count_raises_naming_each_count():
    runs = site_levels.run_level_counts(
        _site(), (2, 3), design=_designed, place=_fake_place(infeasible=(2, 3))
    )

    with pytest.raises(WorkflowContractError, match="2: infeasible"):
        site_levels.select_level_count(runs)


def test_given_levels_are_the_default_and_design_with_the_cases_utilities(
    monkeypatch,
):
    from OpenPinch.application._problem.accessors import design

    calls = []
    monkeypatch.setattr(
        design._DesignAccessor,
        "_site_design",
        lambda self, **kwargs: calls.append(self._problem) or "designed",
    )
    monkeypatch.setattr(site_levels, "run_level_counts", pytest.fail, raising=True)
    problem = _site()

    assert problem.design.total_site_heat_exchanger_network() == "designed"
    assert calls == [problem]


def test_placed_levels_sweep_the_configured_counts(monkeypatch):
    seen = {}

    def run(problem, counts, *, design, early_stop=False, placement=None, expired=None):
        seen.update(
            counts=counts,
            early_stop=early_stop,
            placement=placement,
            expired=None if expired is None else expired(),
        )
        raise RuntimeError("stop here")

    monkeypatch.setattr(site_levels, "run_level_counts", run)

    with pytest.raises(RuntimeError, match="stop here"):
        _site().design.total_site_heat_exchanger_network(
            levels="placed", placement={"evaluation_limit": 10}
        )
    assert seen == {
        "counts": (2, 3, 4, 5),
        "early_stop": False,
        "placement": {"evaluation_limit": 10},
        "expired": None,
    }

    with pytest.raises(RuntimeError, match="stop here"):
        _site().design.total_site_heat_exchanger_network(
            options={
                "HENS_SITE_LEVELS": "placed",
                "HENS_SITE_LEVEL_COUNTS": [3, 2],
                "HENS_SITE_LEVEL_EARLY_STOP": True,
            },
        )
    assert seen["counts"] == (2, 3)
    assert seen["early_stop"] is True

    with pytest.raises(RuntimeError, match="stop here"):
        _site().design.total_site_heat_exchanger_network(
            levels="placed", level_counts=[4]
        )
    assert seen["counts"] == (4,)

    with pytest.raises(RuntimeError, match="stop here"):
        _site().design.total_site_heat_exchanger_network(
            options={"HENS_SITE_LEVELS": "placed", "HENS_SITE_TIME_LIMIT": 3600.0}
        )
    assert seen["expired"] is False


def test_an_unknown_levels_choice_is_refused():
    with pytest.raises(ValueError, match="levels"):
        _site().design.total_site_heat_exchanger_network(levels="some")


@pytest.mark.parametrize("counts", [[1, 2], [], [2, 2]])
def test_level_counts_need_unique_values_of_at_least_two(counts):
    with pytest.raises(ValueError, match="HENS_SITE_LEVEL_COUNTS"):
        _site().design.total_site_heat_exchanger_network(
            options={"HENS_SITE_LEVEL_COUNTS": counts}
        )


def _utility_names(network) -> dict[str, set[str]]:
    """Each process's heater and cooler utilities."""
    used: dict[str, set[str]] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            process, utility = _process(exchanger.sink_stream), exchanger.source_stream
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            process, utility = _process(exchanger.source_stream), exchanger.sink_stream
        else:
            continue
        used.setdefault(process, set()).add(utility.split(".", 1)[1])
    return used


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_placed_site_network_is_the_cheapest_over_the_counts():
    problem = _site()

    view = problem.design.total_site_heat_exchanger_network(
        levels="placed",
        level_counts=[2, 3],
        placement=SHORT_PLACEMENT,
        width=0,
        depth=1,
        options={"HENS_EVM_N_AD_BRANCHES": 1},
    )

    sweep = view.result.provenance.effective_settings["HENS_SITE_LEVEL_SWEEP"]
    assert [row["count"] for row in sweep] == [2, 3]
    designed = [row for row in sweep if row["status"] == "designed"]
    assert designed
    best = min(row["total_annual_cost"] for row in designed)
    network = view.selected_network
    assert network.total_annual_cost == pytest.approx(best)
    by_count = {
        outcome.network.summary_metrics["utility_level_count"]: outcome.network
        for outcome in view.result.ranked_networks
    }
    assert sorted(by_count) == sorted(row["count"] for row in designed)
    for count, counted in by_count.items():
        placed = (
            {f"iso_{level}" for level in range(1, count + 1)}
            | {f"sensible_{level}" for level in range(1, count + 1)}
            # The case's own utilities that stand in for placement's
            # fallbacks.
            | {"HPS", "CW"}
        )
        used = _utility_names(counted)
        assert len(used) == 2
        assert set().union(*used.values()) <= placed
    assert view.result.provenance.method_id == (
        "design.total_site_heat_exchanger_network"
    )


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_the_steam_main_site_designs_with_placed_levels():
    """Each process holds two placed levels on one stream (Heating's C1
    takes heat from both, Cooling's H3 raises both), staged between its
    recovery matches. The site seed keeps room for both."""
    view = _steam_main_site().design.total_site_heat_exchanger_network(
        levels="placed",
        level_counts=[2, 3],
        placement=SHORT_PLACEMENT,
        width=0,
        depth=1,
        options={"HENS_EVM_N_AD_BRANCHES": 1, "HENS_SITE_LEVEL_STEP": 0.0},
    )

    sweep = view.result.provenance.effective_settings["HENS_SITE_LEVEL_SWEEP"]
    assert [row["status"] for row in sweep] == ["designed", "designed"]


class _Outcome(SimpleNamespace):
    def model_copy(self, *, update):
        return _Outcome(**(vars(self) | update))


class _Network(SimpleNamespace):
    def model_copy(self, *, update):
        return _Network(**(vars(self) | update))


def test_the_sweep_ranks_every_counts_networks_not_only_its_best(monkeypatch):
    """A count's level-held network that lost to its lowest-cost one stays
    among the ranked results, tagged with its count and site cost."""
    from OpenPinch.analysis.heat_exchanger_networks.results import selection
    from OpenPinch.application._problem.targeting import provenance

    def outcome(name, cost):
        network = _Network(
            name=name, total_annual_cost=cost, summary_metrics={"tag": name}
        )
        return _Outcome(network=network, objective_value=cost)

    by_result = {
        "two": (outcome("2-cheap", 30.0), outcome("2-held", 35.0)),
        "three": (outcome("3-cheap", 20.0), outcome("3-held", 25.0)),
    }
    monkeypatch.setattr(
        selection,
        "ranked_networks",
        lambda result, n=None: by_result[result.name][:n],
    )
    monkeypatch.setattr(selection, "select_network", lambda result: result)
    monkeypatch.setattr(provenance, "make_provenance", lambda *a, **k: "provenance")

    def run(count, name, cost):
        view = SimpleNamespace(result=SimpleNamespace(name=name))
        return site_levels.LevelCountRun(
            count, count, 0, "designed", view, cost, case=f"c{count}"
        )

    problem = SimpleNamespace(
        _require_prepared_root_zone=lambda: None, period_ids=(), _results=None
    )
    view = site_levels.level_sweep_view(
        problem, (run(2, "two", 30.0), run(3, "three", 20.0)), settings={}
    )

    ranked = {
        outcome.network.name: (
            outcome.network.summary_metrics["utility_level_count"],
            outcome.objective_value,
        )
        for outcome in view.result.ranked_networks
    }
    assert ranked == {
        "2-cheap": (2, 30.0),
        "2-held": (2, 35.0),
        "3-cheap": (3, 20.0),
        "3-held": (3, 25.0),
    }


def test_a_placed_sweep_does_not_become_the_problems_design(monkeypatch):
    """The placed networks name the placed case's utilities (iso_1, ...),
    which the problem lacks, so the sweep's result is returned but not
    committed as the problem's cached design for later tools to reuse."""
    from OpenPinch.analysis.heat_exchanger_networks.results import selection
    from OpenPinch.application._problem.targeting import provenance

    monkeypatch.setattr(selection, "ranked_networks", lambda result, n=None: ())
    monkeypatch.setattr(selection, "select_network", lambda result: result)
    monkeypatch.setattr(provenance, "make_provenance", lambda *a, **k: "provenance")
    view = SimpleNamespace(result=SimpleNamespace(name="two"))
    run = site_levels.LevelCountRun(2, 2, 0, "designed", view, 30.0, case="c2")
    results = SimpleNamespace(design="the problem's own design")
    problem = SimpleNamespace(
        _require_prepared_root_zone=lambda: None, period_ids=(), _results=results
    )

    assert site_levels.level_sweep_view(problem, (run,), settings={}) is view
    assert results.design == "the problem's own design"


def test_a_hot_and_a_cold_stand_in_may_share_a_name():
    """A hot and a cold utility may share a name; each side's fallback
    keeps its own."""
    source = _site().to_problem_json()
    for utility in source["utilities"]:
        if utility["name"] in {"HPS", "CW"}:
            utility["name"] = "EDGE"
    problem = PinchProblem(source)

    kept = site_levels._stand_ins(
        problem, {("HU", "Hot"): ((None, 300.0),), ("CU", "Cold"): ((None, -30.0),)}
    )

    assert sorted((u["name"], u["type"]) for u in kept) == [
        ("EDGE", "Cold"),
        ("EDGE", "Hot"),
    ]


def test_every_periods_fallback_keeps_its_nearest_utility():
    """A multiperiod placement may fall back on HU at 145 C in one period and
    at 300 C in another: the case keeps LPS (nearest 145 C) and HPS
    (nearest 300 C), not only the utility nearest the last period's."""
    from OpenPinch.contracts.utility_placement import QuantityValue, UtilitySide

    def row(supply):
        return SimpleNamespace(
            period_id=f"p{supply:g}",
            utility="HU",
            side=UtilitySide.HOT,
            supply_temperature=QuantityValue(value=supply, unit="degC"),
            is_fallback=True,
        )

    placed = SimpleNamespace(
        utility_placement_result=SimpleNamespace(
            retarget_comparison=(row(145.0), row(300.0))
        )
    )

    fallbacks = site_levels._fallback_utilities(placed)
    kept = site_levels._stand_ins(_site(), fallbacks)

    assert fallbacks == {("HU", "Hot"): (("p145", 145.0), ("p300", 300.0))}
    assert sorted(u["name"] for u in kept) == ["HPS", "LPS"]


def test_each_periods_fallback_is_matched_at_that_periods_temperatures():
    """Steam A runs at 140 C in the base period and 290 C at peak; B at
    250 C and 160 C. A fallback at 145 C in the base period and 300 C at
    peak is nearest A in both, so only A stands in: matched at the base
    period's temperatures alone, 300 C would pick B."""
    from OpenPinch.contracts.utility_placement import QuantityValue, UtilitySide
    from tests.analysis.heat_exchanger_networks.test_hen_period_handling import (
        _two_period_problem,
    )

    source = _two_period_problem().to_problem_json()
    steam, water = source["utilities"]

    def periodic(name, base, peak):
        temperatures = {"values": [base + 273.15, peak + 273.15], "unit": "K"}
        return dict(steam, name=name, t_supply=temperatures, t_target=temperatures)

    source["utilities"] = [
        periodic("A", 140.0, 290.0),
        periodic("B", 250.0, 160.0),
        water,
    ]
    problem = PinchProblem(source, project_name="two-period")

    def row(period_id, supply):
        return SimpleNamespace(
            period_id=period_id,
            utility="HU",
            side=UtilitySide.HOT,
            supply_temperature=QuantityValue(value=supply, unit="degC"),
            is_fallback=True,
        )

    placed = SimpleNamespace(
        utility_placement_result=SimpleNamespace(
            retarget_comparison=(row("base", 145.0), row("peak", 300.0))
        )
    )

    kept = site_levels._stand_ins(problem, site_levels._fallback_utilities(placed))

    assert [u["name"] for u in kept] == ["A"]
