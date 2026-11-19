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
    placed levels, each serving both heating and cooling."""
    problem = _site()
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
        names = {u["name"] for u in source["utilities"]} - {"HU", "CU"}
        assert names == {
            f"{side}_{kind}_{level}"
            for side in ("hot", "cold")
            for kind, levels in (("iso", isothermal), ("sensible", sensible))
            for level in range(1, levels + 1)
        }


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

    def run(problem, counts, *, design, early_stop=False, placement=None):
        seen.update(counts=counts, early_stop=early_stop, placement=placement)
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
        placed = {
            f"{side}_iso_{level}"
            for side in ("hot", "cold")
            for level in range(1, count + 1)
        } | {"HU", "CU"}
        used = _utility_names(counted)
        assert len(used) == 2
        assert set().union(*used.values()) <= placed
    assert view.result.provenance.method_id == (
        "design.total_site_heat_exchanger_network"
    )
