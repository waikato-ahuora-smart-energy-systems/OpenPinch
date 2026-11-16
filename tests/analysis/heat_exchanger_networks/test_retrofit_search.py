"""Retrofit search mechanics on a fake fixed-structure solver (no solver).

The fake keeps the real request building, structure and screen, and stands
in for the NLP only: a design's utility cost falls by $100,000/y per
recovery match and $20,000/y per enlarged unit, every installed unit needs
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
        def __init__(self, *, spec, **_settings) -> None:
            self.spec = spec

        def optimise(self, print_output: bool) -> None:
            backend.solved += 1
            spec = self.spec
            enlarged = len(spec.enlargeable_recovery) + len(spec.enlargeable_utilities)
            self.utility_cost_value = (
                1_000_000.0 - 100_000.0 * len(spec.recovery) - 20_000.0 * enlarged
            )
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
        RetrofitSettings(max_modifications=2, new_match_candidates=4, **COSTS),
    )

    kinds = {m.kind for d in result.frontier for m in d.modifications}
    assert kinds == {"area_addition"}
    assert result.evaluated_designs > len(result.frontier)


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
