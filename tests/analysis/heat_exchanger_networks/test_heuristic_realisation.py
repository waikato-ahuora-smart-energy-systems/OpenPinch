"""Realising heuristic pinch design proposals with the pinch side models.

Each side proposal is solved with only its matches allowed, the sides are
amalgamated, stage-reduced and verified, and a proposal the side model cannot
solve is dropped, not raised.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heuristic.enumeration import (
    SideProposal,
)
from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    nominal_approach_temperature,
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind
from tests.analysis.heat_exchanger_networks.test_segmented_streams import (
    _segmented_problem,
)
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)

pytestmark = pytest.mark.synthesis


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _problem(pinch: PinchProblem, dTmin: float, limit: int):
    return InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(pinch, dTmin),
        framework="PDM",
        solver="highs",
        dTmin=dTmin,
        pinch_decompositions={
            side: build_pinch_design_decomposition(pinch, dTmin, pinch_location=side)
            for side in ("above", "below")
        },
        heuristic_network_limit=limit,
    )


def _four_stream() -> PinchProblem:
    return PinchProblem(
        source=json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8")),
        project_name="Four-stream",
    )


def _networks(problem):
    return [problem.extract_network(run_id="heuristic")] + [
        variant.extract_network(run_id="heuristic") for variant in problem.pdm_variants
    ]


def _hot_utility(network) -> float:
    return sum(
        max(state.duty for state in exchanger.period_states)
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY
    )


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_four_stream_proposal_realises_the_textbook_network_at_target():
    problem = _problem(_four_stream(), 14.0, limit=10)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    (network,) = _networks(problem)
    assert verify_network_feasibility(network) == ()
    # The four-stream hot utility target at 14 K, with two recovery matches
    # below the pinch and one above.
    assert _hot_utility(network) == pytest.approx(510.0, rel=1e-4)
    recovery = [e for e in network.exchangers if e.kind is HeatExchangerKind.RECOVERY]
    assert len(recovery) == 3
    assert problem.heuristic_report["networks"] == 1


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_alternative_proposals_realise_as_distinct_verified_networks():
    pinch = PinchProblem("basic_pinch.json")
    pinch.target.all_heat_integration()
    dTmin = nominal_approach_temperature(pinch.master_zone.config, 1.0)
    problem = _problem(pinch, dTmin, limit=10)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    networks = _networks(problem)
    assert len(networks) >= 2
    assert len({network_structure_signature(n) for n in networks}) == len(networks)
    target = problem.above.HU_target_by_period[0]
    for variant in [problem, *problem.pdm_variants]:
        assert variant.case.verify()[0]
    for network in networks:
        assert verify_network_feasibility(network) == ()
        assert _hot_utility(network) == pytest.approx(target, rel=1e-4)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_proposal_the_side_model_cannot_solve_is_dropped():
    problem = _problem(_four_stream(), 14.0, limit=10)
    problem.load_model()
    # Above the pinch the hot stream has no cooler, so it needs a match.
    heater_only = SideProposal(
        location="above",
        stages=1,
        matches=(),
        heaters=((0, 0),),
        coolers=(),
        complete=False,
    )

    assert problem._realise_side("above", heater_only, 0) is None


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_dropped_proposal_is_not_logged_as_an_error(caplog):
    # Proposals the side model cannot solve are dropped by design in a normal
    # run, so they are not errors to the user.
    problem = _problem(_four_stream(), 14.0, limit=10)
    problem.load_model()
    heater_only = SideProposal(
        location="above",
        stages=1,
        matches=(),
        heaters=((0, 0),),
        coolers=(),
        complete=False,
    )

    with caplog.at_level("DEBUG"):
        problem._realise_side("above", heater_only, 0)

    failed = [record for record in caplog.records if "[Failed]" in record.message]
    assert failed and all(record.levelname == "DEBUG" for record in failed)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_segmented_streams_realise_through_the_piecewise_side_model():
    problem = _problem(_segmented_problem(cold_second_duty=200.0), 10.0, limit=5)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    for network in _networks(problem):
        assert verify_network_feasibility(network) == ()
        assert _hot_utility(network) == pytest.approx(100.0, rel=1e-4)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_multiperiod_proposals_share_one_topology_across_periods():
    pinch = PinchProblem("crude_preheat_train_multiperiod.json")
    pinch.target.all_heat_integration()
    dTmin = nominal_approach_temperature(pinch.master_zone.config, 1.0)
    problem = _problem(pinch, dTmin, limit=4)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    for network in _networks(problem):
        assert verify_network_feasibility(network) == ()
        assert all(len(e.period_states) == 3 for e in network.exchangers)


def _fixture(name: str) -> PinchProblem:
    payload = json.loads(
        (REPOSITORY_ROOT / f"tests/fixtures/openhens/{name}-1.json").read_text(
            encoding="utf-8"
        )
    )
    return PinchProblem(source=payload, project_name=name)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
@pytest.mark.parametrize(
    "name",
    [
        # One cold stream at the pinch above it splits three ways: the split
        # branches mix at one temperature, so each match also needs the next
        # stage out.
        "Six-stream-Yee-and-Grossmann-1990",
        # A hot stream just above the pinch needs the coldest cold slices,
        # which a hot stream far above it took first.
        "Six-stream-Spray-Dryer-2025",
    ],
)
def test_the_heuristic_realises_networks_on_literature_cases(name):
    pinch = _fixture(name)
    dTmin = nominal_approach_temperature(pinch.master_zone.config, 1.0)
    problem = _problem(pinch, dTmin, limit=5)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    assert all(count > 0 for count in problem.heuristic_report["realised"].values())
    for network in _networks(problem):
        assert verify_network_feasibility(network) == ()


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_every_heuristic_network_holds_the_utility_targets():
    # The first network of a task was stage-reduced a second time, by its
    # share of recovery, and missed the hot utility target by 801 kW on
    # Ten-stream Ahmad at multiplier 1.5; the task's other networks hit it.
    pinch = _fixture("Ten-stream-Ahmad-1985")
    dTmin = nominal_approach_temperature(pinch.master_zone.config, 1.5)
    problem = _problem(pinch, dTmin, limit=10)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    hot = [_hot_utility(network) for network in _networks(problem)]
    assert max(hot) == pytest.approx(min(hot), rel=1e-6)
