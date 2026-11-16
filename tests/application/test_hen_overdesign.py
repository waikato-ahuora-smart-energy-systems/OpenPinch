"""Overdesign: larger heat-recovery areas, rated over every period.

A margin scales every recovery exchanger's area; a rating solve then keeps
those areas fixed and re-optimises each period's duties for the least
period-weighted utility cost, with bypass where a period needs less.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
    build_duty_optimisation_request,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.domain.enums import HeatExchangerKind
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.application.test_hen_design_basis import _two_period_problem


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _recovery_areas(network) -> dict[str, float]:
    return {
        exchanger.exchanger_id: exchanger.area
        for exchanger in network.exchangers
        if exchanger.kind == HeatExchangerKind.RECOVERY
    }


def _utility_by_period(network) -> dict[str, float]:
    totals: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind == HeatExchangerKind.RECOVERY:
            continue
        for state in exchanger.period_states:
            totals[state.period_id] = totals.get(state.period_id, 0.0) + state.duty
    return totals


@pytest.fixture(scope="module")
def fitted():
    """A two-period network designed on its base period, fitted to both."""
    problem = _two_period_problem()
    view = problem.design.heat_exchanger_network(width=0, depth=0, period_id="base")
    return problem, view.selected_network


def test_rating_needs_an_area_on_every_exchanger():
    network = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)], heaters=[("C1", "HU")], coolers=[("H1", "CU")]
    )
    # Heaters and coolers keep their installed areas too.
    recovery_only = network.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(update={"area": 10.0})
                if exchanger.kind == HeatExchangerKind.RECOVERY
                else exchanger
                for exchanger in network.exchangers
            )
        }
    )

    for unrated in (network, recovery_only):
        with pytest.raises(
            ValueError, match="'rating' objective needs an installed area"
        ):
            build_duty_optimisation_request(unrated, objective="rating")


@pytest.mark.parametrize("margin", [-5.0, "maximise"])
def test_an_invalid_overdesign_is_rejected(margin):
    with pytest.raises(ValueError, match="overdesign_percentage"):
        _two_period_problem().design.heat_exchanger_network(
            overdesign_percentage=margin
        )


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_rating_at_the_installed_areas_keeps_the_design(fitted):
    problem, network = fitted

    rated = problem.design.optimise_duties(network, objective="rating").selected_network

    assert _recovery_areas(rated) == pytest.approx(_recovery_areas(network), rel=1e-9)
    for period, utility in _utility_by_period(network).items():
        assert _utility_by_period(rated)[period] == pytest.approx(
            utility, rel=1e-3, abs=1.0
        )


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_fixed_margin_scales_every_recovery_area_and_raises_no_utility(fitted):
    problem, network = fitted

    margined = problem.design._apply_overdesign(network, 20.0).selected_network

    base_areas = _recovery_areas(network)
    for exchanger_id, area in _recovery_areas(margined).items():
        assert area == pytest.approx(1.2 * base_areas[exchanger_id], rel=1e-9)
    for exchanger in margined.exchangers:
        if exchanger.kind != HeatExchangerKind.RECOVERY:
            base = next(
                e
                for e in network.exchangers
                if e.exchanger_id == exchanger.exchanger_id
            )
            assert exchanger.area == pytest.approx(base.area, rel=1e-9)
    base_utility = _utility_by_period(network)
    for period, utility in _utility_by_period(margined).items():
        assert utility <= base_utility[period] + 1e-3 * max(base_utility[period], 1.0)
    metrics = margined.summary_metrics["overdesign"]
    assert metrics["mode"] == "fixed" and metrics["percentage"] == 20.0
    assert set(metrics["periods"]) == {"base", "peak"}


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_an_optimised_margin_costs_no_more_than_none_or_the_maximum(fitted):
    problem, network = fitted

    optimised = problem.design._apply_overdesign(network, "optimise").selected_network

    metrics = optimised.summary_metrics["overdesign"]
    costs = dict(metrics["search"])
    assert 0.0 <= metrics["percentage"] <= 50.0
    assert metrics["cost"] <= costs[0.0] * (1 + 1e-6)
    assert metrics["cost"] <= costs[50.0] * (1 + 1e-6)


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_cost_optimal_single_period_network_needs_no_margin():
    from OpenPinch.application.case_derivation import single_period_case

    base = single_period_case(_two_period_problem(), "base")
    network = base.design.heat_exchanger_network(width=0, depth=0).selected_network
    optimal = base.design.optimise_duties(network, objective="cost").selected_network

    margined = base.design._apply_overdesign(optimal, "optimise").selected_network

    assert margined.summary_metrics["overdesign"]["percentage"] == pytest.approx(
        0.0, abs=1.0
    )
