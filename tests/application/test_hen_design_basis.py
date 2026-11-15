"""Designing a multiperiod HEN from one basis period, then fitting all periods.

A basis period is a derived single-period case: synthesis runs on it
unchanged, and the fit gives the network one area per exchanger over every
operating period, adding a heater or cooler at a stream end where a period
needs one.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.case_derivation import single_period_case
from tests.analysis.heat_exchanger_networks.test_hen_period_handling import (
    _two_period_problem,
)


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _stream_values(problem, key: str) -> dict[str, float]:
    return {
        record["name"]: record[key].get("value", None)
        for record in problem.to_problem_json()["streams"]
    }


def test_a_single_period_case_holds_that_periods_data_alone():
    problem = _two_period_problem()

    peak = single_period_case(problem, "peak")

    assert list(peak.period_ids) == ["peak"]
    assert _stream_values(peak, "heat_flow")["Raw Milk"] == pytest.approx(3500.0)
    assert _stream_values(peak, "heat_capacity_flowrate")["Raw Milk"] == (
        pytest.approx(12.5)
    )


def test_a_single_period_case_has_that_periods_targets():
    problem = _two_period_problem()
    by_period = problem.target.all_periods.direct_heat_integration()

    for period_id in ("base", "peak"):
        derived = single_period_case(problem, period_id)
        derived.target.direct_heat_integration()
        expected = by_period[period_id].targets[0]
        actual = derived.results.targets[0]
        assert actual.Qh.value == pytest.approx(expected.Qh.value, rel=1e-9)
        assert actual.Qc.value == pytest.approx(expected.Qc.value, rel=1e-9)


def test_a_single_period_case_rejects_an_unknown_period():
    with pytest.raises(ValueError, match="unknown period 'night'"):
        single_period_case(_two_period_problem(), "night")


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_basis_period_design_fitted_to_all_periods_closes_every_balance():
    problem = _two_period_problem()
    basis = single_period_case(problem, "base")
    network = basis.design.heat_exchanger_network(width=0, depth=0).selected_network

    fitted = problem.design._fit_to_all_periods(network).selected_network

    heat_flows = {
        f"{record['zone'].rsplit('/', 1)[-1]}.{record['name']}": record["heat_flow"][
            "values"
        ]
        for record in problem.to_problem_json()["streams"]
    }
    for period, period_id in enumerate(("base", "peak")):
        duty = dict.fromkeys(heat_flows, 0.0)
        for exchanger in fitted.exchangers:
            state = next(s for s in exchanger.period_states if s.period_id == period_id)
            for stream in (exchanger.source_stream, exchanger.sink_stream):
                if stream in duty:
                    duty[stream] += state.duty
        for stream, values in heat_flows.items():
            assert duty[stream] == pytest.approx(values[period], rel=1e-4), (
                period_id,
                stream,
            )
    # One area per exchanger, for both periods.
    assert all(exchanger.area is not None for exchanger in fitted.exchangers)
