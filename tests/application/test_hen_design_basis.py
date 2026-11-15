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


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"design_basis": "all", "period_id": "base"}, "period_id"),
        ({"design_basis": "period"}, "needs a period_id"),
        ({"design_basis": "busiest"}, "design_basis"),
    ],
)
def test_a_contradictory_or_unknown_design_basis_is_rejected(arguments, message):
    with pytest.raises(ValueError, match=message):
        _two_period_problem().design.heat_exchanger_network(**arguments)


def test_the_all_periods_basis_targets_every_period_first(monkeypatch):
    # The fake executor has no multiperiod networks, so stop at the dispatch.
    from OpenPinch.analysis.heat_exchanger_networks import service

    class Dispatched(Exception):
        pass

    def stop(problem, settings, **kwargs):
        raise Dispatched

    monkeypatch.setattr(service, "execute_open_hens_method", stop)
    problem = _two_period_problem()

    with pytest.raises(Dispatched):
        problem.design.heat_exchanger_network(design_basis="all")

    assert set(problem.period_results) == {"base", "peak"}


def test_the_multiperiod_method_is_gone():
    assert not hasattr(
        _two_period_problem().design, "multiperiod_heat_exchanger_network"
    )


def test_a_saved_multiperiod_design_reads_as_the_all_periods_basis():
    from OpenPinch.domain.analysis import AnalysisProvenance

    provenance = AnalysisProvenance(
        owner_id="owner",
        method_id="design.multiperiod_heat_exchanger_network",
        zone_address="Site",
        period_ids=("base", "peak"),
        input_fingerprint="0" * 64,
        effective_settings_json='{"HENS_SEARCH_DEPTH": 1}',
    )

    assert provenance.method_id == "design.heat_exchanger_network"
    assert provenance.effective_settings["HENS_DESIGN_BASIS"] == "all"
    assert provenance.effective_settings["HENS_SEARCH_DEPTH"] == 1


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_period_id_designs_on_that_period_and_fits_every_period():
    problem = _two_period_problem()

    view = problem.design.heat_exchanger_network(width=0, depth=0, period_id="peak")

    provenance = view.result.provenance
    assert provenance.method_id == "design.heat_exchanger_network"
    assert provenance.period_ids == ("base", "peak")
    assert provenance.effective_settings["HENS_DESIGN_BASIS"] == "period"
    assert provenance.effective_settings["basis_period_id"] == "peak"
    assert problem.results.design.provenance == provenance
    periods = {
        state.period_id
        for exchanger in view.selected_network.exchangers
        for state in exchanger.period_states
    }
    assert periods == {"base", "peak"}
