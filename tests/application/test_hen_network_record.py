"""Networks enter the record in their final form on fitted and rated designs."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.results.record import (
    HeatExchangerNetworkRecord,
)
from tests.application.test_hen_design_basis import _available, _periods_problem

_SOLVERS = _available("highs") and _available("ipopt-pyomo")


def _assert_result_is_the_record(view, record):
    assert record.status == "complete"
    assert view.result.ranked_networks == record.outcomes()
    assert view.selected_network == record.outcomes()[0].network
    costs = [entry["total_annual_cost"] for entry in record.snapshot()["entries"]]
    assert costs == sorted(costs)


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_a_median_basis_record_holds_every_network_fitted_to_every_period():
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])
    record = HeatExchangerNetworkRecord()

    view = problem.design.heat_exchanger_network(width=0, depth=1, record=record)

    _assert_result_is_the_record(view, record)
    assert len(record.outcomes()) > 1
    for outcome in record.outcomes():
        assert outcome.network.period_ids == ("p0", "p1", "p2")
    assert view.result.provenance.effective_settings["HENS_DESIGN_BASIS"] == "median"
    assert problem.results.design == view.result


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_an_overdesign_record_holds_every_network_rated_at_the_margin():
    problem = _periods_problem([1.0, 1.2], [0.5, 0.5])
    record = HeatExchangerNetworkRecord()

    view = problem.design.heat_exchanger_network(
        width=0,
        depth=0,
        design_basis="all",
        overdesign_percentage=10.0,
        record=record,
    )

    _assert_result_is_the_record(view, record)
    for outcome in record.outcomes():
        assert outcome.network.summary_metrics["overdesign"]["percentage"] == 10.0
    settings = view.result.provenance.effective_settings
    assert settings["HENS_OVERDESIGN_PERCENTAGE"] == 10.0
