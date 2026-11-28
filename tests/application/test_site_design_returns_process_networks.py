"""A site design whose evolution fails returns its process networks.

Each process is designed alone before the site evolution refines them as
one problem. When every evolution task failed, the design raised and the
verified process networks were thrown away. They are now joined into one
site network each, costed by the site utility balance of their own
utility use, and returned with a warning.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.errors import WorkflowContractError
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.application._problem.accessors import design
from OpenPinch.domain.enums import HeatExchangerKind
from tests.application import test_site_heat_exchanger_network as site

pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.solver,
    pytest.mark.skipif(not site._solvers_available(), reason="needs highspy and ipopt"),
]


def _failing_evolution(self, seeds, **arguments):
    raise WorkflowContractError(
        "heat exchanger network synthesis produced no successful task "
        "outcomes. Failed task details: network_evolution_method(dt_cont "
        "x1): solver did not return a successful heat exchanger network model"
    )


def test_a_site_design_whose_evolution_fails_returns_the_joined_process_networks(
    monkeypatch,
):
    monkeypatch.setattr(design._DesignAccessor, "network_evolution", _failing_evolution)

    with pytest.warns(UserWarning, match="site evolution failed"):
        view = site._site().design.total_site_heat_exchanger_network(width=0, depth=1)

    network = view.selected_network
    assert network.method == "site_pinch_design"
    assert verify_network_feasibility(network) == ()
    processes = {
        site._process(
            exchanger.sink_stream
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY
            else exchanger.source_stream
        )
        for exchanger in network.exchangers
    }
    assert processes == {"Heating", "Cooling"}
    balance = network.source_metadata["site_utility_balance"]
    for row in balance.balances:
        inflow = row.generation + row.letdown_in + row.imported
        outflow = row.use + row.letdown_out + row.dump
        assert inflow == pytest.approx(outflow, abs=1e-6)
    assert network.total_annual_cost == pytest.approx(
        network.capital_cost + balance.operating_cost
    )
    assert network.summary_metrics["site_total_annual_cost"] == pytest.approx(
        network.total_annual_cost
    )
