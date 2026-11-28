"""A site whose processes recover no heat within themselves is designed.

Its processes are linked only through the site's utilities: each process's
network is heaters or coolers alone, so there is nothing for the site
evolution to refine, and the site design raised "no process network
recovers heat" (review finding 33). It now returns the process networks
joined into one site network, costed by the site utility balance, as when
the site evolution fails. Each process here has one side only, so no
synthesis solver runs.
"""

from __future__ import annotations

import pytest

from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind
from tests.application import test_site_heat_exchanger_network as site


def _utility_linked_site() -> PinchProblem:
    """Heating has only C1 to heat; Cooling has only H3 to cool."""
    source = site._site().to_problem_json()
    source["streams"] = [
        stream for stream in source["streams"] if stream["name"] in {"C1", "H3"}
    ]
    return PinchProblem(source)


def test_a_site_linked_only_by_utilities_returns_its_process_networks_joined():
    with pytest.warns(UserWarning, match="no process recovers heat"):
        view = _utility_linked_site().design.total_site_heat_exchanger_network(
            width=0, depth=0
        )

    network = view.selected_network
    assert network.method == "site_pinch_design"
    assert {e.kind for e in network.exchangers} <= {
        HeatExchangerKind.HOT_UTILITY,
        HeatExchangerKind.COLD_UTILITY,
    }
    assert "site_utility_balance" in network.source_metadata
