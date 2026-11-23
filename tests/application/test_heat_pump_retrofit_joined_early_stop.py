"""Joining searches that stopped early.

A retrofit search that runs out of candidates before ``max_modifications``
records the failed solves of that last, empty depth too, so its
``failed_solves`` can be as long as its frontier. Joined, the extra entry
is kept at its step and nothing is indexed past the end.
"""

from __future__ import annotations

from OpenPinch.application.heat_pump_retrofit import joined
from OpenPinch.contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitResult
from tests.application.test_heat_pump_retrofit_joined_frontier import _design


def test_searches_that_stopped_early_join_without_error():
    # Both searches tried a depth that gave no design.
    base = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(0.0), _design(100.0), _design(150.0)),
        failed_solves=(1, 2, 3),
    )
    pumped = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(120.0), _design(130.0)),
        failed_solves=(4, 5),
    )

    result = joined(base, pumped)

    assert [d.net_annual_benefit for d in result.frontier] == [0.0, 120.0, 150.0]
    # Step 1: base 1 and the heat pump step itself (no failures); step 2:
    # base 2 and pumped 4; step 3: base 3 and pumped 5.
    assert result.failed_solves == (1, 6, 8)
