"""The searches of several heat pump sizes join into one frontier.

A retrofit that tries the heat pump at several sizes runs one search per
size, each started from its own heat pump step. Each number of steps
keeps the best design of the plain search and every size's, and the
evaluated designs and failed solves of all of them are added.
"""

from __future__ import annotations

from OpenPinch.application.heat_pump_retrofit import joined
from OpenPinch.contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitResult
from tests.application.test_heat_pump_retrofit_joined_frontier import _design


def test_each_step_keeps_the_best_of_every_size():
    base = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(0.0), _design(100.0), _design(150.0)),
        evaluated_designs=10,
        failed_solves=(1, 2),
    )
    full = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(90.0), _design(160.0)),
        evaluated_designs=5,
        failed_solves=(3,),
    )
    half = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(120.0), _design(140.0), _design(175.0)),
        evaluated_designs=7,
        failed_solves=(4, 5),
    )

    result = joined(base, full, half)

    assert [d.net_annual_benefit for d in result.frontier] == [
        0.0,
        120.0,
        160.0,
        175.0,
    ]
    assert result.evaluated_designs == 22
    assert result.failed_solves == (1, 9, 5)
