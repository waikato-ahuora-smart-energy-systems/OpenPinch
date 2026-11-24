"""Two retrofit searches give one frontier.

A retrofit result holds the best design after each number of steps. The
search without the heat pump and the one started from it each have such
a series, the heat pump step being the second one's first step. Joined,
each number of steps keeps the design with the better net annual
benefit, the existing network stays at step 0, the candidates of both
are counted, and the failed solves are added step by step.
"""

from __future__ import annotations

from OpenPinch.application.heat_pump_retrofit import joined
from OpenPinch.contracts.synthesis.retrofit import (
    HeatExchangerNetworkRetrofitDesign,
    HeatExchangerNetworkRetrofitResult,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _design(benefit: float) -> HeatExchangerNetworkRetrofitDesign:
    return HeatExchangerNetworkRetrofitDesign(
        modifications=(),
        network=HeatExchangerNetwork(),
        hot_utility=0.0,
        cold_utility=0.0,
        utility_cost=0.0,
        annual_savings=benefit,
        capital_cost=0.0,
        annualised_capital_cost=0.0,
        net_annual_benefit=benefit,
        payback_years=None,
    )


def test_each_step_keeps_the_better_of_the_two_searches_designs():
    base = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(0.0), _design(100.0), _design(150.0)),
        evaluated_designs=10,
        failed_solves=(1, 2),
    )
    # Started from the heat pump: its first design is one step in.
    pumped = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(120.0), _design(140.0), _design(170.0)),
        evaluated_designs=7,
        failed_solves=(3, 4),
    )

    result = joined(base, pumped)

    assert [d.net_annual_benefit for d in result.frontier] == [0.0, 120.0, 150.0, 170.0]
    assert [d.dominated for d in result.frontier] == [False, False, False, False]
    assert result.evaluated_designs == 17
    assert result.failed_solves == (1, 5, 4)
