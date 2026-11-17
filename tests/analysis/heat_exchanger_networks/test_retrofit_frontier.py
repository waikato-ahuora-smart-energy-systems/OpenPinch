"""Retrofit frontier over the number of modifications (no solver)."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.frontier import (
    mark_dominated,
)
from OpenPinch.contracts.synthesis.retrofit import (
    HeatExchangerNetworkRetrofitDesign,
    HeatExchangerNetworkRetrofitModification,
    HeatExchangerNetworkRetrofitResult,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _design(count: int, benefit: float) -> HeatExchangerNetworkRetrofitDesign:
    network = HeatExchangerNetwork.from_structure(recovery=[("H1", "C1", 1)])
    modifications = tuple(
        HeatExchangerNetworkRetrofitModification(
            kind="area_addition", exchanger_id=f"E{n}", added_area=1.0
        )
        for n in range(count)
    )
    return HeatExchangerNetworkRetrofitDesign(
        modifications=modifications,
        network=network,
        hot_utility=100.0,
        cold_utility=100.0,
        utility_cost=1000.0 - benefit,
        annual_savings=benefit,
        capital_cost=0.0,
        annualised_capital_cost=0.0,
        net_annual_benefit=benefit,
        payback_years=None,
    )


def test_a_design_no_better_than_one_with_fewer_modifications_is_dominated():
    frontier = mark_dominated(
        [_design(0, 0.0), _design(1, 500.0), _design(2, 400.0), _design(3, 600.0)]
    )

    assert [d.modification_count for d in frontier] == [0, 1, 2, 3]
    assert [d.dominated for d in frontier] == [False, False, True, False]


def test_the_selected_design_has_the_best_net_benefit_with_fewest_modifications():
    result = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(0, 0.0), _design(1, 500.0), _design(2, 500.0))
    )

    assert result.selected.modification_count == 1


def test_dominance_follows_frontier_steps_not_modification_records():
    # In the structure search frontier[n] is n structural changes, each
    # with any area additions it keeps: step 1 here carries three records,
    # step 2 one. Step 2 earns less than step 1 does, so it is dominated;
    # step 1 is not, though step 2 has fewer records.
    frontier = mark_dominated([_design(0, 0.0), _design(3, 500.0), _design(1, 400.0)])

    assert [d.dominated for d in frontier] == [False, False, True]


def test_the_selected_design_breaks_ties_by_fewest_steps():
    result = HeatExchangerNetworkRetrofitResult(
        frontier=(_design(0, 0.0), _design(3, 500.0), _design(1, 500.0))
    )

    assert result.selected is result.frontier[1]
