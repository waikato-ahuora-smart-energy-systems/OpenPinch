"""A multiperiod design's totals are the year's, by the periods' weights.

``total_heat_recovery``, ``total_hot_utility`` and ``total_cold_utility``
summed one period's duties and raised on every design over several
periods ("period_id is required when a network has multiple period
states"), though a property takes no period. They are now the mean over
the periods, weighted by each period's share of the year.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application._problem.accessors.design import (
    HeatExchangerNetworkDesignView,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

# exchanger id -> duty (kW) in the base and peak periods
_DUTIES = {
    "recovery:H1->C1:S1": (300.0, 500.0),
    "hot-utility:HU->C1": (100.0, 400.0),
    "cold-utility:H1->CU": (200.0, 0.0),
}


def _two_period_network() -> HeatExchangerNetwork:
    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=["C1"],
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )

    def states(exchanger):
        return exchanger.model_copy(
            update={
                "period_states": tuple(
                    HeatExchangerPeriodState(
                        period_id=period, period_idx=index, duty=duty, active=duty > 0
                    )
                    for index, (period, duty) in enumerate(
                        zip(("base", "peak"), _DUTIES[exchanger.exchanger_id])
                    )
                )
            }
        )

    return structure.model_copy(
        update={"exchangers": tuple(states(e) for e in structure.exchangers)}
    )


def test_a_multiperiod_designs_totals_are_weighted_by_period():
    view = HeatExchangerNetworkDesignView(
        SimpleNamespace(network=_two_period_network()),
        period_weights={"base": 0.75, "peak": 0.25},
    )

    assert view.total_heat_recovery == pytest.approx(0.75 * 300.0 + 0.25 * 500.0)
    assert view.total_hot_utility == pytest.approx(0.75 * 100.0 + 0.25 * 400.0)
    assert view.total_cold_utility == pytest.approx(0.75 * 200.0)


def test_one_weight_given_for_every_period_weighs_them_all():
    # PROBLEM_PERIOD_WEIGHTS of one value holds for every period.
    import json

    from OpenPinch import PinchProblem
    from OpenPinch.resources import read_sample_case

    source = json.loads(read_sample_case("crude_preheat_train_multiperiod.json"))
    source["options"]["PROBLEM_PERIOD_WEIGHTS"] = [0.25]
    problem = PinchProblem(source, project_name="crude")

    weights = problem.design._period_weights()

    assert list(weights) == ["turndown", "base", "peak"]
    assert len(set(weights.values())) == 1


@pytest.mark.synthesis
@pytest.mark.solver
def test_a_designs_view_takes_the_problems_period_weights():
    import json

    from OpenPinch import PinchProblem
    from OpenPinch.domain.enums import HeatExchangerKind
    from OpenPinch.resources import read_sample_case

    # Turndown, base and peak at 0.2, 0.6 and 0.2 of the year.
    problem = PinchProblem(
        json.loads(read_sample_case("crude_preheat_train_multiperiod.json")),
        project_name="crude",
    )
    view = problem.design.heat_exchanger_network(width=0, depth=0)

    network = view.selected_network
    weighted = sum(
        weight
        * network.total_duty(kind=HeatExchangerKind.HOT_UTILITY, period_id=period)
        for weight, period in zip((0.2, 0.6, 0.2), network.period_ids)
    )
    assert view.total_hot_utility == pytest.approx(weighted)
