"""A cooler off in a period does not hold a heat pump match back there.

A solved network can hold a cooler that is off in a period as no duty,
inactive, with its temperatures unset. A level is matched where its
approach holds in every period the cooler runs: here H1's cooler, off in
the peak, is matched in the base alone, 21.5 K above the evaporator.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_inactive_period as inactive,
)
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as periods,
)


def _evaporator() -> Stream:
    return Stream(
        name="HP_C1", supply_temperature=114.99, target_temperature=115.0, heat_flow=1.0
    )


def test_a_cooler_off_in_a_period_is_checked_where_it_runs():
    network = inactive._with_h1_cooler_off_in_the_peak(periods.two_period_network())

    _network, added = with_heat_pump_matches(
        network,
        condensers=[],
        evaporators=[_evaporator()],
        approach=10.0,
        in_periods={"HP_C1": (_evaporator(), _evaporator())},
    )

    assert {(p.match.hot, p.match.cold) for p in added} == {
        ("H1", "HP_C1"),
        ("H2", "HP_C1"),
    }


def test_a_level_is_not_matched_to_a_cooler_it_never_runs_with():
    # H1's cooler is off in the peak, and this evaporator runs in the peak
    # alone: a match between them would never carry duty.
    network = inactive._with_h1_cooler_off_in_the_peak(periods.two_period_network())

    _network, added = with_heat_pump_matches(
        network,
        condensers=[],
        evaporators=[_evaporator()],
        approach=10.0,
        in_periods={"HP_C1": (None, _evaporator())},
    )

    assert {(p.match.hot, p.match.cold) for p in added} == {("H2", "HP_C1")}
