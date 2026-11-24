"""A heat pump match of several periods holds its approach in each.

A shared heat pump design runs its levels in every period, and the
existing network's coolers and heaters start at each period's own
temperature. A level is matched to a cooler or heater only where the
approach holds in every period both run.

In the base H1's cooler starts at 136.5 degC and in the peak at 122 degC:
an evaporator at 115 degC is 21.5 K below it in the base but 7 K in the
peak, short of a 10 K approach, so it is matched to H2 (169.6 degC) alone.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as periods,
)


def _evaporator() -> Stream:
    return Stream(
        name="HP_C1", supply_temperature=114.99, target_temperature=115.0, heat_flow=1.0
    )


def test_a_level_is_matched_only_where_its_approach_holds_in_every_period():
    _network, added = with_heat_pump_matches(
        periods.two_period_network(),
        condensers=[],
        evaporators=[_evaporator()],
        approach=10.0,
        in_periods={"HP_C1": (_evaporator(), _evaporator())},
    )

    assert {(p.match.hot, p.match.cold) for p in added} == {("H2", "HP_C1")}
