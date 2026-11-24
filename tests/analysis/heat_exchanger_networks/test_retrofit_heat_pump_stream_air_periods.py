"""Air is matched to an evaporator only where it holds in every period.

The air is at the ambient temperature in every period, but an evaporator
of a shared design can sit at a different temperature in each. Air at
15 degC is matched, at an ambient approach of 10 K, to an evaporator at
5 degC in the base only if the evaporator stays 10 K below it in every
period both run: at 8 degC in the peak it does not.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as periods,
)


def _stream(name: str, cold_end: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=cold_end,
        target_temperature=cold_end + 0.01,
        heat_flow=1.0,
    )


def _air() -> Stream:
    return Stream(
        name="AIR_H1", supply_temperature=15.01, target_temperature=15.0, heat_flow=1.0
    )


def _air_matches(evaporator_in_periods, air_in_periods) -> set:
    _network, added = with_heat_pump_matches(
        periods.two_period_network(),
        condensers=[],
        evaporators=[evaporator_in_periods[0] or evaporator_in_periods[1]],
        approach=10.0,
        ambient_sources=[_air()],
        ambient_approach=10.0,
        in_periods={"HP_C1": evaporator_in_periods, "AIR_H1": air_in_periods},
    )
    return {(p.match.hot, p.match.cold) for p in added if p.match.hot == "AIR_H1"}


def test_air_is_not_matched_where_a_period_breaks_its_approach():
    evaporator = (_stream("HP_C1", 5.0), _stream("HP_C1", 8.0))

    assert _air_matches(evaporator, (_air(), _air())) == set()


def test_air_is_not_matched_to_an_evaporator_it_never_runs_with():
    # Air drawn in the peak alone and an evaporator running in the base
    # alone share no period: a match between them would never carry duty.
    evaporator = (_stream("HP_C1", 5.0), None)

    assert _air_matches(evaporator, (None, _air())) == set()
