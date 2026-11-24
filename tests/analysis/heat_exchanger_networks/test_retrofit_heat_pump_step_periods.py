"""A heat pump step of several periods places its matches for each.

The step holds the machine's levels in every period it runs, and the
search's first design matches each level only where its approach holds in
all of them. On the two-period network H1's cooler starts at 136.5 degC
in the base and 122 degC in the peak: an evaporator at 115 degC, 11 K
from it, fails in the peak and is matched to H2 alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    HeatPumpStep,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    _heat_pump_root,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as periods,
)


def _evaporator() -> Stream:
    return Stream(
        name="HP_C1", supply_temperature=114.99, target_temperature=115.0, heat_flow=1.0
    )


def test_the_first_design_matches_a_level_where_it_holds_in_every_period():
    network = periods.two_period_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        evaporators=(_evaporator(),),
        ambient_sources=(),
        base=SimpleNamespace(network=network),
        in_periods={"HP_C1": (_evaporator(), _evaporator())},
    )
    evaluator = SimpleNamespace(
        config=Configuration(),
        settings=RetrofitSettings(approach=11.0),
        names={},
    )

    state = _heat_pump_root(network, step, evaluator)

    served = sorted(m.match.hot for m in state.moves if m.match.cold == "HP_C1")
    assert served == ["H2"]
