"""A heat pump stream no cooler or heater can serve is refused by name.

Every stream of the case needs an exchanger. An evaporator hotter than
every cooler's start (less the approach) gets no match, and the solve
would be refused for a stream "with no exchanger", which says nothing of
the heat pump. The step is refused where it is built, naming the stream
and the approach.

Ciric and Floudas's coolers start at 136.5 and 169.6 degC; at a 10 K
approach nothing can give heat to an evaporator at 165 degC.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    HeatPumpStep,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    _heat_pump_root,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def test_an_evaporator_too_hot_for_every_cooler_is_refused():
    network = _solved_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        evaporators=(
            Stream(
                name="HP_C1",
                supply_temperature=164.99,
                target_temperature=165.0,
                heat_flow=1.0,
            ),
        ),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=None, settings=RetrofitSettings(approach=10.0), names={}
    )

    with pytest.raises(ValueError, match="no cooler.*'HP_C1'.*approach of 10 K"):
        _heat_pump_root(network, step, evaluator)
