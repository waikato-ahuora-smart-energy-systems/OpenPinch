"""An ambient source nothing can serve is refused at the air's approach.

With no ``min_approach_temperature`` the air is matched at its own
contribution plus the heat pump's (11 K by default), not at the process
streams' approach. The refusal quotes the approach the air was held to.
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
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def _stream(name: str, supply: float, target: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=1.0
    )


def test_the_refusal_quotes_the_airs_approach():
    config = Configuration()
    config.update_values(THERMAL_DT_CONT=5.0, HPR_DT_CONT=1.0, HPR_DT_ENV_CONT=10.0)
    network = _solved_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        # Served by H2's cooler, but only 6 K below the air.
        evaporators=(_stream("HP_C1", 8.99, 9.0),),
        ambient_sources=(_stream("AIR_H1", 15.0, 14.99),),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=config, settings=RetrofitSettings(approach=None), names={}
    )

    with pytest.raises(ValueError, match="'AIR_H1'.*approach of 11 K"):
        _heat_pump_root(network, step, evaluator)
