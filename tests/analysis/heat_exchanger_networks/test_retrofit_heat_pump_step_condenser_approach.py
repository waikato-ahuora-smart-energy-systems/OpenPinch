"""With no approach given, a condenser stands off a heater by the heated
stream's own contribution.

C2's heater starts at 175.7 degC: 2.3 K under a condenser whose cold end
is 178 degC. That passed ``THERMAL_DT_CONT`` plus ``HPR_DT_CONT`` (2 K),
but not C2's own 5 K plus the heat pump's 1 K, so nothing can serve the
condenser and the step refuses it, quoting the approach it was held to.
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


def test_a_condenser_stands_off_a_heater_by_its_stream_contribution():
    config = Configuration()
    config.update_values(THERMAL_DT_CONT=1.0, HPR_DT_CONT=1.0)
    network = _solved_network()
    condenser = Stream(
        name="HP_H1", supply_temperature=178.01, target_temperature=178.0, heat_flow=1.0
    )
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(condenser,),
        evaporators=(),
        ambient_sources=(),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=config,
        settings=RetrofitSettings(approach=None),
        names={},
        hot_contributions={"H1": 1.0, "H2": 1.0},
        cold_contributions={"C2": 5.0},
    )

    with pytest.raises(ValueError, match="'HP_H1'.*contribution plus 1 K"):
        _heat_pump_root(network, step, evaluator)
