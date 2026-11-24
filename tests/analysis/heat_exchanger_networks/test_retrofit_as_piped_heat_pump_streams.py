"""The as-piped heat pump target is available as the machine's streams.

``retrofit_heat_pumps`` reports the as-piped target as levels and duties.
A retrofit step that adds the heat pump to the network as streams needs
the streams themselves: each condenser a hot stream and each evaporator a
cold one, as the heat pump service solved them on the network's coolers
and heaters at the network's prices.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
    retrofit_heat_pumps,
)
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _problem,
    _solved_network,
)

# One evaluation takes the search's starting point as it is: enough to
# compare the two readings of one solve.
BUDGET = {"maximum_evaluations": 1}


def _duties(streams) -> list[float]:
    return sorted(float(stream.heat_flow.value) for _key, stream in streams.items())


def test_the_as_piped_heat_pump_holds_the_targets_condensers_and_evaporators():
    summary = retrofit_heat_pumps(_problem(), _solved_network(), args=BUDGET).as_piped
    assert summary.heat_delivered > 0.0

    machine = as_piped_heat_pump(_problem(), _solved_network(), args=BUDGET)

    # A level the solve left without duty has no stream.
    assert _duties(machine.hpr_hot_streams) == pytest.approx(
        sorted(duty for duty in summary.condenser_duties if duty > 0.0)
    )
    assert _duties(machine.hpr_cold_streams) == pytest.approx(
        sorted(duty for duty in summary.evaporator_duties if duty > 0.0)
    )
    assert float(machine.w_net) == pytest.approx(summary.work)
