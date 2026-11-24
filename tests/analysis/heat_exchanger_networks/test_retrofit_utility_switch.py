"""A retrofit may switch an existing heater or cooler to another utility.

The unit keeps its id, its stream and its installed area; only the
utility serving it changes, so a heater on high-pressure steam can move to
cheaper low-pressure steam where the stream's temperatures allow it.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
    apply_utility_switch,
    utility_switch_candidates,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _network() -> HeatExchangerNetwork:
    network = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=[("C1", "HPS")],
        coolers=[("H1", "CW")],
    )
    return network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": 5.0}) for e in network.exchangers
            )
        }
    )


def test_a_switched_heater_keeps_its_id_stream_and_area():
    network = _network()
    (heater,) = (
        e for e in network.exchangers if e.sink_stream == "C1" and e.stage is None
    )

    switched = apply_utility_switch(network, heater.exchanger_id, "LPS")

    (after,) = (e for e in switched.exchangers if e.exchanger_id == heater.exchanger_id)
    assert (after.source_stream, after.sink_stream, after.area) == ("LPS", "C1", 5.0)


def test_each_end_unit_may_switch_to_each_other_utility_of_its_kind():
    network = _network()
    heater = next(
        e.exchanger_id for e in network.exchangers if e.source_stream == "HPS"
    )
    cooler = next(e.exchanger_id for e in network.exchangers if e.sink_stream == "CW")

    candidates = utility_switch_candidates(
        network,
        units={heater, cooler},
        hot_utilities=("HPS", "LPS"),
        cold_utilities=("CW", "AIR"),
    )

    assert candidates == ((heater, "LPS"), (cooler, "AIR"))
