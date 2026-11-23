"""The moves of a heat pump step carry how each match was placed.

A search prices a new match from its move: a split costs ``split_cost``
and each process stream it ties into ``repiping_cost``, and the result
reports whether the match made its stage. The moves of a heat pump step
are built from the matches as they were placed, so a match that joins
its level's stage is reported as joining it, with the level's stream it
splits, and each match repipes its process stream (the heat pump's own
stream is new, so it has no piping to change).
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
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def test_a_joining_match_keeps_its_split_and_each_match_repipes_its_process_stream():
    network = _solved_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        evaporators=(
            Stream(
                name="HP_C1",
                supply_temperature=99.99,
                target_temperature=100.0,
                heat_flow=1.0,
            ),
        ),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=None, settings=RetrofitSettings(approach=10.0), names={}
    )

    state = _heat_pump_root(network, step, evaluator)

    moves = {
        (m.match.hot, m.match.cold): (m.match.new_stage, m.splits, m.repiped)
        for m in state.moves
    }
    assert moves == {
        ("H1", "HP_C1"): (True, (), ("H1",)),
        ("H2", "HP_C1"): (False, ("HP_C1",), ("H2",)),
    }
    assert {m.machine for m in state.moves} == {"HP"}
