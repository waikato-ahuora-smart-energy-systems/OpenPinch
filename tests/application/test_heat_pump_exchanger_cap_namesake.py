"""The exchanger cap counts a heat pump stream's own exchangers.

Stream keys carry the zone a stream comes from. A stream of the same name
in another zone is another stream, and its exchangers are not the heat
pump stream's: counted together they could put a network over the cap
that is within it.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application.heat_pump_design import _most_exchangers


def _exchanger(source: str, sink: str, duty: float = 10.0):
    return SimpleNamespace(
        source_stream=source,
        sink_stream=sink,
        period_states=[SimpleNamespace(duty=duty)],
    )


def test_a_namesake_in_another_zone_is_not_counted():
    network = SimpleNamespace(
        exchangers=[
            _exchanger("P1.X", "P1.C1"),
            _exchanger("P2.X", "P2.C1"),
            _exchanger("P2.X", "P2.C2"),
            _exchanger("P1.X", "P1.C2", duty=0.0),
        ]
    )

    assert _most_exchangers(network, {"P1.X"}) == 1
