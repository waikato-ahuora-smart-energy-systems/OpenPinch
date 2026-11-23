"""A heat pump mark finds its own stream, whatever else shares the name.

Two branches of a zone tree can each hold a stream of the same name. The
mark on one of them must not land on the other: pinch design would then
take an ordinary process stream out of the case to find the process pinch
and leave the heat pump's stream in.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    heat_pump_stream_keys,
)


def _stream(zone, name, supply, target, duty, **more) -> dict:
    return {
        "zone": zone,
        "name": name,
        "t_supply": supply,
        "t_target": target,
        "heat_flow": duty,
        **more,
    }


def _two_branches() -> PinchProblem:
    """P1 and P2 each have an ``Area`` holding a hot stream ``X``; P1's is a
    condenser. The ordinary one comes first in the input."""

    return PinchProblem(
        source={
            "streams": [
                _stream("P2/Area", "X", 220.0, 160.0, 100.0),
                _stream(
                    "P1/Area",
                    "X",
                    200.0,
                    199.99,
                    50.0,
                    heat_pump={"machine": "HP", "role": "condenser"},
                ),
                _stream("P1/Area", "H", 180.0, 60.0, 300.0),
                _stream("P2/Area", "C", 40.0, 120.0, 200.0),
            ],
            "utilities": [
                {
                    "name": "Steam",
                    "type": "Hot",
                    "t_supply": 250.0,
                    "t_target": 249.0,
                    "heat_flow": 0.0,
                },
                {
                    "name": "CW",
                    "type": "Cold",
                    "t_supply": 10.0,
                    "t_target": 15.0,
                    "heat_flow": 0.0,
                },
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {
                        "name": name,
                        "type": "Process Zone",
                        "children": [{"name": "Area", "type": "Unit Operation"}],
                    }
                    for name in ("P1", "P2")
                ],
            },
        },
        project_name="branches",
    )


def test_the_mark_is_read_from_the_marked_stream_not_its_namesake():
    assert heat_pump_stream_keys(_two_branches()) == {"P1.X"}
