"""A process stream can be marked as one side of a heat pump.

A heat pump of a fixed size enters a design as streams: each condenser a
hot stream, each evaporator a cold stream. The mark says which machine a
stream belongs to and its role there, so the design can tell them from
the process's own streams.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.input import StreamSchema


def _stream(heat_pump) -> dict:
    return {
        "zone": "Plant",
        "name": "HP_H1",
        "t_supply": 226.0,
        "t_target": 225.99,
        "heat_flow": 750.0,
        "heat_pump": heat_pump,
    }


def test_a_stream_names_its_heat_pump_machine_and_one_of_three_roles():
    with pytest.raises(ValidationError, match="condenser"):
        StreamSchema.model_validate(_stream({"machine": "HP", "role": "compressor"}))

    marked = StreamSchema.model_validate(
        _stream({"machine": "HP", "role": "condenser"})
    )

    assert (marked.heat_pump.machine, marked.heat_pump.role) == ("HP", "condenser")
    assert marked.model_dump(mode="python")["heat_pump"] == {
        "machine": "HP",
        "role": "condenser",
        "placement": "process",
    }
    for role in ("evaporator", "ambient_source"):
        StreamSchema.model_validate(_stream({"machine": "HP", "role": role}))
    assert StreamSchema.model_validate(_stream(None)).heat_pump is None
