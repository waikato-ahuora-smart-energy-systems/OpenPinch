"""A segmented heat pump stream is added with its segments.

A vapour compression target gives a condenser as one stream in pieces:
desuperheating, condensing and subcooling. Flattened to one supply and one
target temperature it would offer its heat along a straight line between
them, hotter than the condensing level really is.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _heat_pump_stream_record
from OpenPinch.contracts.input import StreamSchema
from tests.application.heat_pump_stream_cases import basic_source

# (supply degC, target degC, kW): desuperheat, condense, subcool.
PIECES = [(215.0, 165.0, 10.0), (165.0, 164.99, 174.0), (164.99, 155.0, 4.0)]


def _segmented_condenser():
    source = basic_source()
    source["streams"].append(
        {
            "zone": "Plant",
            "name": "Condenser",
            "segments": [
                {"t_supply": supply, "t_target": target, "heat_flow": duty}
                for supply, target, duty in PIECES
            ],
        }
    )
    root = PinchProblem(source=source, project_name="seg")._require_prepared_root_zone()
    return next(s for _key, s in root.hot_streams.items() if s.name == "Condenser")


def test_a_segmented_heat_pump_stream_is_recorded_segment_by_segment():
    record = _heat_pump_stream_record(
        _segmented_condenser(),
        zone="seg/Plant",
        dt_cont={"value": 1.0, "unit": "delta_degC"},
        mark={"machine": "HP", "role": "condenser"},
    )

    assert [
        (
            segment["t_supply"]["value"],
            segment["t_target"]["value"],
            segment["heat_flow"]["value"],
        )
        for segment in record["segments"]
    ] == [pytest.approx(piece) for piece in PIECES]
    assert "t_supply" not in record and "heat_flow" not in record
    assert StreamSchema.model_validate(record).heat_pump.role == "condenser"
