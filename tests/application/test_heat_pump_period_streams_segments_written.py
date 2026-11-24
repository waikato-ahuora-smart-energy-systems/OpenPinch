"""A segmented heat pump stream over several periods keeps its segments.

A vapour compression condenser arrives in segments (desuperheating,
condensing, subcooling), and their duties move from period to period.
Each segment is written with its temperatures and duty in every period,
so the stream keeps its condensing plateau in each.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _period_thermal_record
from OpenPinch.contracts.input import StreamSchema
from tests.application.heat_pump_stream_cases import basic_source
from tests.application.test_heat_pump_stream_record_segments import PIECES


def condenser(scale: float, pieces=PIECES):
    source = basic_source()
    source["streams"].append(
        {
            "zone": "Plant",
            "name": "Condenser",
            "segments": [
                {"t_supply": supply, "t_target": target, "heat_flow": duty * scale}
                for supply, target, duty in pieces
            ],
        }
    )
    root = PinchProblem(source=source, project_name="seg")._require_prepared_root_zone()
    return next(s for _key, s in root.hot_streams.items() if s.name == "Condenser")


def test_each_segment_is_written_in_every_period():
    record = _period_thermal_record([condenser(1.0), condenser(2.0)])

    assert [
        segment["heat_flow"]["values"] for segment in record.get("segments", ())
    ] == [pytest.approx([duty, 2.0 * duty]) for _s, _t, duty in PIECES]
    assert "t_supply" not in record and "heat_flow" not in record


def test_each_segment_keeps_its_film_coefficient():
    # A segmented stream's film coefficients are its segments' own: the
    # per-period segments must carry them, as a single period's do.
    record = _period_thermal_record([condenser(1.0), condenser(2.0)])

    assert all("htc" in segment for segment in record["segments"])
