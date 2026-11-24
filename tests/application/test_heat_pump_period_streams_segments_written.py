"""A segmented heat pump stream over several periods keeps its segments.

A vapour compression condenser arrives in segments (desuperheating,
condensing, subcooling), and their duties move from period to period.
Each segment is written with its temperatures and duty in every period,
so the stream keeps its condensing plateau in each.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _period_thermal_record
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


def test_periods_with_different_breakpoints_keep_each_profile_whole():
    # Segment names are positional (S1, S2, ...): a period that condenses
    # at 170 degC with no subcooling must not borrow the other period's
    # subcooling temperatures. Each period's profile is cut at the union
    # of both periods' duty fractions, so its segments chain end to end.
    other = [(215.0, 170.0, 8.0), (170.0, 169.99, 200.0)]
    record = _period_thermal_record([condenser(1.0), condenser(1.0, other)])

    for period, total in ((0, 188.0), (1, 208.0)):
        chain = [
            (
                segment["t_supply"]["values"][period],
                segment["t_target"]["values"][period],
                segment["heat_flow"]["values"][period],
            )
            for segment in record["segments"]
        ]
        assert sum(duty for _s, _t, duty in chain) == pytest.approx(total)
        for (_s, end, _q), (start, _t, _d) in zip(chain, chain[1:]):
            assert start == pytest.approx(end)
