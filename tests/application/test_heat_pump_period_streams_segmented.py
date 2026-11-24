"""A segmented heat pump stream over several periods is taken, not refused.

A vapour compression condenser arrives in segments, and its segments move
from period to period. It was refused while its record held one period's
ends alone, which would lose the condensing plateau; each segment is now
written per period (``_period_thermal_record``), so the stream is taken
in every period it runs.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application.case_derivation import _period_heat_pump_streams
from OpenPinch.domain.stream import Stream
from tests.application.test_heat_pump_stream_record_segments import (
    _segmented_condenser,
)


def test_a_segmented_stream_is_taken_in_every_period():
    output = {
        "hpr_hot_streams": {"Condenser": _segmented_condenser()},
        "hpr_cold_streams": {
            "Evaporator": Stream(
                name="Evaporator",
                supply_temperature=100.0,
                target_temperature=100.01,
                heat_flow=120.0,
            )
        },
        "Q_amb_hot": 0.0,
    }
    target = SimpleNamespace(
        hpr_details=SimpleNamespace(period_outputs={"low": output, "high": output})
    )

    found = _period_heat_pump_streams(target, ["low", "high"])

    assert [s.name for s in found["condenser"]["Condenser"]] == [
        "Condenser",
        "Condenser",
    ]
