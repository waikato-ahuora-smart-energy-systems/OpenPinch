"""A segmented heat pump stream over several periods is refused.

A vapour compression condenser arrives in segments, and its segments move
from period to period. Combining them into one stream with a value per
period is not built; written from one period's ends alone the stream
would lose its condensing plateau.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application.case_derivation import _period_heat_pump_streams
from OpenPinch.domain.stream import Stream
from tests.application.test_heat_pump_stream_record_segments import (
    _segmented_condenser,
)


def test_a_segmented_stream_in_a_period_is_refused():
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

    with pytest.raises(ValueError, match="'Condenser'.*segments.*several periods"):
        _period_heat_pump_streams(target, ["low", "high"])
