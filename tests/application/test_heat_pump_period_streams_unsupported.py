"""What a shared design's period streams cannot yet be written for.

Two things a single-period case handles are not built over several
periods: the air an evaporator draws on, and a segmented stream. Each is
refused by name; a case without the air, or with a stream flattened,
would look complete and target wrongly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application.case_derivation import _period_heat_pump_streams
from OpenPinch.domain.stream import Stream


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=duty
    )


def _target(**period):
    output = {
        "hpr_hot_streams": {"HP_H1": _stream("HP_H1", 230.0, 229.99, 300.0)},
        "hpr_cold_streams": {"HP_C1": _stream("HP_C1", 100.0, 100.01, 200.0)},
        "Q_amb_hot": 0.0,
    } | period
    return SimpleNamespace(
        hpr_details=SimpleNamespace(period_outputs={"low": output, "high": output})
    )


def test_air_drawn_on_in_a_period_is_refused():
    with pytest.raises(ValueError, match="ambient air.*several periods"):
        _period_heat_pump_streams(_target(Q_amb_hot=50.0), ["low", "high"])
