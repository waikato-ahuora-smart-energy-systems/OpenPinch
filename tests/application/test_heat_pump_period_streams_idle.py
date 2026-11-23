"""A heat pump stream is written as off in a period the machine is off.

A shared design leaves a period with no load out of its period outputs,
or gives it an output with no streams. The stream still needs a value in
that period, as every stream of the problem has: no duty, at the
temperatures it has when the machine runs.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application.case_derivation import (
    _period_heat_pump_streams,
    _period_thermal_record,
)
from OpenPinch.domain.stream import Stream


def _running() -> dict:
    return {
        "hpr_hot_streams": {
            "HP_H1": Stream(
                name="HP_H1",
                supply_temperature=230.0,
                target_temperature=229.99,
                heat_flow=600.0,
            )
        },
        "hpr_cold_streams": {
            "HP_C1": Stream(
                name="HP_C1",
                supply_temperature=100.0,
                target_temperature=100.01,
                heat_flow=400.0,
            )
        },
        "Q_amb_hot": 0.0,
    }


def test_a_period_left_out_or_without_streams_gives_the_stream_no_duty():
    # Left out altogether in "off season"; present with no streams in "idle".
    idle = {"hpr_hot_streams": {}, "hpr_cold_streams": {}, "Q_amb_hot": 0.0}
    target = SimpleNamespace(
        hpr_details=SimpleNamespace(
            period_outputs={"idle": idle, "season": _running()}
        )
    )

    found = _period_heat_pump_streams(target, ["off season", "idle", "season"])
    condenser = _period_thermal_record(found["condenser"]["HP_H1"])

    assert condenser["heat_flow"]["values"] == [0.0, 0.0, 600.0]
    assert condenser["t_supply"]["values"] == [230.0, 230.0, 230.0]
    assert condenser["t_target"]["values"] == [229.99, 229.99, 229.99]
