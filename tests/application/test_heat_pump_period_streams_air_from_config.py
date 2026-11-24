"""The air of a shared design is written even where its own period has none.

A shared design's own collections hold one period, and its air stream is
that period's: a period that draws no air leaves it out, while another
period may draw on air. The air stream was refused over several periods;
it is now written in each period at that period's duty. Where the target
holds no air stream to copy, it is built from the target's configuration:
latent at the ambient temperature (``ENV_T``, 15 degC by default) over
``THERMAL_DT_PHASE_CHANGE``, at ``HPR_DT_ENV_CONT``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application.case_derivation import _period_heat_pump_streams
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=duty
    )


def _output(drawn: float) -> dict:
    return {
        "hpr_hot_streams": {"HP_H1": _stream("HP_H1", 230.0, 229.99, 300.0)},
        "hpr_cold_streams": {"HP_C1": _stream("HP_C1", 5.0, 5.01, 200.0)},
        "Q_amb_hot": drawn,
    }


def test_air_is_built_from_the_configuration_when_the_target_holds_none():
    target = SimpleNamespace(
        config=Configuration(),
        hpr_details=SimpleNamespace(
            amb_streams=StreamCollection(),
            period_outputs={"low": _output(0.0), "high": _output(50.0)},
        ),
    )

    found = _period_heat_pump_streams(target, ["low", "high"])

    low, high = found["ambient_source"]["AIR_H1"]
    assert low is None
    assert float(high.heat_flow.value) == pytest.approx(50.0)
    assert float(high.supply_temperature.value) == pytest.approx(15.0)
    assert float(high.delta_t_contribution.value) == pytest.approx(10.0)
