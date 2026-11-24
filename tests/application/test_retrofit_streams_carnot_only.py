"""A heat pump retrofit runs the stream step, for either cycle.

The Carnot and vapour compression cycles add the as-piped heat pump as
streams. Each kept a linked move (``heat_pump_streams=False``) until the
user retired it on 2026-11-24: on the guide's case the stream step netted
$2.81M/y where the linked Carnot move netted $1.23M/y, and on a
low-temperature retrofit the linked vapour compression move found no heat
pump where the stream step netted 67,000 $/y.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


@pytest.mark.parametrize("cycle", ["carnot", "vapour_compression"])
def test_a_heat_pump_retrofit_runs_the_stream_step(monkeypatch, cycle):
    from OpenPinch.application import heat_pump_retrofit

    calls = []

    def streams(problem, existing, settings, **arguments):
        calls.append(settings.heat_pump_cycle)
        return object()

    monkeypatch.setattr(heat_pump_retrofit, "retrofit_with_heat_pump", streams)

    levels_problem().design.retrofit(
        EXISTING, heat_pump_cycle=cycle, heat_pump_cost=100.0
    )

    assert calls == [cycle]
