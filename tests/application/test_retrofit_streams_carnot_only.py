"""The stream step takes the Carnot and vapour compression cycles.

A vapour compression retrofit heat pump keeps the linked move unless the
stream step is asked for (``heat_pump_streams=True``), until the two are
compared; Carnot runs the stream step by default. Vapour compression with
the stream step was refused until the stream step could model it.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_vapour_compression_runs_the_stream_step_when_asked(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
    from OpenPinch.application import heat_pump_retrofit

    calls = []

    def streams(problem, existing, settings, **arguments):
        calls.append(("streams", settings.heat_pump_cycle))
        return object()

    def linked(problem, existing, settings, **arguments):
        calls.append(("linked", settings.heat_pump_cycle))
        return object()

    monkeypatch.setattr(heat_pump_retrofit, "retrofit_with_heat_pump", streams)
    monkeypatch.setattr(search, "run_retrofit_search", linked)

    levels_problem().design.retrofit(
        EXISTING, heat_pump_cycle="vapour_compression", heat_pump_streams=True
    )
    levels_problem().design.retrofit(EXISTING, heat_pump_cycle="vapour_compression")

    assert calls == [
        ("streams", "vapour_compression"),
        ("linked", "vapour_compression"),
    ]


def test_streams_with_no_cycle_are_refused():
    with pytest.raises(ValueError, match="heat_pump_streams.*None"):
        levels_problem().design.retrofit(
            EXISTING, heat_pump_cycle=None, heat_pump_streams=True
        )
