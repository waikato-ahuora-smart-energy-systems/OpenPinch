"""A case the heat pump's streams cannot be added to leaves the step out.

The machine's streams join a copy of the problem. Some problems refuse
that: one with process components (its input records do not hold them)
and a site of several process zones (no one zone to add them to). The
retrofit itself is still sound, so its result is the search without the
heat pump, with a warning that gives the reason.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_a_case_that_refuses_the_streams_gives_the_plain_result(monkeypatch):
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    monkeypatch.setattr(
        heat_pump_retrofit, "run_retrofit_search", lambda *args, **kwargs: base
    )
    monkeypatch.setattr(
        heat_pump_retrofit,
        "heat_pump_step",
        lambda *args, **kwargs: (object(), object()),
    )

    def refused(*args, **kwargs):
        raise ValueError("heat pump streams cannot be added to a problem with process")

    monkeypatch.setattr(heat_pump_retrofit, "heat_pump_case", refused)

    with pytest.warns(UserWarning, match="heat pump step.*left out.*process"):
        result = heat_pump_retrofit.retrofit_with_heat_pump(
            SimpleNamespace(period_ids={"0": 0}),
            object(),
            RetrofitSettings(heat_pump_cycle="carnot"),
        )

    assert result is base
