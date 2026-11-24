"""A retrofit tries the heat pump at each of several sizes.

``retrofit_with_heat_pump(..., sizes=)`` builds one heat pump step per
share of the heaters' load and runs one search from each. A size whose
step cannot be used is left out with a warning; the others are still
joined with the plain search.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_each_size_is_searched_and_one_that_fails_is_left_out(monkeypatch):
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    searched = []

    def search(problem, existing, settings, **arguments):
        if "heat_pump" not in arguments:
            return base
        size = arguments["heat_pump"]
        if size == 0.5:
            raise ValueError("no heater starts cold enough")
        searched.append(size)
        return f"search at {size}"

    case = SimpleNamespace(_build_execution_master_zone=lambda: object())
    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)
    monkeypatch.setattr(
        heat_pump_retrofit,
        "heat_pump_step",
        lambda *args, load_fraction=None, **kwargs: (load_fraction, object()),
    )
    monkeypatch.setattr(heat_pump_retrofit, "heat_pump_case", lambda *a, **k: case)
    monkeypatch.setattr(heat_pump_retrofit, "snapshot_problem", lambda p: p)
    joined = []
    monkeypatch.setattr(
        heat_pump_retrofit,
        "joined",
        lambda first, *rest: joined.append((first, rest)) or "joined",
    )

    with pytest.warns(UserWarning, match="size 0.5.*left out.*no heater"):
        result = heat_pump_retrofit.retrofit_with_heat_pump(
            SimpleNamespace(period_ids={"0": 0}),
            object(),
            RetrofitSettings(heat_pump_cycle="carnot"),
            sizes=(1.0, 0.75, 0.5),
        )

    assert result == "joined"
    assert searched == [1.0, 0.75]
    assert joined == [(base, ("search at 1.0", "search at 0.75"))]
