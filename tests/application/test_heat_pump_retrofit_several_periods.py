"""A retrofit over several periods tries its heat pump step.

The step was left out with a warning while the as-piped target read the
existing network's first period alone. The target is now one machine
designed for every period's coolers and heaters, and the step, its case
and its matches are built per period, so the step is tried.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_a_problem_with_several_periods_tries_its_heat_pump_step(monkeypatch):
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    monkeypatch.setattr(
        heat_pump_retrofit, "run_retrofit_search", lambda *args, **kwargs: base
    )
    built = []
    monkeypatch.setattr(
        heat_pump_retrofit,
        "heat_pump_step",
        lambda *args, **kwargs: built.append(args) or (None, None),
    )
    problem = SimpleNamespace(period_ids={"base": 0, "peak": 1})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = heat_pump_retrofit.retrofit_with_heat_pump(
            problem, object(), RetrofitSettings(heat_pump_cycle="carnot")
        )

    assert result is base
    assert len(built) == 1
