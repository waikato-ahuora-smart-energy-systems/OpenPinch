"""The heat pump step is left out of a retrofit over several periods.

The as-piped heat pump target reads the existing network's first period
alone, and the machine's streams would carry that period's duties in
every period. Until the step is built over periods it is left out with a
warning, and the result is the search without the heat pump.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_a_problem_with_several_periods_gets_no_heat_pump_step(monkeypatch):
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

    with pytest.warns(UserWarning, match="heat pump step.*left out.*2 periods"):
        result = heat_pump_retrofit.retrofit_with_heat_pump(
            problem, object(), RetrofitSettings(heat_pump_cycle="carnot")
        )

    assert result is base
    assert built == []
