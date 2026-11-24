"""No heat pump step when no modification is allowed.

The heat pump step is a modification. With ``max_modifications=0`` the
retrofit only rates the existing network, so the step is not built and
the result is the plain search's.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_with_no_modifications_allowed_no_step_is_built(monkeypatch):
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

    result = heat_pump_retrofit.retrofit_with_heat_pump(
        SimpleNamespace(period_ids={"0": 0}),
        object(),
        RetrofitSettings(heat_pump_cycle="carnot", max_modifications=0),
    )

    assert result is base
    assert built == []
