"""An as-piped target that finds no machine leaves the step out.

A vapour compression cycle cannot serve every network: the HPR service
raises when no machine it can model improves on the existing heaters and
coolers (a heater above what the refrigerants reach, say). The retrofit
itself is still sound, so its result is the search without the heat
pump, with a warning that gives the reason.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_a_target_with_no_machine_gives_the_plain_result(monkeypatch):
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    monkeypatch.setattr(
        heat_pump_retrofit, "run_retrofit_search", lambda *args, **kwargs: base
    )

    def no_machine(*args, **kwargs):
        raise ValueError("found no heat pump design that improves on the baseline")

    monkeypatch.setattr(heat_pump_retrofit, "as_piped_heat_pump", no_machine)

    with pytest.warns(UserWarning, match="heat pump step.*left out.*no heat pump"):
        result = heat_pump_retrofit.retrofit_with_heat_pump(
            SimpleNamespace(period_ids={"0": 0}),
            object(),
            RetrofitSettings(heat_pump_cycle="vapour_compression"),
        )

    assert result is base
