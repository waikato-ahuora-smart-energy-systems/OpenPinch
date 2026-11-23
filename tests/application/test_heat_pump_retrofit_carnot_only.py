"""``retrofit_with_heat_pump`` takes the Carnot cycle only.

The function clears the cycle for its plain search, which would skip the
search's own check of it. A vapour compression or unknown cycle is
refused first, so the helper never builds a machine labelled with a cycle
it did not model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


@pytest.mark.parametrize("cycle", ["vapour_compression", "brayton"])
def test_a_cycle_other_than_carnot_is_refused(monkeypatch, cycle):
    def search(*args, **kwargs):
        raise AssertionError("the search ran")

    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)

    with pytest.raises(ValueError, match="carnot.*" + cycle):
        heat_pump_retrofit.retrofit_with_heat_pump(
            SimpleNamespace(period_ids={"0": 0}),
            object(),
            RetrofitSettings(heat_pump_cycle=cycle),
        )
