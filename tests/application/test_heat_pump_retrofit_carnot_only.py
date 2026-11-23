"""``retrofit_with_heat_pump`` takes the Carnot and vapour compression cycles.

The function clears the cycle for its plain search, which would skip the
search's own check of it. Any other cycle is refused first, so the helper
never builds a machine labelled with a cycle it did not model. Vapour
compression was refused too until the as-piped target could solve it and
the machine be priced by its compressors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


class _Searched(Exception):
    pass


def _search(*args, **kwargs):
    raise _Searched


@pytest.mark.parametrize("cycle", ["brayton", None])
def test_a_cycle_other_than_carnot_or_vapour_compression_is_refused(monkeypatch, cycle):
    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", _search)

    with pytest.raises(ValueError, match="carnot.*" + str(cycle)):
        heat_pump_retrofit.retrofit_with_heat_pump(
            SimpleNamespace(period_ids={"0": 0}),
            object(),
            RetrofitSettings(heat_pump_cycle=cycle),
        )


def test_a_vapour_compression_retrofit_runs_its_search(monkeypatch):
    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", _search)

    with pytest.raises(_Searched):
        heat_pump_retrofit.retrofit_with_heat_pump(
            SimpleNamespace(period_ids={"0": 0}),
            object(),
            RetrofitSettings(heat_pump_cycle="vapour_compression"),
        )
