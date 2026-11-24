"""design.retrofit's keywords for the linked heat pump move are gone.

The linked move was retired on 2026-11-24 (the user's choice): a heat pump
cycle always adds the as-piped heat pump as streams, so
``heat_pump_streams`` (False chose the linked move) has nothing left to
choose.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_design_retrofit_takes_no_heat_pump_streams(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    monkeypatch.setattr(search, "run_retrofit_search", lambda *a, **k: object())

    with pytest.raises(TypeError, match="heat_pump_streams"):
        levels_problem().design.retrofit(
            EXISTING, heat_pump_cycle="carnot", heat_pump_streams=False
        )
