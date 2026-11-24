"""The retrofit search's linked heat pump move is retired.

A heat pump enters a retrofit as streams (``retrofit_with_heat_pump``,
which ``design.retrofit`` runs for a heat pump cycle); the user chose on
2026-11-24 to retire the linked move, an evaporator and condenser paired as
utilities, that ``RetrofitSettings.heat_pump_cycle`` enabled. A search
given a cycle is refused, before any solve.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    run_retrofit_search,
)
from tests.analysis.heat_exchanger_networks.test_retrofit_search import (
    _existing_network,
    _problem,
)


@pytest.mark.parametrize("cycle", ["carnot", "vapour_compression"])
def test_a_search_given_a_heat_pump_cycle_is_refused(cycle):
    with pytest.raises(ValueError, match="linked heat pump move.*retired"):
        run_retrofit_search(
            _problem(), _existing_network(), RetrofitSettings(heat_pump_cycle=cycle)
        )
