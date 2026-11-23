"""The stream step is built for the Carnot cycle only.

A vapour compression retrofit heat pump keeps the linked move. Asking for
the stream step with it is refused by name rather than ignored.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_streams_with_vapour_compression_are_refused():
    with pytest.raises(ValueError, match="heat_pump_streams.*carnot"):
        levels_problem().design.retrofit(
            EXISTING,
            heat_pump_cycle="vapour_compression",
            heat_pump_streams=True,
        )
