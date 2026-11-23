"""A heat pump size outside 0 to 1 of the heaters' load is refused.

``load_fraction`` is written onto a copy of the configuration without the
field validation ``HPR_LOAD_FRACTION`` normally gets, so the bounds the
key carries (0 to 1) are checked where the size is taken. A size of 1.5
would otherwise ask the target for half as much again as the heaters
carry.
"""

from __future__ import annotations

import math

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
)
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _problem,
    _solved_network,
)


@pytest.mark.parametrize("size", [1.5, -0.1, math.nan, math.inf])
def test_a_size_outside_zero_to_one_is_refused(size):
    with pytest.raises(ValueError, match="load_fraction.*between 0 and 1"):
        as_piped_heat_pump(
            _problem(),
            _solved_network(),
            args={"maximum_evaluations": 1},
            load_fraction=size,
        )
