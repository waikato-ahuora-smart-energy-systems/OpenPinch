"""Each period's as-piped profiles are read from that period's states:
the coolers and heaters as they run in it, not as in the first period."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_period_profiles,
)
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as case,
)


def test_each_periods_profiles_are_its_own_coolers_and_heaters():
    profiles = as_piped_period_profiles(
        case.two_period_problem(), case.two_period_network()
    )

    # In the peak H1's cooler takes 9000 kW and the heater 20000 kW.
    _T, H_hot, H_cold = profiles["peak"]
    assert abs(H_hot).max() == pytest.approx(9000.0 + 1665.1)
    assert abs(H_cold).max() == pytest.approx(20000.0)
