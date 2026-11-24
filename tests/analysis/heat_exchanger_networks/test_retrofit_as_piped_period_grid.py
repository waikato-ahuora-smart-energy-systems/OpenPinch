"""Every period's as-piped profiles lie on one temperature grid.

A shared heat pump design compares the periods interval by interval, so
each period's grid holds every period's temperatures, not its own alone.
"""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_period_profiles,
)
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as case,
)


def test_every_periods_profiles_lie_on_one_temperature_grid():
    profiles = as_piped_period_profiles(
        case.two_period_problem(), case.two_period_network()
    )

    # The peak's colder H1 cooler and heater inlets are on the base's grid.
    base, peak = (profiles[period][0] for period in case.PERIODS)
    np.testing.assert_allclose(base, peak)
