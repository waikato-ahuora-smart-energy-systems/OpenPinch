"""A cooler or heater off in a period gives no segment there.

A solved network can hold an exchanger that is off in a period as no
duty, inactive, with its process-side temperatures unset. That period's
as-piped profiles leave it out; the other periods keep it.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_period_profiles,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as periods,
)


def _with_h1_cooler_off_in_the_peak(network):
    def off(exchanger):
        if exchanger.exchanger_id != "cold-utility:H1->CU":
            return exchanger
        idle = HeatExchangerPeriodState(
            period_id="peak", period_idx=1, duty=0.0, active=False
        )
        return exchanger.model_copy(
            update={"period_states": (exchanger.period_states[0], idle)}
        )

    return network.model_copy(
        update={"exchangers": tuple(off(e) for e in network.exchangers)}
    )


def test_an_exchanger_off_in_a_period_is_no_segment_there():
    network = _with_h1_cooler_off_in_the_peak(periods.two_period_network())

    profiles = as_piped_period_profiles(periods.two_period_problem(), network)

    assert abs(profiles["peak"][1]).max() == pytest.approx(1665.1)
    assert abs(profiles["base"][1]).max() == pytest.approx(13603.6 + 1665.1)
