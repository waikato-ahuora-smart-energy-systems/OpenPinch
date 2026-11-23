"""A heat pump stream inside a utility level's band is served by that level.

A steam level is written as a band 0.1 K wide (MPS: 180 to 179.9 degC).
A converged utility heat pump target puts its condenser and evaporator
inside those bands: here the condenser's hot end is 1e-5 K below the top
of the MPS band and the evaporator's cold end 0.024 K above the bottom
of the LPS band, each after the 1 K approach. A level serves a stream
whole or not at all, so written as they are the evaporator would be
heated from MPS and the condenser would raise LPS: the site would then
import 1464 kW and dump 1800 kW. The steam is at one temperature in the
plant, so the case moves each stream to the end of its band.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application.case_derivation import _case_with_utility_heat_pump_zones
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_pumps.two_level_site import (
    SITE_COLD_UTILITY,
    SITE_HOT_UTILITY,
    two_level_site,
)

CONDENSER_DUTY = 800.0
EVAPORATOR_DUTY = 664.3


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=supply,
        target_temperature=target,
        heat_flow=duty,
        heat_transfer_coefficient=1.0,
    )


def _converged_target():
    """The streams a full search gives on the two-level site."""
    return SimpleNamespace(
        provenance=SimpleNamespace(zone_address="two level site"),
        hpr_hot_streams={
            "HP_H1": _stream("HP_H1", 180.99999, 180.98999, CONDENSER_DUTY)
        },
        hpr_cold_streams={
            "HP_C1": _stream("HP_C1", 138.92356, 138.93356, EVAPORATOR_DUTY)
        },
    )


def test_streams_inside_a_levels_band_exchange_with_that_level():
    case = _case_with_utility_heat_pump_zones(
        two_level_site(), _converged_target(), dt_cont=1.0
    )

    site = case.target.total_site_heat_integration()

    assert site.hot_utility_target == pytest.approx(
        SITE_HOT_UTILITY - CONDENSER_DUTY, abs=1e-6
    )
    assert site.cold_utility_target == pytest.approx(
        SITE_COLD_UTILITY - EVAPORATOR_DUTY, abs=1e-6
    )
