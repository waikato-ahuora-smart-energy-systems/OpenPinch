"""The heat pump's zones keep the target's approaches under a multiplier.

A utility heat pump target stands its streams off a utility level by
``HPR_DT_CONT``, as given, plus the level's own contribution times the
site's dt_cont multiplier. The new zones inherit that multiplier, which
scales every stream's contribution in them, so the heat pump's streams
are written with ``HPR_DT_CONT`` over the multiplier and the band move
reads each level's contribution times it. Written at ``HPR_DT_CONT``
itself the streams would stand twice as far off as the target put them,
and the levels could not serve them.

Here the multiplier is 2 and each utility's contribution 0.25 K, so the
target's approach is 1 + 0.5 = 1.5 K: the condenser's hot end sits 1e-5 K
below 181.5 degC, the top of the MPS band plus the approach, and the
evaporator's cold end 0.024 K above 138.4 degC. Process B ends at
165 degC so that MPS still reaches it at the doubled contributions; the
site imports 800 kW and rejects 1000 kW as before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _case_with_utility_heat_pump_zones
from tests.analysis.heat_pumps.two_level_site import (
    SITE_COLD_UTILITY,
    SITE_HOT_UTILITY,
    two_level_site_source,
)
from tests.application.test_utility_heat_pump_level_bands import (
    CONDENSER_DUTY,
    EVAPORATOR_DUTY,
    _stream,
)


def _site_at_twice_the_contributions() -> PinchProblem:
    source = two_level_site_source()
    for utility in source["utilities"]:
        utility["dt_cont"] = {"value": 0.25, "unit": "degC"}
    b_cold = next(s for s in source["streams"] if s["name"] == "B cold")
    b_cold["t_target"] = {"value": 165.0, "unit": "degC"}
    problem = PinchProblem(source=source, project_name="two level site")
    problem.set_dt_cont_multiplier(2.0)
    return problem


def test_the_zones_target_as_the_heat_pump_did_under_a_dt_cont_multiplier():
    problem = _site_at_twice_the_contributions()
    before = problem.target.total_site_heat_integration()
    assert before.hot_utility_target == pytest.approx(SITE_HOT_UTILITY)
    assert before.cold_utility_target == pytest.approx(SITE_COLD_UTILITY)
    target = SimpleNamespace(
        provenance=SimpleNamespace(zone_address="two level site"),
        hpr_hot_streams={
            "HP_H1": _stream("HP_H1", 181.49999, 181.48999, CONDENSER_DUTY)
        },
        hpr_cold_streams={
            "HP_C1": _stream("HP_C1", 138.42356, 138.43356, EVAPORATOR_DUTY)
        },
    )

    case = _case_with_utility_heat_pump_zones(problem, target, dt_cont=1.0)
    site = case.target.total_site_heat_integration()

    assert site.hot_utility_target == pytest.approx(
        SITE_HOT_UTILITY - CONDENSER_DUTY, abs=1e-6
    )
    assert site.cold_utility_target == pytest.approx(
        SITE_COLD_UTILITY - EVAPORATOR_DUTY, abs=1e-6
    )
