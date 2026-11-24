"""A heat pump cannot be written into a zone whose multiplier is 0.

A heat pump's streams are written with their approach over the zone's
dt_cont multiplier, so the zone scales them back to it. A zone tree may
give a multiplier of 0, which keeps no contribution at all: the heat pump
could not stand off the process or the levels as its target assumed, so
the case is refused with a ``ValueError`` naming the zone. Today the
utility zones divide by the multiplier and raise ``ZeroDivisionError``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application.case_derivation import _case_with_utility_heat_pump_zones
from tests.analysis.heat_pumps.two_level_site import recovering_two_level_site


def test_a_utility_heat_pump_on_a_site_with_no_multiplier_is_refused():
    problem = recovering_two_level_site()
    problem.set_dt_cont_multiplier(0.0)
    root = problem._require_prepared_root_zone()
    target = SimpleNamespace(provenance=SimpleNamespace(zone_address=root.address))

    with pytest.raises(ValueError, match="multiplier of 0"):
        _case_with_utility_heat_pump_zones(problem, target, dt_cont=1.0)


def test_a_retrofit_heat_pump_in_a_zone_with_no_multiplier_is_refused():
    # The retrofit's case divides by the zone multiplier as well.
    from OpenPinch.application.heat_pump_retrofit import heat_pump_case
    from OpenPinch.domain.stream import Stream
    from OpenPinch.domain.stream_collection import StreamCollection
    from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
        levels_problem,
    )

    def stream(name: str, supply: float, target: float) -> Stream:
        return Stream(
            name=name,
            supply_temperature=supply,
            target_temperature=target,
            heat_flow=10.0,
            heat_transfer_coefficient=1.0,
        )

    problem = levels_problem()
    problem.set_dt_cont_multiplier(0.0)
    solved = SimpleNamespace(
        hpr_hot_streams={"a": stream("HP_H1", 110.0, 109.99)},
        hpr_cold_streams={"b": stream("HP_C1", 33.99, 34.0)},
        amb_streams=StreamCollection(),
    )

    with pytest.raises(ValueError, match="multiplier of 0"):
        heat_pump_case(problem, solved, machine="HP")
