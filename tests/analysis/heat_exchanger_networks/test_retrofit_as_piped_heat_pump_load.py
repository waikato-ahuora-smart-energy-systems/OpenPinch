"""The as-piped heat pump can be asked for part of the heaters' load.

A retrofit tries the heat pump at several sizes. ``load_fraction`` caps
the heat the machine delivers at that share of what the existing
network's heaters carry (``HPR_LOAD_FRACTION`` on the target the
retrofit solves), leaving the problem's own configuration alone.

Ciric and Floudas's network heats C2 with 17,518.7 kW of hot utility.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
)
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _problem,
    _solved_network,
)

HEATERS = 17518.7


def _delivered(machine) -> float:
    return sum(float(s.heat_flow.value) for _key, s in machine.hpr_hot_streams.items())


def test_a_load_fraction_caps_what_the_machine_delivers():
    problem = _problem()

    machine = as_piped_heat_pump(
        problem,
        _solved_network(),
        args={"maximum_evaluations": 1},
        load_fraction=0.4,
    )

    assert 0.0 < _delivered(machine) <= 0.4 * HEATERS * (1.0 + 1e-6)
    assert float(problem.master_zone.config.hpr.load_fraction) == pytest.approx(1.0)
