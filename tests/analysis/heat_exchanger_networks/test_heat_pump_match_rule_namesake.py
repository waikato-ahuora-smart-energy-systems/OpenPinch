"""The match rules fall on the marked stream, not on a namesake.

With a hot stream ``X`` in two branches of the zone tree, one of them a
condenser, the cooler ban belongs to the condenser alone. On the other it
would take the cooler away from an ordinary process stream.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_heat_pump_stream_keys import (
    _two_branches,
)


def test_the_cooler_ban_falls_on_the_condenser_and_not_its_namesake():
    prepared = problem_to_solver_arrays(_two_branches(), 10.0)
    hot = prepared.axis_maps["hot_process_streams"]

    coolers = prepared.arrays["forbidden_cu"]

    assert {key: int(coolers[0][index]) for key, index in hot.items()} == {
        "P1.X": 1,
        "P1.H": 0,
        "P2.X": 0,
    }
