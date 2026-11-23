"""Which of its utility level targets a network holds.

A pinch design that cannot hold the levels on one side of the pinch is
designed there without them, but its network still records every target, so
that it does not rank as a level holder. A step that keeps a design's levels
must keep only those the design holds: held to a target its parent missed,
it could fail or move the split the parent chose.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.test_retrofit_loops import _network


def _with_targets(duties: dict, targets: dict):
    return _network(duties).model_copy(
        update={"source_metadata": {"utility_level_targets": targets}}
    )


def test_a_network_holds_the_levels_whose_duty_meets_the_target():
    from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
        held_utility_levels,
        holds_utility_levels,
    )

    duties = {"H1->CU": 90.0, "H2->CU": 10.0, "HU->C1": 200.0, "HU->C2": 30.0}
    # 100 kW of cooling as targeted; 230 kW of heating against 250 kW.
    missed = _with_targets(duties, {"CU": [100.0], "HU": [250.0]})
    met = _with_targets(duties, {"CU": [100.0], "HU": [230.0]})

    assert held_utility_levels(missed) == {"CU": [100.0]}
    assert not holds_utility_levels(missed)
    assert held_utility_levels(met) == {"CU": [100.0], "HU": [230.0]}
    assert holds_utility_levels(met)
    assert held_utility_levels(_network(duties)) == {}
