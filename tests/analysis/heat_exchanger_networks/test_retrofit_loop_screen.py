"""The loop screen ranks a loop by no more than the heat it can carry."""

from __future__ import annotations


def test_the_screen_ranks_a_loop_by_no_more_than_it_can_carry():
    """A pair with a large cooler and heater but little room for a loop
    must not displace a smaller pair whose loop carries more."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        HeatRecoveryLoop,
        screen_heat_recovery_loops,
    )
    from tests.analysis.heat_exchanger_networks.test_retrofit_loops import _network

    network = _network({"H1->CU": 90.0, "H2->CU": 0.0, "HU->C1": 200.0, "HU->C2": 30.0})
    # H1 could give C1 90 kW by their duties, but a loop between them
    # carries 20 kW at most; the loop to C2 carries all of C2's 30 kW.
    narrow = HeatRecoveryLoop("H1", "C1", 380.0, 20.0)
    whole = HeatRecoveryLoop("H1", "C2", 380.0, 60.0)

    ranked = screen_heat_recovery_loops(network, [narrow, whole], limit=2)

    assert ranked == [whole, narrow]
