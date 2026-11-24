"""Where a heat recovery loop that serves part of two streams is placed.

Each loop exchanger is counter-current, so the hot stream leaves the loop
cooler the approach above the loop's cool end and the cold stream leaves the
loop heater the approach below its warm end. The solve holds exactly that: a
loop forced in at 109.4 degC heats its cold stream to 99.9 degC at a 10 K
approach.
"""

from __future__ import annotations

import pytest


def test_a_partial_loop_is_placed_by_the_ends_its_exchangers_hold():
    """Placing the loop by the other ends understated each side by its flow
    times the loop's 1 K span, which moves the balance point when the flows
    differ."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        LOOP_SPAN,
        heat_recovery_loop_candidates,
    )

    hot, cold = (473.15, 313.15, 2.0), (303.15, 423.15, 1.0)

    (loop,) = heat_recovery_loop_candidates(
        {"H1": hot, "C1": cold}, {"H1": "Site/A", "C1": "Site/B"}, approach=10.0
    )

    # 100.67 kW each way: collected from 200 to 149.67 degC at 2 kW/K and
    # delivered from 30 to 130.67 degC at 1 kW/K.
    assert loop.temperature == pytest.approx(413.3167, abs=1e-3)
    collected = hot[2] * (hot[0] - (loop.temperature - LOOP_SPAN / 2 + 10.0))
    delivered = cold[2] * ((loop.temperature + LOOP_SPAN / 2 - 10.0) - cold[0])
    assert collected == pytest.approx(delivered) == pytest.approx(100.667, abs=1e-3)


def test_a_partial_loop_keeps_the_approach_where_the_streams_enter_it():
    """The other end of each exchanger counts too: the hot stream enters the
    loop cooler against the loop's warm end and the cold stream enters the
    loop heater against its cool end. A hot stream with 200 times the cold
    stream's flow balances 0.25 K above the highest loop it can heat, where
    the cooler would have no feasible duty; the loop stays at that limit."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        LOOP_SPAN,
        heat_recovery_loop_candidates,
    )

    (loop,) = heat_recovery_loop_candidates(
        {"H1": (473.15, 313.15, 200.0), "C1": (303.15, 600.0, 1.0)},
        {"H1": "Site/A", "C1": "Site/B"},
        approach=10.0,
    )

    # 200 degC less the approach is the loop's warm end at most.
    assert loop.temperature == pytest.approx(473.15 - 10.0 - LOOP_SPAN / 2)


def test_a_candidate_records_the_heat_its_loop_can_carry():
    """The screen ranks loops by the utility they could relieve. A partial
    loop can relieve no more than it can carry at its temperature, however
    large its streams' cooler and heater are, so each candidate records it."""
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.loops import (
        heat_recovery_loop_candidates,
    )

    processes = {"H1": "Site/A", "C1": "Site/B"}

    (partial,) = heat_recovery_loop_candidates(
        {"H1": (473.15, 313.15, 1.0), "C1": (303.15, 423.15, 1.0)},
        processes,
        approach=10.0,
    )
    (whole,) = heat_recovery_loop_candidates(
        {"H1": (473.0, 393.0, 2.0), "C1": (313.0, 363.0, 2.0)},
        processes,
        approach=10.0,
    )

    assert partial.capacity == pytest.approx(75.5)
    # Between targets with room for it, a loop can carry the smaller of the
    # two streams' whole duties: 160 kW of cooling, 100 kW of heating.
    assert whole.capacity == pytest.approx(100.0)
