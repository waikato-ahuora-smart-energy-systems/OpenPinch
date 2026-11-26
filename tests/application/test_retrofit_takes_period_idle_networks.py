"""A retrofit takes a network that idles an exchanger in a period.

On the lukewarm period case a design on every period idles its waste heat
matches in the third period, where the cold stream reaches some of them
hotter than the waste heat stream. Rated with every exchanger free to run
in every period, such a network has no solution, and the retrofit refused
it as not meeting its targets at its installed areas.
"""

from __future__ import annotations

import pytest

from tests.application import test_overdesign_keeps_period_idle_networks as lukewarm

pytestmark = lukewarm.pytestmark


def test_a_retrofit_takes_every_network_a_design_on_every_period_returns():
    designed = lukewarm.lukewarm_period_problem().design.heat_exchanger_network(
        design_basis="all", width=0, depth=0
    )
    networks = [outcome.network for outcome in designed.result.ranked_networks[:8]]

    for network in networks:
        result = lukewarm.lukewarm_period_problem().design.retrofit(
            network, max_modifications=0
        )
        (existing,) = result.frontier
        assert existing.utility_cost == pytest.approx(network.utility_cost, rel=1e-6)
