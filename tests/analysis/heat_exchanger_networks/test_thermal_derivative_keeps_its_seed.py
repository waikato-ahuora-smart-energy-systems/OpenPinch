"""A standalone thermal derivative is never dearer than its seed.

The thermal derivative minimises hot utility, not total annual cost, so on
its own it can return a network dearer than the one it started from: 9 %
on the reviewer's four-stream case (review finding 29). Decided 2026-11-28:
it keeps that objective, its role before network evolution, but ranks its
seeds alongside its own networks.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.test_full_sequence import (
    _small_problem,
    _use_fake_default_executor,
)


def test_a_standalone_thermal_derivative_is_never_dearer_than_its_seed(monkeypatch):
    # The fake executor's thermal derivative network costs 1,606.5 against
    # its pinch design seed's 1,530.
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()
    seed = problem.design.pinch_design().selected_network

    view = problem.design.thermal_derivative((seed,))

    assert view.selected_network.total_annual_cost <= seed.total_annual_cost
