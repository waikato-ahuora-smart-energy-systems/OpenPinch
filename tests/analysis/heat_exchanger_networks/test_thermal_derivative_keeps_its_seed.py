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


def test_the_cheapest_of_several_seeds_wins_on_the_task_run_for_it(monkeypatch):
    # Two pinch design seeds at multipliers 1.0 (1,650) and 0.2 (1,530),
    # each with its own task; the second is the cheapest. Its outcome named
    # a task rebuilt for it alone, numbered as the first seed's and never
    # run (Codex review on #570).
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()
    seeds = tuple(
        problem.design.pinch_design(
            options={"HENS_DT_CONT_MULTIPLIERS": [multiplier]}
        ).selected_network
        for multiplier in (1.0, 0.2)
    )

    view = problem.design.thermal_derivative(seeds)

    assert view.selected_network.total_annual_cost == seeds[1].total_annual_cost
    assert view.selected_network.task_id in view.result.manifest.task_ids
