"""Multistart restarts: the warm start seeds the first, the rest explore."""

from __future__ import annotations

import numpy as np

import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter
from OpenPinch.contracts.hpr import HPRBackendResult
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS
from OpenPinch.optimisation.errors import NoOptimisationCandidatesError

from .helpers import _base_args

_BOUNDS = [(0.0, 1.0), (10.0, 20.0)]


def _search(*, starts, restarts=4, search_seed=0):
    calls = []

    def optimiser(problem, *, method, options):
        calls.append(
            (
                [tuple(float(v) for v in point) for point in problem.initial_points],
                options.seed,
            )
        )
        raise NoOptimisationCandidatesError("none")

    adapter.run_hpr_candidate_search(
        objective=lambda x, _args, debug=False: HPRBackendResult.failure(),
        initial_points=starts,
        bounds=_BOUNDS,
        args=_base_args(max_multi_start=restarts, search_seed=search_seed),
        optimiser=optimiser,
    )
    return calls


def test_only_the_first_restart_starts_from_the_warm_start():
    calls = _search(starts=np.array([[0.5, 15.0]]))

    assert calls[0][0] == [(0.5, 15.0)]
    later = [points for points, _ in calls[1:]]
    assert all(len(points) == 1 for points in later)
    assert all(points != [(0.5, 15.0)] for points in later)


def test_later_restarts_start_from_distinct_points_inside_the_bounds():
    calls = _search(starts=np.array([[0.5, 15.0]]))

    points = [points[0] for points, _ in calls[1:]]
    assert len(set(points)) == len(points) == 3
    for x, y in points:
        assert 0.0 <= x <= 1.0
        assert 10.0 <= y <= 20.0
    # Latin hypercube: one point in each third of every coordinate's range.
    assert sorted(int(x * 3) for x, _ in points) == [0, 1, 2]


def test_without_a_warm_start_every_restart_explores():
    calls = _search(starts=None, restarts=3)

    assert all(len(points) == 1 for points, _ in calls)
    assert len({points[0] for points, _ in calls}) == 3


def test_the_restarts_are_reproducible_and_shift_with_the_search_seed():
    first = _search(starts=np.array([[0.5, 15.0]]))
    again = _search(starts=np.array([[0.5, 15.0]]))
    shifted = _search(starts=np.array([[0.5, 15.0]]), search_seed=7)

    assert first == again
    assert [seed for _, seed in first] == [0, 1, 2, 3]
    assert [seed for _, seed in shifted] == [7, 8, 9, 10]
    assert [points for points, _ in shifted[1:]] != [points for points, _ in first[1:]]


def test_the_search_seed_is_a_setting_defaulting_to_zero():
    assert CONFIG_FIELD_SPECS["HPR_SEARCH_SEED"].default == 0
