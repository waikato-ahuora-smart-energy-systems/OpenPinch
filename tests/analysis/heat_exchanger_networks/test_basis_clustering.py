"""Weighted k-medoids over operating periods, for the median design basis."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.basis.clustering import (
    weighted_k_medoids,
)

# Six periods in two well-separated groups on two features.
FEATURES = np.array(
    [[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [5.0, 5.0], [5.3, 4.9], [4.8, 5.2]]
)
WEIGHTS = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 3.0])


def _groups(result, ids=None):
    """Clusters as sets of row ids, with each cluster's medoid id."""

    ids = list(range(len(result.labels))) if ids is None else ids
    return {
        (ids[medoid], frozenset(ids[i] for i, c in enumerate(result.labels) if c == n))
        for n, medoid in enumerate(result.medoids)
    }


def test_one_cluster_takes_the_weighted_medoid_of_every_period():
    result = weighted_k_medoids(FEATURES, WEIGHTS, 1)

    distances = np.linalg.norm(FEATURES[:, None] - FEATURES[None], axis=2)
    assert result.medoids == (int(np.argmin(distances @ WEIGHTS)),)
    assert set(result.labels) == {0}


def test_two_clusters_split_the_two_groups():
    result = weighted_k_medoids(FEATURES, WEIGHTS, 2)

    assert {members for _medoid, members in _groups(result)} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
    }


def test_as_many_clusters_as_periods_makes_each_period_its_own_medoid():
    result = weighted_k_medoids(FEATURES, WEIGHTS, len(FEATURES))

    assert sorted(result.medoids) == list(range(len(FEATURES)))
    assert all(result.medoids[c] == i for i, c in enumerate(result.labels))


def test_the_period_order_does_not_change_the_clusters():
    order = [4, 0, 5, 2, 3, 1]

    shuffled = weighted_k_medoids(FEATURES[order], WEIGHTS[order], 2)

    assert _groups(shuffled, order) == _groups(weighted_k_medoids(FEATURES, WEIGHTS, 2))


def test_a_duplicated_period_clusters_as_one_with_double_weight():
    duplicated = weighted_k_medoids(
        np.vstack([FEATURES, FEATURES[[3]]]), np.append(WEIGHTS, WEIGHTS[3]), 2
    )
    doubled = weighted_k_medoids(
        FEATURES, WEIGHTS + np.eye(len(WEIGHTS))[3] * WEIGHTS[3], 2
    )

    as_original = list(range(len(FEATURES))) + [3]
    assert _groups(duplicated, as_original) == _groups(doubled)


def test_scaling_every_weight_changes_nothing():
    assert _groups(weighted_k_medoids(FEATURES, 7.5 * WEIGHTS, 2)) == _groups(
        weighted_k_medoids(FEATURES, WEIGHTS, 2)
    )


def test_a_heavy_enough_period_becomes_its_clusters_medoid():
    heavy = WEIGHTS.copy()
    heavy[4] = 100.0

    result = weighted_k_medoids(FEATURES, heavy, 2)

    assert 4 in result.medoids


def test_more_clusters_than_periods_is_rejected():
    with pytest.raises(ValueError, match="between 1 and 6"):
        weighted_k_medoids(FEATURES, WEIGHTS, 7)


def test_identical_periods_as_medoids_each_keep_their_own_cluster():
    features = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])

    result = weighted_k_medoids(features, np.ones(3), 3)

    assert sorted(result.medoids) == [0, 1, 2]
    assert sorted(result.labels) == [0, 1, 2]
