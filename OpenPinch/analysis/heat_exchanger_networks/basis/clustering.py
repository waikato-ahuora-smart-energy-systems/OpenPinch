"""Weighted k-medoids (PAM: build, then swap) over operating periods.

Plain numpy, so it runs wherever OpenPinch does (Pyodide included), and
deterministic: the build step fixes the first medoids and ties go to the
lowest row. Each row is a period with a weight; a medoid minimises the sum of
weight times distance over its cluster's members.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A swap must lower the weighted cost by more than this share of it.
_SWAP_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Clustering:
    """Medoid rows, in build order, and each row's cluster (a medoid position)."""

    medoids: tuple[int, ...]
    labels: tuple[int, ...]


def weighted_k_medoids(features, weights, k: int) -> Clustering:
    """``k`` clusters of the rows of ``features``, weighted by ``weights``.

    Distances are Euclidean. Build adds, ``k`` times, the row that lowers the
    weighted cost most; swap then replaces a medoid by a non-medoid while
    that lowers the cost, taking the best swap each pass. Rows join their
    nearest medoid, a tie going to the earlier medoid.
    """

    features = np.asarray(features, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rows = len(features)
    if not 1 <= k <= rows:
        raise ValueError(f"k must be between 1 and {rows}; got {k}.")
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)

    def cost(medoids: list[int]) -> float:
        return float(weights @ distances[:, medoids].min(axis=1))

    medoids: list[int] = []
    for _cluster in range(k):
        candidates = [row for row in range(rows) if row not in medoids]
        medoids.append(min(candidates, key=lambda row: (cost(medoids + [row]), row)))

    current = cost(medoids)
    while True:
        best = None
        for position in range(k):
            for row in range(rows):
                if row in medoids:
                    continue
                trial = medoids[:position] + [row] + medoids[position + 1 :]
                value = cost(trial)
                if value < current - _SWAP_TOLERANCE * max(current, 1.0) and (
                    best is None or value < best[0]
                ):
                    best = (value, trial)
        if best is None:
            break
        current, medoids = best

    labels = tuple(int(np.argmin(distances[row, medoids])) for row in range(rows))
    return Clustering(medoids=tuple(medoids), labels=labels)


__all__ = ["Clustering", "weighted_k_medoids"]
