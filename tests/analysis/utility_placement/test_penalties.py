"""Weighted aggregation and scalar feasibility partition tests."""

from __future__ import annotations

import itertools

import pytest

from OpenPinch.analysis.utility_placement.penalties import (
    aggregate_weighted_objective,
    default_utility_penalty,
    feasible_objective_scalar,
    infeasible_objective_scalar,
)


def test_default_utility_penalty_excludes_positive_hu_and_cu_duty() -> None:
    assert default_utility_penalty(
        names=("steam", "HU", "cu"),
        duties=(20.0, 5.0, 10.0),
        reference_duty=100.0,
    ) == pytest.approx(0.15)
    assert default_utility_penalty(
        names=("HU", "CU"),
        duties=(0.0, 0.0),
        reference_duty=100.0,
    ) == 0.0


def test_default_utility_penalty_is_case_and_order_independent() -> None:
    pairs = ((" hu ", 5.0), ("Process steam", 20.0), ("Cu", 10.0))
    observed = {
        default_utility_penalty(
            names=tuple(name for name, _ in permutation),
            duties=tuple(duty for _, duty in permutation),
            reference_duty=100.0,
        )
        for permutation in itertools.permutations(pairs)
    }

    assert len(observed) == 1
    assert next(iter(observed)) == pytest.approx(0.15)


def test_default_utility_penalty_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        default_utility_penalty(
            names=("HU",),
            duties=(),
            reference_duty=1.0,
        )
    for reference, duties in (
        (-1.0, (0.0,)),
        (float("nan"), (0.0,)),
        (1.0, (-1.0,)),
        (1.0, (float("inf"),)),
    ):
        with pytest.raises(ValueError, match="finite and non-negative"):
            default_utility_penalty(
                names=("HU",),
                duties=duties,
                reference_duty=reference,
            )


def test_weighted_aggregation_preserves_raw_weights_and_zero_weight_terms() -> None:
    assert aggregate_weighted_objective(
        (10.0, -4.0, 999.0), (2.0, 3.0, 0.0)
    ) == pytest.approx(8.0)


def test_weighted_aggregation_is_permutation_stable() -> None:
    pairs = ((1e16, 1.0), (1.0, 1.0), (-1e16, 1.0), (3.0, 2.0))
    observed = {
        aggregate_weighted_objective(
            tuple(value for value, _ in permutation),
            tuple(weight for _, weight in permutation),
        )
        for permutation in itertools.permutations(pairs)
    }
    assert len(observed) == 1


def test_weighted_aggregation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        aggregate_weighted_objective((1.0,), ())
    with pytest.raises(ValueError, match="finite"):
        aggregate_weighted_objective((float("nan"),), (1.0,))
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_weighted_objective((1.0,), (-1.0,))


def test_feasible_and_infeasible_scalars_are_disjoint_and_monotone() -> None:
    feasible = [
        feasible_objective_scalar(cost, scale=10.0)
        for cost in (-100.0, -1.0, 0.0, 2.0, 100.0)
    ]
    infeasible = [
        infeasible_objective_scalar(value) for value in (0.0, 0.1, 1.0, 100.0)
    ]
    assert feasible == sorted(feasible)
    assert infeasible == sorted(infeasible)
    assert all(0.0 < value < 1.0 for value in feasible)
    assert all(1.0 <= value < 2.0 for value in infeasible)
    assert max(feasible) < min(infeasible)


@pytest.mark.parametrize(
    ("cost", "scale"),
    [(float("nan"), 1.0), (0.0, float("inf")), (0.0, 0.0)],
)
def test_feasible_scalar_rejects_invalid_inputs(cost: float, scale: float) -> None:
    with pytest.raises(ValueError):
        feasible_objective_scalar(cost, scale=scale)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_infeasible_scalar_rejects_invalid_violation(value: float) -> None:
    with pytest.raises(ValueError):
        infeasible_objective_scalar(value)
