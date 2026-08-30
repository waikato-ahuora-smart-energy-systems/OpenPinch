"""Weighted aggregation and scalar feasibility partition tests."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.numerics import g_ineq_penalty
from OpenPinch.analysis.utility_placement import penalties
from OpenPinch.analysis.utility_placement.penalties import (
    aggregate_weighted_objective,
    feasible_objective_scalar,
    infeasible_objective_scalar,
)
from OpenPinch.domain.enums import PenaltyForm


def test_g_penalty_is_squared_and_dimensionless() -> None:
    assert penalties.g_penalty(
        hot_fallback_duty=20.0,
        cold_fallback_duty=10.0,
        required_hot_duty=100.0,
        required_cold_duty=50.0,
    ) == pytest.approx(0.8)
    assert penalties.g_penalty(
        hot_fallback_duty=0.0,
        cold_fallback_duty=0.0,
        required_hot_duty=0.0,
        required_cold_duty=0.0,
    ) == 0.0


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 1_000.0])
def test_g_penalty_is_invariant_under_common_duty_scaling(scale) -> None:
    observed = penalties.g_penalty(
        hot_fallback_duty=20.0 * scale,
        cold_fallback_duty=10.0 * scale,
        required_hot_duty=100.0 * scale,
        required_cold_duty=50.0 * scale,
    )
    assert observed == pytest.approx(0.8)


@pytest.mark.parametrize(
    "values",
    [
        (-1.0, 0.0, 1.0, 1.0),
        (0.0, float("nan"), 1.0, 1.0),
        (0.0, 0.0, float("inf"), 1.0),
        (1.0, 0.0, 0.0, 1.0),
    ],
)
def test_g_penalty_rejects_invalid_duties(values) -> None:
    hot_fallback, cold_fallback, hot_required, cold_required = values

    with pytest.raises(ValueError):
        penalties.g_penalty(
            hot_fallback_duty=hot_fallback,
            cold_fallback_duty=cold_fallback,
            required_hot_duty=hot_required,
            required_cold_duty=cold_required,
        )


@given(
    hot_fraction=st.floats(min_value=0.0, max_value=1.0),
    cold_fraction=st.floats(min_value=0.0, max_value=1.0),
    hot_required=st.floats(min_value=1.0, max_value=1e6),
    cold_required=st.floats(min_value=1.0, max_value=1e6),
    scale=st.floats(min_value=0.01, max_value=100.0),
)
def test_g_penalty_scale_invariance_property(
    hot_fraction,
    cold_fraction,
    hot_required,
    cold_required,
    scale,
) -> None:
    baseline = penalties.g_penalty(
        hot_fallback_duty=hot_fraction * hot_required,
        cold_fallback_duty=cold_fraction * cold_required,
        required_hot_duty=hot_required,
        required_cold_duty=cold_required,
    )
    scaled = penalties.g_penalty(
        hot_fallback_duty=hot_fraction * hot_required * scale,
        cold_fallback_duty=cold_fraction * cold_required * scale,
        required_hot_duty=hot_required * scale,
        required_cold_duty=cold_required * scale,
    )

    canonical = g_ineq_penalty(
        [hot_fraction, cold_fraction],
        form=PenaltyForm.SQUARE,
    )
    assert baseline == pytest.approx(canonical)
    assert scaled == pytest.approx(baseline)


def test_penalized_feasible_scalar_is_bounded_and_monotone() -> None:
    observed = [
        penalties.penalized_feasible_objective_scalar(0.4, penalty)
        for penalty in (0.0, 0.1, 1.0, 100.0)
    ]

    assert observed == sorted(observed)
    assert observed[0] == pytest.approx(0.4)
    assert all(0.0 < value < 1.0 for value in observed)


@pytest.mark.parametrize(
    ("base", "penalty"),
    [(0.0, 0.0), (1.0, 0.0), (0.5, -1.0), (0.5, float("nan"))],
)
def test_penalized_feasible_scalar_rejects_invalid_inputs(base, penalty) -> None:
    with pytest.raises(ValueError):
        penalties.penalized_feasible_objective_scalar(base, penalty)


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
