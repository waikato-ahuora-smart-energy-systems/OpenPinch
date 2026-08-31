"""Property-based invariants for Unit 2 numerical placement services."""

from __future__ import annotations

import pickle
from decimal import Decimal, localcontext

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch.analysis.utility_placement.evaluation import PlacementEvaluationSession
from OpenPinch.analysis.utility_placement.penalties import (
    aggregate_weighted_objective,
    feasible_objective_scalar,
    infeasible_objective_scalar,
)
from OpenPinch.analysis.utility_placement.thermodynamics import stream_entropy_change
from tests.analysis.utility_placement.test_evaluation import _Adapter, _case


@settings(max_examples=100, derandomize=True, deadline=None)
@given(
    duty=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    inlet=st.floats(
        min_value=1.0, max_value=2000.0, allow_nan=False, allow_infinity=False
    ),
    ratio=st.floats(
        min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False
    ).filter(lambda value: abs(value - 1.0) > 1e-8),
)
def test_entropy_kernel_matches_decimal_oracle(duty, inlet, ratio) -> None:
    outlet = inlet * ratio
    observed = stream_entropy_change(duty, inlet, outlet)
    with localcontext() as context:
        context.prec = 50
        d_duty = Decimal(str(duty))
        d_inlet = Decimal(str(inlet))
        d_outlet = Decimal(str(outlet))
        expected = float(d_duty / abs(d_outlet - d_inlet) * (d_outlet / d_inlet).ln())
    assert observed == pytest.approx(expected, rel=2e-12, abs=1e-12)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(
    cost=st.floats(
        min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=1e-12, max_value=1e12, allow_nan=False, allow_infinity=False
    ),
    violation=st.floats(
        min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False
    ),
)
def test_penalty_partition_is_bounded(cost, scale, violation) -> None:
    feasible = feasible_objective_scalar(cost, scale=scale)
    infeasible = infeasible_objective_scalar(violation)
    assert 0.0 < feasible < 1.0 <= infeasible < 2.0


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_weighted_aggregation_is_order_independent(pairs) -> None:
    forward = aggregate_weighted_objective(
        tuple(value for value, _ in pairs), tuple(weight for _, weight in pairs)
    )
    reverse = aggregate_weighted_objective(
        tuple(value for value, _ in reversed(pairs)),
        tuple(weight for _, weight in reversed(pairs)),
    )
    assert forward == reverse


def test_evaluation_session_pickle_reconstructs_empty_process_local_memo() -> None:
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    session.evaluate(model.initial_points[0])

    restored = pickle.loads(pickle.dumps(session))

    assert restored.evaluation_count == 0
    assert restored.memo_hit_count == 0
    assert restored.evaluate(model.initial_points[0]).feasible
