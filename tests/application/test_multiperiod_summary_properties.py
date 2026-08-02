"""Property coverage for aligned multi-period targeting summaries."""

from __future__ import annotations

import pytest
from hypothesis import given, seed, settings

from OpenPinch.application._problem.periods.aggregation import (
    combine_period_outputs,
    weighted_average_output,
)
from OpenPinch.contracts.output import TargetOutput
from tests.strategies.period_outputs import aligned_period_outputs


@seed(20260715)
@settings(max_examples=50, deadline=None)
@given(case=aligned_period_outputs())
def test_generated_period_outputs_round_trip_through_json(case):
    for output in case.outputs:
        restored = TargetOutput.model_validate_json(output.model_dump_json())
        assert restored.model_dump(mode="json") == output.model_dump(mode="json")

    weighted = weighted_average_output(case.outputs, case.weights)
    restored_weighted = TargetOutput.model_validate_json(weighted.model_dump_json())
    assert restored_weighted.model_dump(mode="json") == weighted.model_dump(mode="json")


@seed(20260715)
@settings(max_examples=75, deadline=None)
@given(case=aligned_period_outputs())
def test_weighted_summary_preserves_order_range_and_inputs(case):
    before = [output.model_dump(mode="json") for output in case.outputs]

    weighted = weighted_average_output(case.outputs, case.weights)

    assert [target.scope for target in weighted.targets] == [
        "Site",
        "Site/Process",
    ]
    hot_utility = weighted.targets[0].Qh.value
    tolerance = max(1.0, max(case.primary_hot_utility)) * 1e-12
    assert min(case.primary_hot_utility) - tolerance <= hot_utility
    assert hot_utility <= max(case.primary_hot_utility) + tolerance
    expected = sum(
        value * weight
        for value, weight in zip(
            case.primary_hot_utility,
            case.weights,
            strict=True,
        )
    ) / sum(case.weights)
    assert hot_utility == pytest.approx(expected)
    assert [output.model_dump(mode="json") for output in case.outputs] == before


@seed(20260715)
@settings(max_examples=75, deadline=None)
@given(case=aligned_period_outputs())
def test_weighted_summary_is_invariant_to_weight_scale(case):
    baseline = weighted_average_output(case.outputs, case.weights)
    scaled = weighted_average_output(
        case.outputs,
        tuple(weight * 7.0 for weight in case.weights),
    )

    assert [target.scope for target in scaled.targets] == [
        target.scope for target in baseline.targets
    ]
    for baseline_target, scaled_target in zip(
        baseline.targets,
        scaled.targets,
        strict=True,
    ):
        assert scaled_target.Qh.value == pytest.approx(baseline_target.Qh.value)
        assert scaled_target.Qc.value == pytest.approx(baseline_target.Qc.value)
        assert scaled_target.area.value == pytest.approx(baseline_target.area.value)


@seed(20260715)
@settings(max_examples=50, deadline=None)
@given(case=aligned_period_outputs())
def test_optional_pinch_diagnostics_have_explicit_partial_missing_policy(case):
    weighted = weighted_average_output(case.outputs, case.weights)
    cold_pinch = weighted.targets[0].pinch_temp.cold_temp

    if all(case.cold_pinch_present):
        assert cold_pinch is not None
        cold_values = tuple(
            90.0 + hot_utility / 100.0 for hot_utility in case.primary_hot_utility
        )
        tolerance = max(1.0, max(cold_values)) * 1e-12
        assert min(cold_values) - tolerance <= cold_pinch.value
        assert cold_pinch.value <= max(cold_values) + tolerance
    else:
        assert cold_pinch is None

    combined = combine_period_outputs(case.outputs)
    assert [target.period_id for target in combined.targets] == [
        "base",
        "base",
        "peak",
        "peak",
    ]
