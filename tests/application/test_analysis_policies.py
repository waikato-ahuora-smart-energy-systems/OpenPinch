"""Metric completeness and execution lifecycle properties."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import create_model

from OpenPinch import PinchProblem
from OpenPinch.application._problem.periods.aggregation import weighted_average_output
from OpenPinch.contracts.output import TargetOutput
from OpenPinch.contracts.reporting import TargetResults
from tests.application.test_hpr_period_batch_boundaries import _two_period_payload
from tests.application.test_multiperiod_summary import _target
from tests.contracts.test_hpr_target_simulation_record import _record


def test_new_metric_without_policy_is_rejected():
    from OpenPinch.contracts.report_metrics import metric_specifications

    Extended = create_model("Extended", __base__=TargetResults, new_metric=(float, 0.0))
    with pytest.raises(ValueError, match="new_metric"):
        metric_specifications(Extended)


def test_aggregate_does_not_claim_first_period_simulation_record():
    a, b = _target(period_id="base"), _target(period_id="peak")
    a.hpr_target_simulation_record = _record(period_id="base")
    b.hpr_target_simulation_record = _record(period_id="peak")
    output = weighted_average_output(
        [
            TargetOutput(targets=[a]),
            TargetOutput(targets=[b]),
        ],
        [1, 1],
    )
    assert output.targets[0].hpr_target_simulation_record is None
    assert a.hpr_target_simulation_record.period_id == "base"


@settings(max_examples=8, deadline=None)
@given(st.lists(st.sampled_from(["base", "peak"]), min_size=1, max_size=5))
def test_generated_period_sequences_never_leak_identity(periods):
    p = PinchProblem(_two_period_payload(), project_name="Site")
    for sid in periods:
        p.target.direct_heat_integration(zone="AreaA", period_id=sid)
        expected_index = p.period_ids[sid]
        assert all(
            t.period_id == sid and t.period_idx == expected_index
            for t in p.results.targets
        )
        assert all(t.provenance.period_ids == (sid,) for t in p.results.targets)


def test_unknown_derived_metric_requires_implementation():
    from OpenPinch.contracts.report_metrics import report_field

    Extended = create_model(
        "ExtendedDerived",
        __base__=TargetResults,
        extra_derived=(float, report_field("extra_derived", "derived", 1.0)),
    )
    rows = [
        Extended.model_validate(_target(period_id=sid).model_dump())
        for sid in ("base", "peak")
    ]
    with pytest.raises(ValueError, match="extra_derived"):
        weighted_average_output([TargetOutput(targets=[row]) for row in rows], [1, 1])


def test_each_declared_numeric_metric_obeys_its_policy():
    from OpenPinch.contracts.report_metrics import (
        AggregationPolicy,
        metric_specifications,
    )
    from OpenPinch.domain.value import Value

    for name, spec in metric_specifications(TargetResults).items():
        if spec.aggregation not in {
            AggregationPolicy.WEIGHTED_MEAN,
            AggregationPolicy.MAXIMUM,
        }:
            continue
        a, b = _target(period_id="base"), _target(period_id="peak")
        setattr(
            a,
            name,
            2.0 if spec.representation == "scalar" else Value(2.0, spec.unit or "kW"),
        )
        setattr(
            b,
            name,
            6.0 if spec.representation == "scalar" else Value(6.0, spec.unit or "kW"),
        )
        # The HPR total is derived from an explicit complete cost breakdown.
        if name in {"hpr_operating_cost", "hpr_annualized_capital_cost"}:
            other = (
                "hpr_operating_cost"
                if name == "hpr_annualized_capital_cost"
                else "hpr_annualized_capital_cost"
            )
            setattr(a, other, Value(1.0, spec.unit))
            setattr(b, other, Value(1.0, spec.unit))
        row = weighted_average_output(
            [TargetOutput(targets=[a]), TargetOutput(targets=[b])], [1, 3]
        ).targets[0]
        assert float(getattr(row, name)) == pytest.approx(
            6.0 if spec.aggregation is AggregationPolicy.MAXIMUM else 5.0
        ), name
