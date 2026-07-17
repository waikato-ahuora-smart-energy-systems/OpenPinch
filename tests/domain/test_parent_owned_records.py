"""Behavioural invariants for records owned by domain aggregates."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from OpenPinch.contracts.input import StreamSegmentSchema
from OpenPinch.domain.enums import ProblemTableLabel as PT
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.value import Value
from tests.strategies.stream_segments import segmented_streams


def test_heat_exchanger_normalizes_parent_owned_records_from_mappings() -> None:
    exchanger = HeatExchanger(
        kind="recovery",
        source_stream="hot",
        sink_stream="cold",
        source_stream_role="Process",
        sink_stream_role="Process",
        stage=1,
        period_states=[{"period_id": "base", "period_idx": 0, "duty": 10.0}],
        area=2.0,
        segment_area_contributions=[
            {
                "period": "base",
                "hot_segment_identity": "hot.S1",
                "cold_segment_identity": "cold.S1",
                "duty": 10.0,
                "hot_inlet_temperature": 200.0,
                "hot_outlet_temperature": 180.0,
                "cold_inlet_temperature": 100.0,
                "cold_outlet_temperature": 120.0,
                "hot_htc": 10.0,
                "cold_htc": 10.0,
                "overall_htc": 5.0,
                "lmtd": 1.0,
                "area": 2.0,
            }
        ],
    )

    state = exchanger.state()
    area_slice = exchanger.segment_area_contributions[0]
    assert type(state).__name__ == "HeatExchangerPeriodState"
    assert state.period_id == "base"
    assert type(area_slice).__name__ == "HeatExchangerAreaSlice"
    assert area_slice.period == "base"
    assert exchanger.segment_area_by_period == {"base": 2.0}
    assert exchanger.model_dump()["period_states"][0]["period_id"] == "base"
    schema = HeatExchanger.model_json_schema()
    assert "HeatExchangerPeriodState" in schema["$defs"]
    assert "HeatExchangerAreaSlice" in schema["$defs"]


def test_stream_accepts_schema_segment_without_optional_name() -> None:
    schema = StreamSegmentSchema(t_supply=150.0, t_target=100.0, heat_flow=50.0)
    stream = Stream(name="parent", segments=[schema])

    assert stream.segment_count == 1
    assert stream.segments[0].name == "Stream"
    assert stream.segments[0].parent is stream


@seed(20260715)
@given(segmented_streams())
@settings(max_examples=30)
def test_mapping_schema_and_owned_segment_normalization_are_equivalent(stream) -> None:
    mappings = [
        {
            "name": segment.name,
            "t_supply": float(segment.t_supply),
            "t_target": float(segment.t_target),
            "heat_flow": float(segment.heat_flow),
            "dt_cont": float(segment.dt_cont),
            "htc": float(segment.htc),
        }
        for segment in stream.segments
    ]
    schemas = [StreamSegmentSchema.model_validate(mapping) for mapping in mappings]
    owned_source = Stream(name="source", segments=mappings)
    parents = (
        Stream(name="mapping", segments=mappings),
        Stream(name="schema", segments=schemas),
        Stream(name="owned", segments=owned_source.segments),
    )
    expected = np.asarray(
        [
            [mapping["t_supply"], mapping["t_target"], mapping["heat_flow"]]
            for mapping in mappings
        ]
    )

    for parent in parents:
        assert isinstance(parent.segments, tuple)
        assert all(segment.parent is parent for segment in parent.segments)
        actual = np.asarray(
            [
                [
                    float(segment.t_supply),
                    float(segment.t_target),
                    float(segment.heat_flow),
                ]
                for segment in parent.segments
            ]
        )
        np.testing.assert_allclose(actual, expected)


@seed(20260715)
@given(
    values=st.one_of(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=6,
        ),
    ),
    unit=st.sampled_from(["-", "kW", "degC", "delta_degC", "$/y"]),
)
@settings(max_examples=60)
def test_value_dictionary_round_trip_property(values, unit) -> None:
    original = Value(values, unit=unit)
    restored = Value.from_dict(original.to_dict())

    assert restored.unit == original.unit
    np.testing.assert_allclose(restored.period_values, original.period_values)


@seed(20260715)
@given(
    temperatures=st.lists(
        st.integers(min_value=-200, max_value=500),
        min_size=3,
        max_size=8,
        unique=True,
    ),
    candidate=st.floats(
        min_value=-199.5,
        max_value=499.5,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=60)
def test_problem_table_generated_interval_insertion_invariants(
    temperatures,
    candidate,
) -> None:
    ordered = np.asarray(sorted(temperatures, reverse=True), dtype=float)
    assume(ordered[-1] < candidate < ordered[0])
    assume(np.min(np.abs(ordered - candidate)) > 1e-6)
    heat = ordered * 2.0 + 7.0
    table = ProblemTable({PT.T: ordered, PT.H_NET: heat})
    before = {float(t): float(h) for t, h in zip(ordered, heat)}

    assert table.insert_temperature_interval(candidate) == 1
    assert table.shape[0] == len(ordered) + 1
    assert np.all(np.diff(table[PT.T]) < 0.0)
    for temperature, original_heat in before.items():
        index = int(np.flatnonzero(np.isclose(table[PT.T], temperature))[0])
        assert table.loc[index, PT.H_NET] == original_heat
    assert table.insert_temperature_interval(candidate) == 0
    assert table.shape[0] == len(ordered) + 1
