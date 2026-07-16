"""Structural and generated invariants for owner-oriented private helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import OpenPinch
import OpenPinch.classes
from OpenPinch.classes import ProblemTable, Stream, Value
from OpenPinch.classes._heat_exchanger.area import HeatExchangerAreaSlice
from OpenPinch.classes._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.classes._stream.segment import StreamSegment
from OpenPinch.classes.heat_exchanger import HeatExchanger
from OpenPinch.lib.enums import ProblemTableLabel as PT
from OpenPinch.lib.schemas.io import StreamSegmentSchema
from tests.strategies.stream_segments import segmented_streams


def test_owner_package_hierarchy_and_retired_paths() -> None:
    classes_dir = Path(OpenPinch.classes.__file__).parent
    expected = {
        "_heat_exchanger/area.py",
        "_heat_exchanger/period_state.py",
        "_pinch_problem/accessors/component.py",
        "_pinch_problem/accessors/design.py",
        "_pinch_problem/accessors/plot.py",
        "_pinch_problem/accessors/target.py",
        "_pinch_problem/input/loading.py",
        "_pinch_problem/input/semantics.py",
        "_pinch_problem/input/validation.py",
        "_pinch_problem/output/reporting.py",
        "_pinch_problem/output/result_extraction.py",
        "_pinch_problem/periods/aggregation.py",
        "_pinch_problem/periods/execution.py",
        "_pinch_problem/targeting/dispatch.py",
        "_pinch_problem/targeting/execution.py",
        "_pinch_problem/targeting/plan.py",
        "_pinch_workspace/case_inputs.py",
        "_pinch_workspace/comparison.py",
        "_pinch_workspace/execution.py",
        "_pinch_workspace/state.py",
        "_pinch_workspace/views.py",
        "_problem_table/constants.py",
        "_problem_table/equality.py",
        "_problem_table/intervals.py",
        "_stream/profile.py",
        "_stream/segment.py",
        "_stream/segments.py",
        "_stream/thermodynamics.py",
        "_stream/value_state.py",
        "_stream_collection/filters.py",
        "_stream_collection/numeric_view.py",
        "_stream_collection/serialization.py",
        "_stream_collection/sorting.py",
        "_value/coercion.py",
        "_value/units.py",
    }
    assert all((classes_dir / path).is_file() for path in expected)
    assert not (classes_dir / "_problem").exists()
    assert not (classes_dir / "_workspace").exists()
    assert not (classes_dir / "_heat_exchanger_area.py").exists()
    assert not (classes_dir / "_stream_profile.py").exists()


def test_owner_packages_are_cold_importable_without_public_record_aliases() -> None:
    code = """
import OpenPinch
import OpenPinch.classes
import OpenPinch.classes._heat_exchanger.area
import OpenPinch.classes._heat_exchanger.period_state
import OpenPinch.classes._pinch_problem.input.validation
import OpenPinch.classes._pinch_problem.targeting.execution
import OpenPinch.classes._pinch_workspace.state
import OpenPinch.classes._problem_table.intervals
import OpenPinch.classes._stream.segment
import OpenPinch.classes._stream_collection.serialization
import OpenPinch.classes._value.units
assert not hasattr(OpenPinch, 'StreamSegment')
assert not hasattr(OpenPinch.classes, 'StreamSegment')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_heat_exchanger_normalizes_parent_owned_records_from_mappings() -> None:
    exchanger = HeatExchanger(
        kind="recovery",
        source_stream="hot",
        sink_stream="cold",
        source_stream_role="process",
        sink_stream_role="process",
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
    assert isinstance(exchanger.state(), HeatExchangerPeriodState)
    assert isinstance(exchanger.segment_area_contributions[0], HeatExchangerAreaSlice)
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


@given(segmented_streams())
@settings(max_examples=30)
def test_mapping_schema_and_internal_segment_normalization_are_equivalent(
    stream,
) -> None:
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
    parents = (
        Stream(name="mapping", segments=mappings),
        Stream(name="schema", segments=schemas),
        Stream(name="internal", segments=stream.segments),
    )

    expected = np.asarray(
        [
            [mapping["t_supply"], mapping["t_target"], mapping["heat_flow"]]
            for mapping in mappings
        ]
    )
    for parent in parents:
        assert isinstance(parent.segments, tuple)
        assert all(isinstance(segment, StreamSegment) for segment in parent.segments)
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
