from __future__ import annotations

import numpy as np
from hypothesis import given, settings

from OpenPinch.classes import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import tol
from OpenPinch.lib.schemas.io import TargetInput
from OpenPinch.services.common.problem_table_analysis import (
    create_problem_table_with_t_int,
    problem_table_algorithm,
)
from tests.strategies.stream_segments import segmented_streams


@given(segmented_streams())
@settings(max_examples=40)
def test_segmented_stream_preserves_continuity_and_aggregate_invariants(stream):
    assert all(
        np.allclose(
            left.t_target.period_values,
            right.t_supply.period_values,
            rtol=0.0,
            atol=tol,
        )
        for left, right in zip(stream.segments[:-1], stream.segments[1:])
    )
    assert float(stream.t_supply) == float(stream.segments[0].t_supply)
    assert float(stream.t_target) == float(stream.segments[-1].t_target)
    assert np.isclose(
        float(stream.heat_flow),
        sum(float(segment.heat_flow) for segment in stream.segments),
        rtol=1e-12,
        atol=tol,
    )


@given(segmented_streams())
@settings(max_examples=30)
def test_segment_schema_json_round_trip_preserves_order(stream):
    payload = {
        "streams": [
            {
                "zone": "Site",
                "name": stream.name,
                "segments": [
                    {
                        "name": segment.name,
                        "t_supply": float(segment.t_supply),
                        "t_target": float(segment.t_target),
                        "heat_flow": float(segment.heat_flow),
                    }
                    for segment in stream.segments
                ],
            }
        ]
    }
    first = TargetInput.model_validate(payload)
    second = TargetInput.model_validate_json(first.model_dump_json())
    assert [segment.name for segment in second.streams[0].segments] == [
        segment.name for segment in first.streams[0].segments
    ]
    assert second == first


@given(segmented_streams())
@settings(max_examples=20)
def test_flat_and_segmented_problem_tables_are_property_equivalent(stream):
    flat = StreamCollection(
        [
            Stream(
                name=f"Flat {index}",
                t_supply=segment.t_supply,
                t_target=segment.t_target,
                heat_flow=segment.heat_flow,
                htc=segment.htc,
            )
            for index, segment in enumerate(stream.segments)
        ]
    )
    nested = StreamCollection([stream])

    flat_table = problem_table_algorithm(
        create_problem_table_with_t_int(flat),
        flat.get_hot_streams(),
        flat.get_cold_streams(),
    )
    nested_table = problem_table_algorithm(
        create_problem_table_with_t_int(nested),
        nested.get_hot_streams(),
        nested.get_cold_streams(),
    )
    assert np.allclose(flat_table.data, nested_table.data, equal_nan=True)
