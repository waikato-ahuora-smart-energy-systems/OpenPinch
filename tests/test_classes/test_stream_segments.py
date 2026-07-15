from __future__ import annotations

import pickle
from copy import deepcopy

import numpy as np
import pytest

import OpenPinch
from OpenPinch import PinchWorkspace, StreamSegment
from OpenPinch.classes import Stream
from OpenPinch.classes._problem._validation import validate_problem_inputs
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import tol
from OpenPinch.lib.schemas.io import TargetInput
from OpenPinch.services.common.problem_table_analysis import (
    create_problem_table_with_t_int,
    problem_table_algorithm,
)
from OpenPinch.services.input_data_processing.data_preparation import prepare_problem
from OpenPinch.utils.stream_linearisation import align_temperature_heat_profiles


def _hot_segments() -> list[StreamSegment]:
    return [
        StreamSegment(name="S1", t_supply=200.0, t_target=150.0, heat_flow=50.0),
        StreamSegment(name="S2", t_supply=150.0, t_target=100.0, heat_flow=100.0),
    ]


def test_segmented_parent_derives_aggregates_and_owns_ordered_children():
    stream = Stream(name="Variable CP", segments=_hot_segments())

    assert stream.has_segments
    assert stream.segment_count == 2
    assert isinstance(stream.segments, tuple)
    assert [segment.name for segment in stream.segments] == ["S1", "S2"]
    assert all(segment.parent is stream for segment in stream.segments)
    assert float(stream.t_supply) == pytest.approx(200.0)
    assert float(stream.t_target) == pytest.approx(100.0)
    assert float(stream.heat_flow) == pytest.approx(150.0)
    assert float(stream.CP) == pytest.approx(1.5)


def test_segmented_parent_rejects_discontinuity_without_reordering():
    with pytest.raises(ValueError, match="target temperature must match"):
        Stream(
            name="Gap",
            segments=[
                StreamSegment(t_supply=200.0, t_target=160.0, heat_flow=40.0),
                StreamSegment(t_supply=150.0, t_target=100.0, heat_flow=100.0),
            ],
        )


def test_segmented_parent_rejects_an_explicit_empty_profile():
    with pytest.raises(ValueError, match="at least one segment"):
        Stream(name="Empty", segments=[])


def test_segment_mutation_is_transactional_and_updates_parent_revision():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    original_revision = stream._numeric_revision

    with pytest.raises(ValueError, match="target temperature must match"):
        stream.segments[0].t_target = 140.0

    assert float(stream.segments[0].t_target) == pytest.approx(150.0)
    stream.segments[0].heat_flow = 60.0
    assert float(stream.heat_flow) == pytest.approx(160.0)
    assert stream._numeric_revision > original_revision


def test_multiperiod_continuity_is_enforced_for_every_period():
    with pytest.raises(ValueError, match="in every period"):
        Stream(
            name="Multiperiod gap",
            segments=[
                StreamSegment(
                    t_supply=[200.0, 210.0],
                    t_target=[150.0, 160.0],
                    heat_flow=[50.0, 50.0],
                ),
                StreamSegment(
                    t_supply=[150.0, 159.0],
                    t_target=[100.0, 110.0],
                    heat_flow=[100.0, 100.0],
                ),
            ],
        )


def test_copy_and_expanded_numeric_view_preserve_parent_identity():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    copied = deepcopy(stream)
    collection = StreamCollection()
    collection.add(copied, key="Site.Variable CP")

    view = collection.segment_numeric_view()
    assert copied.segments[0].parent is copied
    assert view.parent_index.tolist() == [0, 0]
    assert view.segment_index.tolist() == [0, 1]
    assert view.parent_name.tolist() == ["Variable CP", "Variable CP"]
    assert view.parent_key.tolist() == ["Site.Variable CP", "Site.Variable CP"]
    assert view.period_index.tolist() == [0, 0]
    assert collection.to_dict()["name"] == ["Variable CP"]
    expanded = collection.to_dict(expand_segments=True)
    assert expanded["parent_name"] == [
        "Variable CP",
        "Variable CP",
    ]
    assert expanded["parent_key"] == ["Site.Variable CP", "Site.Variable CP"]
    assert expanded["segment_identity"] == [
        "Site.Variable CP.S1",
        "Site.Variable CP.S2",
    ]


def test_expanded_numeric_view_replaces_stale_cache_signatures():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    collection = StreamCollection([stream])

    first = collection.segment_numeric_view()
    stream.update_segment(0, heat_flow=60.0)
    second = collection.segment_numeric_view()

    assert second is not first
    assert second.heat_flow.tolist() == pytest.approx([60.0, 100.0])
    assert len(collection._numeric_cache) == 1


def test_attached_segment_rejects_parent_controlled_metadata_changes():
    stream = Stream(name="Variable CP", segments=_hot_segments())

    with pytest.raises(ValueError, match="controlled by its parent"):
        stream.segments[0].active = False
    with pytest.raises(ValueError, match="controlled by its parent"):
        stream.segments[0].is_process_stream = False
    with pytest.raises(ValueError, match="controlled by its parent"):
        stream.segments[0].dt_cont_multiplier = 2.0
    with pytest.raises(ValueError, match="controlled by its parent"):
        stream.segments[0].set_period_context(
            period_ids={"peak": 0},
            weights=np.asarray([1.0]),
            num_periods=1,
        )

    assert stream.segments[0].active is True
    assert stream.segments[0].is_process_stream is True


def test_replacement_segments_inherit_existing_period_context():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    stream.set_period_context(
        period_ids={"winter": 0, "summer": 1},
        weights=np.asarray([0.4, 0.6]),
        num_periods=2,
    )

    stream.replace_segments(_hot_segments())

    assert all(
        segment.period_ids == {"winter": 0, "summer": 1} for segment in stream.segments
    )
    assert stream.num_periods == 2
    assert stream.heat_flow.num_periods == 2
    assert all(segment.num_periods == 2 for segment in stream.segments)


def test_segmented_parent_rejects_nonpositive_segment_htc():
    with pytest.raises(ValueError, match="HTC must be positive"):
        Stream(
            name="Invalid HTC",
            segments=[
                StreamSegment(
                    t_supply=200.0,
                    t_target=150.0,
                    heat_flow=50.0,
                    htc=0.0,
                )
            ],
        )


def test_segmented_utility_inversion_reverses_the_ordered_profile():
    utility = Stream(
        name="Segmented utility",
        is_process_stream=False,
        segments=_hot_segments(),
    )

    utility.invert()

    assert utility.is_process_stream is True
    assert float(utility.t_supply) == pytest.approx(100.0)
    assert float(utility.t_target) == pytest.approx(200.0)
    assert [float(segment.t_supply) for segment in utility.segments] == [100.0, 150.0]
    assert [float(segment.t_target) for segment in utility.segments] == [150.0, 200.0]
    assert all(segment.parent is utility for segment in utility.segments)


def test_pickle_round_trip_preserves_order_ownership_and_continuity():
    original = Stream(name="Variable CP", segments=_hot_segments())

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is Stream
    assert all(type(segment) is StreamSegment for segment in restored.segments)
    assert [segment.name for segment in restored.segments] == ["S1", "S2"]
    assert all(segment.parent is restored for segment in restored.segments)
    assert float(restored.segments[0].t_target) == pytest.approx(
        float(restored.segments[1].t_supply)
    )


def test_stream_refactor_preserves_public_class_identity_and_defining_module():
    assert OpenPinch.StreamSegment is StreamSegment
    assert Stream.__module__ == "OpenPinch.classes.stream"
    assert StreamSegment.__module__ == "OpenPinch.classes.stream"


def test_workspace_bundle_and_case_copy_preserve_nested_segment_order(tmp_path):
    payload = {
        "streams": [
            {
                "zone": "Site",
                "name": "Variable CP",
                "segments": [
                    {"name": "S1", "t_supply": 200, "t_target": 150, "heat_flow": 50},
                    {"name": "S2", "t_supply": 150, "t_target": 100, "heat_flow": 100},
                ],
            }
        ],
        "utilities": [],
    }
    workspace = PinchWorkspace(payload, project_name="Site")
    workspace.copy_case(source_name="baseline", new_name="copied", activate=False)

    restored = PinchWorkspace.load_bundle(
        workspace.save_bundle(tmp_path / "segmented-workspace.json")
    )
    stream = next(iter(restored.case("copied").master_zone.process_streams))

    assert [segment.name for segment in stream.segments] == ["S1", "S2"]
    assert all(segment.parent is stream for segment in stream.segments)


def test_structured_segment_input_creates_one_prepared_parent():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Variable CP",
                    "segments": [
                        {"t_supply": 200, "t_target": 150, "heat_flow": 50},
                        {"t_supply": 150, "t_target": 100, "heat_flow": 100},
                    ],
                }
            ]
        }
    )

    zone = prepare_problem(streams=inputs.streams, project_name="Site")
    stream = next(iter(zone.process_streams))
    assert len(zone.process_streams) == 1
    assert stream.segment_count == 2
    assert float(stream.heat_flow) == pytest.approx(150.0)


def test_adjacent_flat_rows_remain_independent_physical_streams():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Flat leg 1",
                    "t_supply": 200,
                    "t_target": 150,
                    "heat_flow": 50,
                },
                {
                    "zone": "Site",
                    "name": "Flat leg 2",
                    "t_supply": 150,
                    "t_target": 100,
                    "heat_flow": 100,
                },
            ]
        }
    )

    streams = prepare_problem(
        streams=inputs.streams,
        project_name="Site",
    ).process_streams

    assert len(streams) == 2
    assert all(not stream.has_segments for stream in streams)


def test_profile_input_is_authoritative_and_validates_duplicate_aggregate():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Variable CP",
                    "t_supply": 200,
                    "t_target": 100,
                    "heat_flow": 150,
                    "profile": {
                        "points": [
                            {"temperature": 200, "cumulative_heat": 0},
                            {"temperature": 150, "cumulative_heat": 50},
                            {"temperature": 100, "cumulative_heat": 150},
                        ]
                    },
                }
            ]
        }
    )

    stream = next(
        iter(
            prepare_problem(streams=inputs.streams, project_name="Site").process_streams
        )
    )
    assert stream.segment_count == 2

    inputs.streams[0].heat_flow = 149.0
    with pytest.raises(ValueError, match="authoritative profile"):
        prepare_problem(streams=inputs.streams, project_name="Site")


def test_problem_table_matches_equivalent_flat_segments():
    flat = StreamCollection(
        [
            Stream(name="S1", t_supply=200, t_target=150, heat_flow=50),
            Stream(name="S2", t_supply=150, t_target=100, heat_flow=100),
        ]
    )
    nested = StreamCollection([Stream(name="Parent", segments=_hot_segments())])

    flat_pt = problem_table_algorithm(
        create_problem_table_with_t_int(flat),
        flat.get_hot_streams(),
        flat.get_cold_streams(),
    )
    nested_pt = problem_table_algorithm(
        create_problem_table_with_t_int(nested),
        nested.get_hot_streams(),
        nested.get_cold_streams(),
    )
    assert np.allclose(flat_pt.data, nested_pt.data, equal_nan=True)


def test_multiperiod_profile_alignment_uses_stable_union_fraction_grid():
    first, second = align_temperature_heat_profiles(
        [
            [[0.0, 200.0], [25.0, 175.0], [100.0, 100.0]],
            [[0.0, 210.0], [50.0, 160.0], [80.0, 130.0], [100.0, 110.0]],
        ]
    )

    assert first.shape == second.shape == (5, 2)
    np.testing.assert_allclose(first[:, 0], [0.0, 25.0, 50.0, 80.0, 100.0])
    np.testing.assert_allclose(second[:, 0], [0.0, 25.0, 50.0, 80.0, 100.0])


def test_profile_alignment_rejects_heat_coordinate_reversal():
    with pytest.raises(ValueError, match="strictly monotonic"):
        align_temperature_heat_profiles([[[0.0, 200.0], [50.0, 150.0], [40.0, 100.0]]])


def test_structured_profile_rejects_nonincreasing_cumulative_heat_with_path():
    data = {
        "streams": [
            {
                "zone": "Site",
                "name": "Invalid profile",
                "profile": {
                    "points": [
                        {"temperature": 200, "cumulative_heat": 0},
                        {"temperature": 150, "cumulative_heat": 50},
                        {"temperature": 100, "cumulative_heat": 40},
                    ]
                },
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"streams\[0\]\.profile\.points\[2\]\.cumulative_heat",
    ):
        validate_problem_inputs(data)

    inputs = TargetInput.model_validate(data)
    with pytest.raises(ValueError, match="cumulative heat must increase strictly"):
        prepare_problem(streams=inputs.streams, project_name="Site")


def test_profile_preparation_rejects_raw_temperature_reversal_before_linearising():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Reversing profile",
                    "profile": {
                        "linearisation_tolerance": 1000.0,
                        "points": [
                            {"temperature": 200, "cumulative_heat": 0},
                            {"temperature": 150, "cumulative_heat": 50},
                            {"temperature": 180, "cumulative_heat": 100},
                            {"temperature": 100, "cumulative_heat": 150},
                        ],
                    },
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="preserve one direction"):
        prepare_problem(streams=inputs.streams, project_name="Site")


def test_profile_preparation_applies_minimum_span_to_isothermal_leg():
    data = {
        "streams": [
            {
                "zone": "Site",
                "name": "Near-isothermal profile",
                "profile": {
                    "points": [
                        {"temperature": 200, "cumulative_heat": 0},
                        {"temperature": 150, "cumulative_heat": 50},
                        {"temperature": 150, "cumulative_heat": 100},
                        {"temperature": 100, "cumulative_heat": 150},
                    ]
                },
            }
        ]
    }
    inputs = validate_problem_inputs(data)

    stream = next(
        iter(
            prepare_problem(
                streams=inputs.streams,
                project_name="Site",
            ).process_streams
        )
    )

    assert stream.segment_count >= 2
    assert all(
        abs(float(segment.t_supply) - float(segment.t_target)) >= 0.01 - tol
        for segment in stream.segments
    )


def test_structured_multiperiod_profile_linearises_onto_one_stable_union_grid():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Calculated multiperiod profile",
                    "profile": {
                        "linearisation_tolerance": 0.1,
                        "points": [
                            {
                                "cumulative_heat": {"values": [0, 0]},
                                "temperature": {"values": [200, 200]},
                            },
                            {
                                "cumulative_heat": {"values": [25, 25]},
                                "temperature": {"values": [175, 190]},
                            },
                            {
                                "cumulative_heat": {"values": [50, 50]},
                                "temperature": {"values": [150, 150]},
                            },
                            {
                                "cumulative_heat": {"values": [75, 75]},
                                "temperature": {"values": [125, 110]},
                            },
                            {
                                "cumulative_heat": {"values": [100, 100]},
                                "temperature": {"values": [100, 100]},
                            },
                        ],
                    },
                }
            ]
        }
    )

    stream = next(
        iter(
            prepare_problem(
                streams=inputs.streams,
                project_name="Site",
            ).process_streams
        )
    )

    assert stream.segment_count > 1
    assert all(segment.t_supply.num_periods == 2 for segment in stream.segments)
    assert all(
        np.allclose(left.t_target.period_values, right.t_supply.period_values)
        for left, right in zip(stream.segments[:-1], stream.segments[1:])
    )
