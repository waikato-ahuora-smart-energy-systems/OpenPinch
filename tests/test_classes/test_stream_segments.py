from __future__ import annotations

import pickle
from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import OpenPinch
from OpenPinch import PinchWorkspace
from OpenPinch.classes import Stream
from OpenPinch.classes._pinch_problem.input.validation import validate_problem_inputs
from OpenPinch.classes._stream.segment import StreamSegment
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import tol
from OpenPinch.lib.schemas.io import TargetInput
from OpenPinch.services.common.problem_table_analysis import (
    create_problem_table_with_t_int,
    problem_table_algorithm,
)
from OpenPinch.services.input_data_processing.data_preparation import prepare_problem
from OpenPinch.utils.stream_linearisation import align_temperature_heat_profiles
from tests.strategies.stream_segments import segmented_streams


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


def test_stream_owned_values_reject_in_place_mutation_and_explicit_update_commits():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    original_revision = stream._numeric_revision

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.segments[0].t_target.value = 140.0
    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.segments[0].heat_flow[0] = 60.0
    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.heat_flow.unit = "MW"

    assert float(stream.segments[0].t_target) == pytest.approx(150.0)
    assert float(stream.heat_flow) == pytest.approx(150.0)
    assert stream._numeric_revision == original_revision

    stream.update_segment(0, heat_flow=60.0)

    assert float(stream.segments[0].heat_flow) == pytest.approx(60.0)
    assert float(stream.heat_flow) == pytest.approx(160.0)
    assert stream._numeric_revision > original_revision


def test_sparse_batch_segment_mutation_commits_once_and_preserves_order(monkeypatch):
    stream = Stream(name="Variable CP", segments=_hot_segments())
    original_names = [segment.name for segment in stream.segments]
    replace_calls = 0
    original_replace = stream.replace_segments

    def counted_replace(segments):
        nonlocal replace_calls
        replace_calls += 1
        original_replace(segments)

    monkeypatch.setattr(stream, "replace_segments", counted_replace)

    stream.update_segments(
        {
            0: {"heat_flow": 60.0, "htc": 2.0},
            1: {"heat_flow": 120.0},
        }
    )

    assert replace_calls == 1
    assert [segment.name for segment in stream.segments] == original_names
    assert [float(segment.heat_flow) for segment in stream.segments] == pytest.approx(
        [60.0, 120.0]
    )
    assert float(stream.heat_flow) == pytest.approx(180.0)


def test_batch_segment_mutation_validates_every_update_before_commit():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    original_segments = stream.segments
    original_revision = stream._numeric_revision

    with pytest.raises(ValueError, match="controlled by parent"):
        stream.update_segments({0: {"heat_flow": 60.0}, 1: {"active": False}})

    assert stream.segments == original_segments
    assert stream._numeric_revision == original_revision
    assert [float(segment.heat_flow) for segment in stream.segments] == pytest.approx(
        [50.0, 100.0]
    )


def test_batch_segment_mutation_rejects_bad_indexes_and_empty_mapping_is_noop():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    original_segments = stream.segments
    original_revision = stream._numeric_revision

    stream.update_segments({})
    assert stream.segments is original_segments
    assert stream._numeric_revision == original_revision

    with pytest.raises(IndexError, match="out of range"):
        stream.update_segments({2: {"heat_flow": 10.0}})
    with pytest.raises(TypeError, match="indexes must be integers"):
        stream.update_segments({"0": {"heat_flow": 10.0}})

    assert stream.segments is original_segments
    assert stream._numeric_revision == original_revision


def test_segmented_parent_price_is_duty_weighted_and_cost_conserves_children():
    stream = Stream(
        name="Priced utility",
        is_process_stream=False,
        segments=[
            StreamSegment(
                t_supply=200.0,
                t_target=150.0,
                heat_flow=50.0,
                price=20.0,
                is_process_stream=False,
            ),
            StreamSegment(
                t_supply=150.0,
                t_target=100.0,
                heat_flow=100.0,
                price=80.0,
                is_process_stream=False,
            ),
        ],
    )

    assert float(stream.price) == pytest.approx(60.0)
    assert float(stream.ut_cost) == pytest.approx(9.0)
    assert float(stream.ut_cost) == pytest.approx(
        sum(float(segment.ut_cost) for segment in stream.segments)
    )


def test_segmented_parent_explicit_price_broadcasts_but_child_can_diverge():
    stream = Stream(
        name="Priced utility",
        price=40.0,
        is_process_stream=False,
        segments=[
            StreamSegment(
                t_supply=200.0,
                t_target=150.0,
                heat_flow=50.0,
                price=20.0,
                is_process_stream=False,
            ),
            StreamSegment(
                t_supply=150.0,
                t_target=100.0,
                heat_flow=100.0,
                price=80.0,
                is_process_stream=False,
            ),
        ],
    )

    assert [float(segment.price) for segment in stream.segments] == pytest.approx(
        [40.0, 40.0]
    )
    stream.update_segment(0, price=10.0)
    assert [float(segment.price) for segment in stream.segments] == pytest.approx(
        [10.0, 40.0]
    )
    assert float(stream.price) == pytest.approx(30.0)

    stream.price = 25.0
    assert [float(segment.price) for segment in stream.segments] == pytest.approx(
        [25.0, 25.0]
    )


def test_segmented_parent_multiperiod_price_and_cost_are_derived_per_period():
    stream = Stream(
        name="Multiperiod priced utility",
        is_process_stream=False,
        segments=[
            StreamSegment(
                t_supply=[200.0, 210.0],
                t_target=[150.0, 160.0],
                heat_flow=[50.0, 100.0],
                price=[20.0, 40.0],
                is_process_stream=False,
            ),
            StreamSegment(
                t_supply=[150.0, 160.0],
                t_target=[100.0, 110.0],
                heat_flow=[100.0, 100.0],
                price=[80.0, 20.0],
                is_process_stream=False,
            ),
        ],
    )

    np.testing.assert_allclose(stream.price.period_values, [60.0, 30.0])
    np.testing.assert_allclose(stream.ut_cost.period_values, [9.0, 6.0])
    stream.set_value_attr_at_idx("price", 50.0, idx=1)
    np.testing.assert_allclose(stream.price.period_values, [60.0, 50.0])
    for segment in stream.segments:
        np.testing.assert_allclose(
            segment.price.period_values[:,],
            [float(segment.price.period_values[0]), 50.0],
        )


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


def test_segmented_parent_dt_cont_assignment_propagates_and_invalidates_view():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    stream.dt_cont_multiplier = 2.0
    collection = StreamCollection([stream])
    before = collection.segment_numeric_view()

    stream.dt_cont = 7.5

    after = collection.segment_numeric_view()
    assert after is not before
    assert float(stream.dt_cont) == pytest.approx(7.5)
    assert float(stream.dt_cont_act) == pytest.approx(15.0)
    assert [float(segment.dt_cont) for segment in stream.segments] == pytest.approx(
        [7.5, 7.5]
    )
    assert [float(segment.dt_cont_act) for segment in stream.segments] == pytest.approx(
        [15.0, 15.0]
    )
    np.testing.assert_allclose(after.dt_cont, [7.5, 7.5])
    np.testing.assert_allclose(
        after.t_min_star,
        [float(segment.t_min_star) for segment in stream.segments],
    )


def test_segmented_parent_indexed_dt_cont_assignment_propagates_one_period():
    stream = Stream(
        name="Multiperiod variable CP",
        segments=[
            StreamSegment(
                t_supply=[200.0, 210.0],
                t_target=[150.0, 160.0],
                heat_flow=[50.0, 60.0],
                dt_cont=[1.0, 2.0],
            ),
            StreamSegment(
                t_supply=[150.0, 160.0],
                t_target=[100.0, 110.0],
                heat_flow=[100.0, 120.0],
                dt_cont=[3.0, 4.0],
            ),
        ],
    )

    stream.set_value_attr_at_idx("dt_cont", 9.0, idx=1)

    np.testing.assert_allclose(stream.dt_cont.period_values, [1.0, 9.0])
    np.testing.assert_allclose(stream.segments[0].dt_cont.period_values, [1.0, 9.0])
    np.testing.assert_allclose(stream.segments[1].dt_cont.period_values, [3.0, 9.0])
    np.testing.assert_allclose(
        StreamCollection([stream]).segment_numeric_view().dt_cont,
        [1.0, 3.0],
    )
    np.testing.assert_allclose(
        StreamCollection([stream]).segment_numeric_view(1).dt_cont,
        [9.0, 9.0],
    )


def test_segmented_parent_dt_cont_assignment_rolls_back_before_commit(monkeypatch):
    stream = Stream(name="Variable CP", segments=_hot_segments())
    collection = StreamCollection([stream])
    before_view = collection.segment_numeric_view()
    before_segments = stream.segments
    before_parent_revision = stream._numeric_revision
    before_segment_revisions = [
        segment._numeric_revision for segment in stream.segments
    ]

    def reject_candidates(_segments):
        raise ValueError("candidate validation failed")

    monkeypatch.setattr(stream, "_validate_segments", reject_candidates)

    with pytest.raises(ValueError, match="candidate validation failed"):
        stream.dt_cont = 8.0

    assert stream.segments == before_segments
    assert all(
        actual is expected
        for actual, expected in zip(stream.segments, before_segments, strict=True)
    )
    assert float(stream.dt_cont) == pytest.approx(0.0)
    assert [float(segment.dt_cont) for segment in stream.segments] == pytest.approx(
        [0.0, 0.0]
    )
    assert stream._numeric_revision == before_parent_revision
    assert [segment._numeric_revision for segment in stream.segments] == (
        before_segment_revisions
    )
    assert collection.segment_numeric_view() is before_view


def test_flat_stream_dt_cont_mutation_contract_is_unchanged():
    stream = Stream(
        name="Flat",
        t_supply=[200.0, 210.0],
        t_target=[100.0, 110.0],
        heat_flow=[100.0, 120.0],
        dt_cont=[1.0, 2.0],
    )

    stream.set_value_attr_at_idx("dt_cont", 4.0, idx=1)
    np.testing.assert_allclose(stream.dt_cont.period_values, [1.0, 4.0])

    stream.dt_cont = 6.0
    np.testing.assert_allclose(stream.dt_cont.period_values, [6.0, 6.0])


@given(
    segmented_streams(),
    st.floats(
        min_value=0.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=30)
def test_segmented_parent_dt_cont_assignment_invariant(stream, contribution):
    collection = StreamCollection([stream])

    stream.dt_cont = contribution

    assert float(stream.dt_cont) == pytest.approx(contribution)
    assert all(
        float(segment.dt_cont) == pytest.approx(contribution)
        for segment in stream.segments
    )
    np.testing.assert_allclose(
        collection.segment_numeric_view().dt_cont,
        np.full(stream.segment_count, contribution),
    )


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


def test_segmented_period_context_can_be_cleared_consistently():
    stream = Stream(name="Variable CP", segments=_hot_segments())
    stream.set_period_context(
        period_ids={"winter": 0, "summer": 1},
        weights=[0.4, 0.6],
        num_periods=2,
    )

    stream.set_period_context(None, None, None)

    assert stream.period_ids is None
    assert stream.weights is None
    assert stream.num_periods is None
    assert all(segment.period_ids is None for segment in stream.segments)
    assert all(segment.weights is None for segment in stream.segments)


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


def test_copy_and_pickle_preserve_independent_segment_prices_and_parent_cost():
    original = Stream(
        name="Priced utility",
        is_process_stream=False,
        segments=[
            StreamSegment(
                t_supply=200.0,
                t_target=150.0,
                heat_flow=50.0,
                price=20.0,
                is_process_stream=False,
            ),
            StreamSegment(
                t_supply=150.0,
                t_target=100.0,
                heat_flow=100.0,
                price=80.0,
                is_process_stream=False,
            ),
        ],
    )

    for restored in (deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert [float(segment.price) for segment in restored.segments] == pytest.approx(
            [20.0, 80.0]
        )
        assert float(restored.price) == pytest.approx(60.0)
        assert float(restored.ut_cost) == pytest.approx(9.0)


@given(
    segmented_streams(),
    st.floats(
        min_value=0.0,
        max_value=500.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=30)
def test_segment_price_and_parent_cost_conservation_invariant(stream, base_price):
    stream.update_segments(
        {index: {"price": base_price + index} for index in range(stream.segment_count)}
    )

    duties = np.asarray([float(segment.heat_flow) for segment in stream.segments])
    prices = np.asarray([float(segment.price) for segment in stream.segments])
    expected_price = float(np.sum(duties * prices) / np.sum(duties))
    expected_cost = float(np.sum(duties * prices) / 1000.0)
    assert float(stream.price) == pytest.approx(expected_price)
    assert float(stream.ut_cost) == pytest.approx(expected_cost)
    assert float(stream.ut_cost) == pytest.approx(
        sum(float(segment.ut_cost) for segment in stream.segments)
    )


def test_stream_segment_is_private_and_preserves_internal_model_name():
    assert not hasattr(OpenPinch, "StreamSegment")
    assert not hasattr(OpenPinch.classes, "StreamSegment")
    assert not hasattr(
        __import__("OpenPinch.classes.stream", fromlist=["*"]), "StreamSegment"
    )
    assert Stream.__module__ == "OpenPinch.classes.stream"
    assert StreamSegment.__module__ == "OpenPinch.classes._stream.segment"


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


def test_structured_segmented_utility_preserves_child_price_overrides():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Hot process",
                    "t_supply": 200,
                    "t_target": 100,
                    "heat_flow": 100,
                },
                {
                    "zone": "Site",
                    "name": "Cold process",
                    "t_supply": 50,
                    "t_target": 150,
                    "heat_flow": 100,
                },
            ],
            "utilities": [
                {
                    "name": "Segmented steam",
                    "type": "Hot",
                    "price": 40,
                    "segments": [
                        {
                            "t_supply": 250,
                            "t_target": 220,
                            "heat_flow": 50,
                            "price": 20,
                        },
                        {
                            "t_supply": 220,
                            "t_target": 180,
                            "heat_flow": 100,
                        },
                    ],
                }
            ],
        }
    )

    zone = prepare_problem(
        streams=inputs.streams,
        utilities=inputs.utilities,
        project_name="Site",
    )
    utility = next(
        stream for stream in zone.hot_utilities if stream.name == "Segmented steam"
    )

    assert utility.segment_count == 2
    assert utility.is_process_stream is False
    assert [float(segment.price) for segment in utility.segments] == pytest.approx(
        [20.0, 40.0]
    )
    assert float(utility.price) == pytest.approx(100.0 / 3.0)
    assert float(utility.ut_cost) == pytest.approx(5.0)


def test_structured_utility_profile_uses_one_parent_price_for_all_segments():
    inputs = TargetInput.model_validate(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Hot process",
                    "t_supply": 200,
                    "t_target": 100,
                    "heat_flow": 100,
                },
                {
                    "zone": "Site",
                    "name": "Cold process",
                    "t_supply": 50,
                    "t_target": 150,
                    "heat_flow": 100,
                },
            ],
            "utilities": [
                {
                    "name": "Profile steam",
                    "type": "Hot",
                    "price": 30,
                    "profile": {
                        "points": [
                            {"temperature": 250, "cumulative_heat": 0},
                            {"temperature": 220, "cumulative_heat": 50},
                            {"temperature": 180, "cumulative_heat": 150},
                        ]
                    },
                }
            ],
        }
    )

    zone = prepare_problem(
        streams=inputs.streams,
        utilities=inputs.utilities,
        project_name="Site",
    )
    utility = next(
        stream for stream in zone.hot_utilities if stream.name == "Profile steam"
    )

    assert utility.has_segments
    assert all(
        float(segment.price) == pytest.approx(30.0) for segment in utility.segments
    )
    assert float(utility.price) == pytest.approx(30.0)


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
