"""Sort freshness and deep-copy regressions from the third audit pass."""

from copy import deepcopy

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _collection():
    return StreamCollection(
        [
            Stream(
                name="A",
                supply_temperature=150.0,
                target_temperature=100.0,
                heat_flow=10.0,
            ),
            Stream(
                name="B",
                supply_temperature=200.0,
                target_temperature=100.0,
                heat_flow=10.0,
            ),
        ]
    )


@settings(max_examples=20, deadline=None)
@example(temperature=300.0)
@given(
    temperature=st.floats(
        min_value=201.0, max_value=350.0, allow_nan=False, allow_infinity=False
    )
)
def test_order_refreshes_after_numeric_mutation(temperature):
    collection = _collection()
    assert [stream.name for stream in collection] == ["B", "A"]
    collection["A"].supply_temperature = temperature
    assert [stream.name for stream in collection] == ["A", "B"]
    assert collection.get_index(collection["A"]) == 0
    assert collection[0] is collection["A"]


@pytest.mark.parametrize("key", ["name", lambda stream: stream.name])
def test_order_refreshes_after_metadata_mutation(key):
    collection = _collection()
    collection.set_sort_key(key, reverse=False)
    assert [stream.name for stream in collection] == ["A", "B"]
    collection["A"].name = "Z"
    assert [stream.name for stream in collection] == ["B", "Z"]


def test_callable_sort_observes_external_state_changes():
    collection = _collection()
    priority = {"A": 0, "B": 1}
    collection.set_sort_key(lambda stream: priority[stream.name])
    assert [stream.name for stream in collection] == ["A", "B"]
    priority["A"] = 2
    assert [stream.name for stream in collection] == ["B", "A"]


@pytest.mark.parametrize(
    "copy_method", [lambda collection: collection.copy(deep=True), deepcopy]
)
def test_deep_copy_preserves_callable_sort_and_isolates_streams(copy_method):
    collection = _collection()
    collection.set_sort_key(lambda stream: stream.name, reverse=False)
    collection["A"].supply_temperature = 250.0
    copied = copy_method(collection)
    assert [stream.name for stream in copied] == ["A", "B"]
    copied["A"].supply_temperature = 300.0
    assert collection["A"].supply_temperature.value == 250.0
    copied.remove("B")
    assert "B" in collection
