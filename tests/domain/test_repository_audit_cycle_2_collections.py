"""Collection ownership and period-selection audit regressions."""

from copy import copy

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from OpenPinch.analysis.numerics import get_period_index
from OpenPinch.domain._value.resolution import get_period_value
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value


def _stream(name, temperatures=(150.0, 200.0)):
    return Stream(
        name=name,
        supply_temperature=list(temperatures),
        target_temperature=[100.0, 100.0],
        heat_flow=[10.0, 20.0],
    )


@pytest.mark.parametrize("copy_method", [lambda collection: collection.copy(), copy])
@settings(max_examples=12, deadline=None)
@example(count=2)
@given(count=st.integers(min_value=1, max_value=6))
def test_shallow_copy_owns_its_container(copy_method, count):
    original = StreamCollection([_stream(str(i)) for i in range(count)])
    original.numeric_view(idx=0)
    copied = copy_method(original)
    assert copied["0"] is original["0"]
    copied.remove("0")
    copied.add(_stream("new"))
    assert original.get_stream_names() == [str(i) for i in range(count)]
    assert copied.get_stream_names() == [str(i) for i in range(1, count)] + ["new"]
    assert original.numeric_view(idx=0).parent_name.tolist() == [
        str(i) for i in range(count)
    ]


def test_shallow_copy_preserves_callable_sort_order():
    collection = StreamCollection([_stream("a"), _stream("b")])
    collection.set_sort_key(lambda stream: stream.name, reverse=False)
    assert [stream.name for stream in collection.copy()] == ["a", "b"]


@pytest.mark.parametrize("idx", [-1, "-1", 0.5, True, np.bool_(False)])
@pytest.mark.parametrize(
    "consumer",
    ["value", "resolver", "parent", "segments", "report", "sum", "analysis", "config"],
)
def test_invalid_period_indices_do_not_select_another_period(idx, consumer):
    value = Value([10.0, 20.0], "kW")
    collection = StreamCollection([_stream("hot")])
    with pytest.raises((TypeError, ValueError, IndexError)):
        if consumer == "value":
            value[idx]
        elif consumer == "resolver":
            get_period_value(value, period_idx=idx)
        elif consumer == "parent":
            collection.numeric_view(idx=idx)
        elif consumer == "segments":
            collection.segment_numeric_view(idx=idx)
        elif consumer == "sum":
            collection.sum_stream_attribute("heat_flow", idx=idx)
        elif consumer == "analysis":
            get_period_index({"base": 0, "peak": 1}, {"period_idx": idx})
        elif consumer == "config":
            config = Configuration({"PROBLEM_PERIOD_IDS": ["base", "peak"]})
            config.for_period(period_idx=idx)
        else:
            collection.to_dict(idx=idx)


@pytest.mark.parametrize("idx", [1, np.int64(1), "1"])
def test_supported_period_indices_agree(idx):
    value = Value([10.0, 20.0], "kW")
    assert value[idx].value == 20.0
    assert get_period_value(value, period_idx=idx) == 20.0
    collection = StreamCollection([_stream("hot")])
    assert collection.numeric_view(idx=idx).heat_flow.tolist() == [20.0]
    assert collection.to_dict(idx=idx)["heat_flow"] == [20.0]


def test_configuration_rejects_conflicting_period_selectors():
    config = Configuration({"PROBLEM_PERIOD_IDS": ["base", "peak"]})
    with pytest.raises(ValueError, match="period_idx"):
        config.for_period(period_id="base", period_idx=1)
    assert config.for_period(period_id="peak", period_idx=1).period_id == "peak"
