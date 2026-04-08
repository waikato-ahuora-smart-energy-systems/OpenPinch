"""Regression tests for the stream collection classes."""

import pytest

from OpenPinch.classes import *
from OpenPinch.lib import *

"""Tests for StreamCollection."""

import csv
from uuid import uuid4


@pytest.fixture
def sample_streams():
    """Return sample streams data used by this test module."""
    s1 = Stream(name="A", t_supply=150, t_target=100, heat_flow=1)
    s2 = Stream(name="B", t_supply=180, t_target=120, heat_flow=1)
    s3 = Stream(name="C", t_supply=130, t_target=90, heat_flow=1)
    return s1, s2, s3


def test_add_and_get(sample_streams):
    s1, s2, _ = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    sc.add(s2)

    names = [s.name for s in sc]
    assert set(names) == {"A", "B"}


def test_no_overwrite(sample_streams):
    s1, _, _ = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    # Add another stream with same name
    s1_dup = Stream(name="A", t_supply=100, t_target=50)
    sc.add(s1_dup)

    names = [s.name for s in sc]
    assert len(names) == 2
    assert "A" in names


def test_overwrite_allowed(sample_streams):
    s1, _, _ = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    s1_new = Stream(name="A", t_supply=10, t_target=20)
    sc.add(s1_new, prevent_overwrite=False)

    assert len(list(sc)) == 1
    assert sc._streams["A"].t_supply == 10  # Overwritten


def test_remove(sample_streams):
    s1, s2, _ = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    sc.add(s2)

    sc.remove("A")
    names = [s.name for s in sc]
    assert names == ["B"]


def test_sort_by_supply_temp(sample_streams):
    s1, s2, s3 = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    sc.add(s2)
    sc.add(s3)

    # Sort descending by t_supply
    sc.set_sort_key(lambda s: s.t_supply, reverse=True)
    temps = [s.t_supply for s in sc]
    assert temps == [180, 150, 130]


def test_sort_by_target_then_supply(sample_streams):
    s1, s2, s3 = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    sc.add(s2)
    sc.add(s3)

    # Sort descending by (t_target, t_supply)
    sc.set_sort_key(lambda s: (s.t_target, s.t_supply), reverse=True)
    targets = [s.t_target for s in sc]
    assert targets == [120, 100, 90]


def test_empty_collection():
    sc = StreamCollection()
    assert list(sc) == []


def test_repr(sample_streams):
    s1, _, _ = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    rep = repr(sc)
    assert "StreamCollection" in rep
    assert "A" in rep


def test_dynamic_sort_after_adding(sample_streams):
    s1, s2, s3 = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    sc.add(s2)

    # Set sort by t_supply descending
    sc.set_sort_key(lambda s: s.t_supply, reverse=True)

    # Now add a new stream AFTER setting the sort
    new_stream = Stream(name="D", t_supply=160, t_target=110)
    sc.add(new_stream)

    # Check that new stream appears in the correct position
    names_in_order = [s.name for s in sc]
    temps_in_order = [s.t_supply for s in sc]

    assert names_in_order == ["B", "D", "A"]
    assert temps_in_order == [180, 160, 150]


def test_get_hot_streams_filters_and_preserves_sort_settings():
    sc = StreamCollection()
    sc.add(Stream(name="H1", t_supply=200, t_target=120, heat_flow=1))
    sc.add(Stream(name="C1", t_supply=90, t_target=160, heat_flow=1))
    sc.add(Stream(name="H2", t_supply=180, t_target=100, heat_flow=1))
    sc.set_sort_key("t_supply", reverse=False)

    hot_streams = sc.get_hot_streams()

    assert len(hot_streams) == 2
    assert set(hot_streams._streams.keys()) == {"H1", "H2"}
    assert [s.name for s in hot_streams] == ["H2", "H1"]
    assert hot_streams._sort_reverse is False


def test_get_hot_streams_includes_process_and_utility_when_enabled():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H_PRO",
            t_supply=200,
            t_target=150,
            heat_flow=1,
            is_process_stream=True,
        )
    )
    sc.add(
        Stream(
            name="H_UT",
            t_supply=220,
            t_target=170,
            heat_flow=1,
            is_process_stream=False,
        )
    )
    sc.add(
        Stream(
            name="C_PRO", t_supply=80, t_target=120, heat_flow=1, is_process_stream=True
        )
    )
    sc.add(
        Stream(
            name="C_UT", t_supply=70, t_target=130, heat_flow=1, is_process_stream=False
        )
    )

    hot_streams = sc.get_hot_streams(
        include_process_streams=True, include_utility_streams=True
    )

    assert set(hot_streams._streams.keys()) == {"H_PRO", "H_UT"}


def test_get_hot_streams_can_invert_utility_without_mutating_original_streams():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H_PRO",
            t_supply=180,
            t_target=100,
            heat_flow=1,
            is_process_stream=True,
        )
    )
    sc.add(
        Stream(
            name="C_UT", t_supply=60, t_target=140, heat_flow=1, is_process_stream=False
        )
    )

    original_utility = sc["C_UT"]
    original_supply = original_utility.t_supply
    original_target = original_utility.t_target
    original_type = original_utility.type

    hot_streams = sc.get_hot_streams(
        include_process_streams=True,
        include_utility_streams=True,
        invert_utility=True,
    )

    assert set(hot_streams._streams.keys()) == {"H_PRO", "C_UT"}
    assert hot_streams["C_UT"].type == StreamType.Hot.value
    assert hot_streams["C_UT"].t_supply == original_target
    assert hot_streams["C_UT"].t_target == original_supply

    assert original_utility.type == original_type
    assert original_utility.t_supply == original_supply
    assert original_utility.t_target == original_target


def test_get_cold_streams_filters_and_preserves_sort_settings():
    sc = StreamCollection()
    sc.add(Stream(name="C1", t_supply=80, t_target=140, heat_flow=1))
    sc.add(Stream(name="H1", t_supply=220, t_target=160, heat_flow=1))
    sc.add(Stream(name="C2", t_supply=70, t_target=160, heat_flow=1))
    sc.set_sort_key("t_target", reverse=True)

    cold_streams = sc.get_cold_streams()

    assert len(cold_streams) == 2
    assert set(cold_streams._streams.keys()) == {"C1", "C2"}
    assert [s.name for s in cold_streams] == ["C2", "C1"]
    assert cold_streams._sort_reverse is True


def test_get_cold_streams_includes_process_and_utility_when_enabled():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="C_PRO", t_supply=80, t_target=140, heat_flow=1, is_process_stream=True
        )
    )
    sc.add(
        Stream(
            name="C_UT", t_supply=90, t_target=150, heat_flow=1, is_process_stream=False
        )
    )
    sc.add(
        Stream(
            name="H_PRO",
            t_supply=210,
            t_target=120,
            heat_flow=1,
            is_process_stream=True,
        )
    )
    sc.add(
        Stream(
            name="H_UT",
            t_supply=220,
            t_target=130,
            heat_flow=1,
            is_process_stream=False,
        )
    )

    cold_streams = sc.get_cold_streams(
        include_process_streams=True, include_utility_streams=True
    )

    assert set(cold_streams._streams.keys()) == {"C_PRO", "C_UT"}


def test_get_cold_streams_can_invert_utility_without_mutating_original_streams():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="C_PRO", t_supply=90, t_target=160, heat_flow=1, is_process_stream=True
        )
    )
    sc.add(
        Stream(
            name="H_UT",
            t_supply=220,
            t_target=130,
            heat_flow=1,
            is_process_stream=False,
        )
    )

    original_utility = sc["H_UT"]
    original_supply = original_utility.t_supply
    original_target = original_utility.t_target
    original_type = original_utility.type

    cold_streams = sc.get_cold_streams(
        include_process_streams=True,
        include_utility_streams=True,
        invert_utility=True,
    )

    assert set(cold_streams._streams.keys()) == {"C_PRO", "H_UT"}
    assert cold_streams["H_UT"].type == StreamType.Cold.value
    assert cold_streams["H_UT"].t_supply == original_target
    assert cold_streams["H_UT"].t_target == original_supply

    assert original_utility.type == original_type
    assert original_utility.t_supply == original_supply
    assert original_utility.t_target == original_target


def test_export_to_csv(sample_streams):
    sc = StreamCollection()
    for stream in sample_streams:
        sc.add(stream)

    filename = f"test_streams_{uuid4().hex}.csv"
    output_path = sc.export_to_csv(filename)

    try:
        assert output_path.exists()
        with output_path.open(newline="", encoding="utf-8") as csvfile:
            rows = list(csv.DictReader(csvfile))

        assert rows
        assert rows[0].keys() == {
            "name",
            "t_supply",
            "t_target",
            "heat_flow",
            "dt_cont",
            "htc",
        }
        assert {row["name"] for row in rows} == {s.name for s in sample_streams}
    finally:
        output_path.unlink(missing_ok=True)


# ===== Merged from test_stream_collection_extra.py =====
"""Additional branch coverage tests for StreamCollection."""

from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection


def test_stream_collection_get_index_and_getitem_by_name():
    collection = StreamCollection()
    s1 = Stream(name="S1", t_supply=120.0, t_target=80.0, heat_flow=10.0, htc=1.0)
    s2 = Stream(name="S2", t_supply=130.0, t_target=90.0, heat_flow=20.0, htc=1.0)
    collection.add(s1)
    collection.add(s2)

    idx = collection.get_index(s1)
    assert idx in (0, 1)
    assert collection["S2"] is s2
