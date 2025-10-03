import pytest

from OpenPinch.classes import *
from OpenPinch.lib import *

"""Tests for StreamCollection."""


@pytest.fixture
def sample_streams():
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
