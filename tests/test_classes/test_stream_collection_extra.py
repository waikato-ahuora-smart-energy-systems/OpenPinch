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
