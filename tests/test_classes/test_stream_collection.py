"""Regression tests for the stream collection classes."""

import csv
from uuid import uuid4

import numpy as np
import pytest

from OpenPinch.classes import *
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib import *

"""Tests for StreamCollection."""


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


def test_copy_preserves_keys_and_can_deep_copy_streams():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="Steam",
            t_supply=220,
            t_target=180,
            heat_flow=1,
            is_process_stream=False,
        ),
        key="Hot Utility.Steam",
    )

    copied = sc.copy(deep=True)

    assert list(copied._streams.keys()) == ["Hot Utility.Steam"]
    assert copied["Hot Utility.Steam"] is not sc["Hot Utility.Steam"]
    assert copied["Hot Utility.Steam"].t_supply == sc["Hot Utility.Steam"].t_supply


def test_copy_without_deep_shares_stream_references():
    sc = StreamCollection()
    stream = Stream(name="H1", t_supply=200, t_target=120, heat_flow=1)
    sc.add(stream, key="AreaA.H1")

    copied = sc.copy()

    assert copied["AreaA.H1"] is stream


def test_copy_can_override_sort_key(sample_streams):
    s1, s2, s3 = sample_streams
    sc = StreamCollection()
    sc.add(s1)
    sc.add(s2)
    sc.add(s3)
    sc.set_sort_key("t_supply", reverse=True)

    copied = sc.copy(sort_key="t_target", reverse=False)

    assert [s.name for s in copied] == ["C", "A", "B"]
    assert copied._sort_reverse is False


def test_reset_heat_flows_zeros_all_streams_in_place():
    sc = StreamCollection()
    sc.add(Stream(name="H1", t_supply=200, t_target=120, heat_flow=15.0))
    sc.add(Stream(name="C1", t_supply=60, t_target=140, heat_flow=25.0))

    returned = sc.set_common_stream_attribute("heat_flow", 0.0)

    assert returned is sc
    assert all(stream.heat_flow == 0.0 for stream in sc)


def test_sum_heat_flow_returns_total_for_all_streams():
    sc = StreamCollection()
    sc.add(Stream(name="H1", t_supply=200, t_target=120, heat_flow=15.0))
    sc.add(Stream(name="C1", t_supply=60, t_target=140, heat_flow=25.0))

    assert sc.sum_stream_attribute("heat_flow") == pytest.approx(40.0)


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

    assert set(hot_streams._streams.keys()) == {"C_UT"}
    assert hot_streams["C_UT"].type == ST.Hot.value
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

    assert set(cold_streams._streams.keys()) == {"H_UT"}
    assert cold_streams["H_UT"].type == ST.Cold.value
    assert cold_streams["H_UT"].t_supply == original_target
    assert cold_streams["H_UT"].t_target == original_supply

    assert original_utility.type == original_type
    assert original_utility.t_supply == original_supply
    assert original_utility.t_target == original_target


def test_get_process_streams_returns_hot_and_cold_process_streams():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H_PRO",
            t_supply=200,
            t_target=150,
            heat_flow=1,
            is_process_stream=True,
        ),
        key="AreaA.H_PRO",
    )
    sc.add(
        Stream(
            name="C_PRO",
            t_supply=80,
            t_target=140,
            heat_flow=1,
            is_process_stream=True,
        ),
        key="AreaA.C_PRO",
    )
    sc.add(
        Stream(
            name="H_UT",
            t_supply=220,
            t_target=170,
            heat_flow=1,
            is_process_stream=False,
        ),
        key="Hot Utility.H_UT",
    )
    sc.set_sort_key("t_supply", reverse=False)

    process_streams = sc.get_process_streams()

    assert set(process_streams._streams.keys()) == {"AreaA.H_PRO", "AreaA.C_PRO"}
    assert [s.name for s in process_streams] == ["C_PRO", "H_PRO"]
    assert process_streams._sort_reverse is False


def test_get_utility_streams_returns_hot_and_cold_utility_streams():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H_PRO",
            t_supply=200,
            t_target=150,
            heat_flow=1,
            is_process_stream=True,
        ),
        key="AreaA.H_PRO",
    )
    sc.add(
        Stream(
            name="H_UT",
            t_supply=220,
            t_target=170,
            heat_flow=1,
            is_process_stream=False,
        ),
        key="Hot Utility.H_UT",
    )
    sc.add(
        Stream(
            name="C_UT",
            t_supply=90,
            t_target=150,
            heat_flow=1,
            is_process_stream=False,
        ),
        key="Cold Utility.C_UT",
    )
    sc.set_sort_key("t_target", reverse=True)

    utility_streams = sc.get_utility_streams()

    assert set(utility_streams._streams.keys()) == {
        "Hot Utility.H_UT",
        "Cold Utility.C_UT",
    }
    assert [s.name for s in utility_streams] == ["H_UT", "C_UT"]
    assert utility_streams._sort_reverse is True


def test_validate_state_alignment_returns_none_for_all_scalar_streams():
    sc = StreamCollection()
    sc.add(Stream(name="H1", t_supply=200, t_target=120, heat_flow=1))
    sc.add(Stream(name="C1", t_supply=60, t_target=140, heat_flow=1))

    state_ids, weights = sc.validate_state_alignment()

    assert state_ids is None
    assert weights is None
    assert sc.state_ids is None
    assert sc.weights is None


def test_validate_state_alignment_uses_first_stateful_stream_as_canonical():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H1",
            t_supply={
                "values": [200.0, 180.0],
                "state_ids": ["0", "1"],
                "unit": "degC",
            },
            t_target={
                "values": [120.0, 100.0],
                "state_ids": ["0", "1"],
                "unit": "degC",
            },
            heat_flow={"values": [100.0, 80.0], "state_ids": ["0", "1"], "unit": "kW"},
            htc=1.0,
        ),
        key="AreaA.H1",
    )
    sc.add(
        Stream(
            name="HU",
            t_supply=260.0,
            t_target=220.0,
            heat_flow=0.0,
            is_process_stream=False,
        )
    )

    state_ids, weights = sc.validate_state_alignment()

    assert state_ids == ["0", "1"]
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))
    assert sc.state_ids == ["0", "1"]
    np.testing.assert_allclose(sc.weights, np.array([0.5, 0.5]))


def test_state_context_is_preserved_on_copy_and_subset():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H1",
            t_supply={
                "values": [200.0, 180.0],
                "state_ids": ["0", "1"],
                "weights": [0.25, 0.75],
                "unit": "degC",
            },
            t_target={
                "values": [120.0, 100.0],
                "state_ids": ["0", "1"],
                "weights": [0.25, 0.75],
                "unit": "degC",
            },
            heat_flow={
                "values": [100.0, 80.0],
                "state_ids": ["0", "1"],
                "weights": [0.25, 0.75],
                "unit": "kW",
            },
            htc=1.0,
        ),
    )
    sc.add(
        Stream(
            name="HU",
            t_supply=260.0,
            t_target=220.0,
            heat_flow=0.0,
            is_process_stream=False,
        )
    )
    sc.validate_state_alignment()

    copied = sc.copy()
    hot_streams = sc.get_hot_streams()

    assert copied.state_ids == ["0", "1"]
    np.testing.assert_allclose(copied.weights, np.array([0.25, 0.75]))
    assert hot_streams.state_ids == ["0", "1"]
    np.testing.assert_allclose(hot_streams.weights, np.array([0.25, 0.75]))


def test_validate_state_alignment_rejects_mismatched_state_ids():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H1",
            t_supply={
                "values": [200.0, 180.0],
                "state_ids": ["0", "1"],
                "unit": "degC",
            },
            t_target={
                "values": [120.0, 100.0],
                "state_ids": ["0", "1"],
                "unit": "degC",
            },
            heat_flow={"values": [100.0, 80.0], "state_ids": ["0", "1"], "unit": "kW"},
            htc=1.0,
        ),
        key="AreaA.H1",
    )
    sc.add(
        Stream(
            name="C1",
            t_supply={
                "values": [60.0, 80.0],
                "state_ids": ["0", "peak"],
                "unit": "degC",
            },
            t_target={
                "values": [140.0, 160.0],
                "state_ids": ["0", "peak"],
                "unit": "degC",
            },
            heat_flow={
                "values": [90.0, 110.0],
                "state_ids": ["0", "peak"],
                "unit": "kW",
            },
            htc=1.0,
        ),
        key="AreaB.C1",
    )

    with pytest.raises(
        ValueError,
        match="state_ids for stream 'AreaB.C1' must align with 'AreaA.H1'",
    ):
        sc.validate_state_alignment()


def test_validate_state_alignment_rejects_mismatched_weights():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H1",
            t_supply={
                "values": [200.0, 180.0],
                "state_ids": ["0", "1"],
                "weights": [0.5, 0.5],
                "unit": "degC",
            },
            t_target={
                "values": [120.0, 100.0],
                "state_ids": ["0", "1"],
                "weights": [0.5, 0.5],
                "unit": "degC",
            },
            heat_flow={
                "values": [100.0, 80.0],
                "state_ids": ["0", "1"],
                "weights": [0.5, 0.5],
                "unit": "kW",
            },
            htc=1.0,
        ),
        key="AreaA.H1",
    )
    sc.add(
        Stream(
            name="C1",
            t_supply={
                "values": [60.0, 80.0],
                "state_ids": ["0", "1"],
                "weights": [0.25, 0.75],
                "unit": "degC",
            },
            t_target={
                "values": [140.0, 160.0],
                "state_ids": ["0", "1"],
                "weights": [0.25, 0.75],
                "unit": "degC",
            },
            heat_flow={
                "values": [90.0, 110.0],
                "state_ids": ["0", "1"],
                "weights": [0.25, 0.75],
                "unit": "kW",
            },
            htc=1.0,
        ),
        key="AreaB.C1",
    )

    with pytest.raises(
        ValueError,
        match="weights for stream 'AreaB.C1' must align with 'AreaA.H1'",
    ):
        sc.validate_state_alignment()


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
            "dt_cont_act",
            "htc",
        }
        assert {row["name"] for row in rows} == {s.name for s in sample_streams}
    finally:
        output_path.unlink(missing_ok=True)


# ===== Merged from test_stream_collection_extra.py =====
"""Additional branch coverage tests for StreamCollection."""


def test_stream_collection_get_index_and_getitem_by_name():
    collection = StreamCollection()
    s1 = Stream(name="S1", t_supply=120.0, t_target=80.0, heat_flow=10.0, htc=1.0)
    s2 = Stream(name="S2", t_supply=130.0, t_target=90.0, heat_flow=20.0, htc=1.0)
    collection.add(s1)
    collection.add(s2)

    idx = collection.get_index(s1)
    assert idx in (0, 1)
    assert collection["S2"] is s2
