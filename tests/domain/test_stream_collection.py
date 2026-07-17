"""Regression tests for the stream collection classes."""

import csv
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from OpenPinch.domain._stream_collection.numeric_view import value_at_idx
from OpenPinch.domain.enums import ST
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.presentation.reporting.stream_collection import (
    export_stream_collection,
)

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


def test_numeric_view_updates_after_stream_mutation():
    stream = Stream(name="H1", t_supply=200, t_target=120, heat_flow=80.0)
    sc = StreamCollection([stream])

    initial = sc.numeric_view()
    assert initial.cp[0] == pytest.approx(1.0)

    stream.heat_flow = 160.0
    updated = sc.numeric_view()

    assert updated.cp[0] == pytest.approx(2.0)
    assert updated is not initial


def test_numeric_view_updates_after_collection_mutation(sample_streams):
    s1, s2, _ = sample_streams
    sc = StreamCollection([s1])
    assert sc.numeric_view().cp.shape == (1,)

    sc.add(s2)

    assert sc.numeric_view().cp.shape == (2,)


def test_numeric_view_none_values_and_legacy_pickle_state_rebuild_cache():
    assert np.isnan(value_at_idx(None))

    stream = Stream(name="H1", t_supply=200, t_target=120, heat_flow=80.0)
    sc = StreamCollection([stream])
    state = sc.__getstate__()
    state.pop("_numeric_cache", None)

    restored = StreamCollection()
    del restored.__dict__["_numeric_cache"]
    restored.__setstate__(state)

    assert restored._numeric_cache == {}
    assert restored[0].name == "H1"


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


def test_to_dict_exports_streams_in_standard_reporting_order():
    sc = StreamCollection()
    streams = [
        Stream(
            name="CU1",
            t_supply=50,
            t_target=100,
            heat_flow=1,
            is_process_stream=False,
        ),
        Stream(name="C2", t_supply=80, t_target=180, heat_flow=1),
        Stream(name="H1", t_supply=200, t_target=100, heat_flow=1),
        Stream(
            name="HU1",
            t_supply=240,
            t_target=230,
            heat_flow=1,
            is_process_stream=False,
        ),
        Stream(name="H2", t_supply=220, t_target=90, heat_flow=1),
        Stream(
            name="CU2",
            t_supply=90,
            t_target=140,
            heat_flow=1,
            is_process_stream=False,
        ),
        Stream(name="C1", t_supply=120, t_target=150, heat_flow=1),
        Stream(
            name="HU2",
            t_supply=260,
            t_target=200,
            heat_flow=1,
            is_process_stream=False,
        ),
        Stream(name="H3", t_supply=200, t_target=150, heat_flow=1),
    ]
    for stream in streams:
        sc.add(stream)
    sc.set_sort_key("name", reverse=False)

    payload = sc.to_dict()
    frame = pd.DataFrame(payload)

    assert frame["name"].tolist() == [
        "H2",
        "H3",
        "H1",
        "C1",
        "C2",
        "HU2",
        "HU1",
        "CU2",
        "CU1",
    ]
    assert frame["category"].tolist() == [
        "hot_stream",
        "hot_stream",
        "hot_stream",
        "cold_stream",
        "cold_stream",
        "hot_utility",
        "hot_utility",
        "cold_utility",
        "cold_utility",
    ]
    assert frame.loc[0, "t_supply"] == pytest.approx(220.0)
    assert frame.loc[0, "t_target"] == pytest.approx(90.0)


def test_to_dict_uses_requested_period_index_for_period_stream_values():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="H1",
            t_supply={"values": [200.0, 180.0], "unit": "degC"},
            t_target={"values": [120.0, 100.0], "unit": "degC"},
            heat_flow={"values": [100.0, 80.0], "unit": "kW"},
        )
    )

    payload = sc.to_dict(idx=1)
    frame = pd.DataFrame(payload)

    assert frame.loc[0, "t_supply"] == pytest.approx(180.0)
    assert frame.loc[0, "t_target"] == pytest.approx(100.0)
    assert frame.loc[0, "heat_flow"] == pytest.approx(80.0)


def test_validate_period_alignment_uses_first_period_stream_as_canonical():
    sc = StreamCollection()
    sc.set_period_context(
        period_ids={"0": 0, "1": 1},
        weights=[0.25, 0.75],
        num_periods=2,
    )
    sc.add(
        Stream(
            name="H1",
            t_supply={
                "values": [200.0, 180.0],
                "unit": "degC",
            },
            t_target={
                "values": [120.0, 100.0],
                "unit": "degC",
            },
            heat_flow={"values": [100.0, 80.0], "period_ids": ["0", "1"], "unit": "kW"},
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

    assert list(sc[0].period_ids.keys()) == ["0", "1"]
    np.testing.assert_allclose(sc[0].weights, np.array([0.25, 0.75]))
    assert sc.period_ids == {"0": 0, "1": 1}
    np.testing.assert_allclose(sc.weights, np.array([0.25, 0.75]))


def test_state_context_is_preserved_on_copy_and_subset():
    sc = StreamCollection()
    sc.set_period_context(
        period_ids={"0": 0, "1": 1},
        weights=[0.25, 0.75],
        num_periods=2,
    )
    sc.add(
        Stream(
            name="H1",
            t_supply={
                "values": [200.0, 180.0],
                "unit": "degC",
            },
            t_target={
                "values": [120.0, 100.0],
                "unit": "degC",
            },
            heat_flow={
                "values": [100.0, 80.0],
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

    copied = sc.copy()
    hot_streams = sc.get_hot_streams()

    assert copied.period_ids == {"0": 0, "1": 1}
    np.testing.assert_allclose(copied.weights, np.array([0.25, 0.75]))
    assert hot_streams.period_ids == {"0": 0, "1": 1}
    np.testing.assert_allclose(hot_streams.weights, np.array([0.25, 0.75]))


def test_export_to_csv(sample_streams):
    sc = StreamCollection()
    for stream in sample_streams:
        sc.add(stream)

    filename = f"test_streams_{uuid4().hex}.csv"
    output_path = export_stream_collection(sc, filename)

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
            "dt_cont_multiplier",
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


def test_stream_collection_warning_paths_and_name_helpers():
    empty = StreamCollection()

    assert empty.num_periods == 1
    with pytest.warns(UserWarning, match="empty stream collection"):
        assert empty.sum_stream_attribute("heat_flow") == pytest.approx(0.0)
    with pytest.warns(UserWarning, match="empty stream collection"):
        assert empty.set_common_stream_attribute("heat_flow", 0.0) == pytest.approx(0.0)

    collection = StreamCollection()
    collection.add(Stream(name="Steam", t_supply=200, t_target=150, heat_flow=10.0))

    assert collection.get_stream_by_name("tea", approximate=True).name == "Steam"
    assert collection.get_stream_names() == ["Steam"]
    with pytest.warns(UserWarning, match="not found"):
        assert collection.get_stream_by_name("missing") is None
    with pytest.warns(UserWarning, match="does not have attribute"):
        assert collection.sum_stream_attribute("missing") == pytest.approx(0.0)
    with pytest.warns(UserWarning, match="does not have attribute"):
        collection.set_common_stream_attribute("missing", 1.0)
    with pytest.warns(UserWarning, match="not found"):
        collection.remove("missing")


def test_stream_collection_period_context_validation_and_adoption_edges():
    collection = StreamCollection()
    collection.set_period_context({"0": 0, "1": 1}, [0.5, 0.5], 2)
    three_period_stream = Stream(
        name="H3",
        t_supply={"values": [200.0, 190.0, 180.0], "unit": "degC"},
        t_target={"values": [120.0, 110.0, 100.0], "unit": "degC"},
        heat_flow={"values": [1.0, 1.0, 1.0], "unit": "kW"},
    )

    with pytest.raises(ValueError, match="must align"):
        collection.add(three_period_stream)

    left = StreamCollection()
    left.set_period_context(None, None, None)
    right = StreamCollection()
    right.set_period_context({"0": 0, "1": 1}, [0.25, 0.75], 2)
    combined = left + right

    assert combined.period_ids == {"0": 0, "1": 1}

    different = StreamCollection()
    different.set_period_context({"base": 0, "peak": 1}, [0.5, 0.5], 2)
    with pytest.raises(ValueError, match="different period_ids"):
        right + different


def test_stream_collection_pickle_state_and_sort_edge_helpers():
    collection = StreamCollection()
    collection.add(Stream(name="H1", t_supply=200, t_target=120, heat_flow=1.0))
    collection.set_sort_key(lambda stream: stream.name)

    state = collection.__getstate__()
    assert state["_sort_spec"] == ("attr", "t_supply")
    state.pop("_numeric_cache")

    restored = StreamCollection()
    restored.__setstate__(state)

    assert restored.numeric_view().cp.tolist() == [pytest.approx(0.0125)]
    assert StreamCollection._descending_sort_value(float("nan")) == float("inf")
    assert (
        StreamCollection._dict_category(
            SimpleNamespace(is_process_stream=False, type="Other")
        )
        == "other"
    )


def test_stream_collection_subset_sort_and_inverted_utility_edges():
    collection = StreamCollection()
    collection.add(
        Stream(
            name="HU",
            t_supply=220,
            t_target=180,
            heat_flow=1.0,
            is_process_stream=False,
        )
    )
    collection.add(
        Stream(
            name="CU",
            t_supply=40,
            t_target=80,
            heat_flow=1.0,
            is_process_stream=False,
        )
    )
    collection.add(Stream(name="H1", t_supply=200, t_target=100, heat_flow=1.0))

    hot_subset = collection.get_hot_streams(sort_attr="name")
    inverted_hot_utilities = collection.get_inverted_hot_utility_streams()
    inverted_cold_utilities = collection.get_inverted_cold_utility_streams()

    assert [stream.name for stream in hot_subset] == ["HU", "H1"]
    assert set(inverted_hot_utilities._streams) == {"CU"}
    assert inverted_hot_utilities["CU"].type == ST.Hot.value
    assert set(inverted_cold_utilities._streams) == {"HU"}
    assert inverted_cold_utilities["HU"].type == ST.Cold.value
