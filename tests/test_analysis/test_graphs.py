"""Regression tests for graphs analysis routines."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from OpenPinch.classes import Zone
from OpenPinch.lib import *
from OpenPinch.services.common.graph_data import (
    _build_gcc_segments,
    _classify_segment,
    _create_curve,
    _create_graph_set,
    _graph_cc,
    clean_composite_curve,
    clean_composite_curve_ends,
    get_output_graph_data,
)
from OpenPinch.services.common.graph_series_meta import GraphSeriesMeta

# ----------------------------------------------------------------------------------------------------
# Unit Tests for Helper Functions
# ----------------------------------------------------------------------------------------------------


def test_create_curve_formats_data_correctly():
    x_vals = [0, 1, 2]
    y_vals = [3.14159, 2.71828, 1.61803]
    curve = _create_curve("Test Curve", LineColour.HotS.value, x_vals, y_vals)
    assert curve["title"] == "Test Curve"
    assert curve["colour"] == LineColour.HotS.value
    assert all("x" in pt and "y" in pt for pt in curve["data_points"])
    assert curve["data_points"][0] == {"x": 0, "y": 3.14}


def test_graph_cc_hot_curve():
    x_vals = [0, 10]
    y_vals = [100, 200]
    result = _graph_cc("Hot CC", ST.Hot.value, y_vals, x_vals)
    assert result[0]["title"] == "Hot CC"
    assert result[0]["arrow"] == ArrowHead.END.value


def test_graph_cc_invalid_type():
    with pytest.raises(ValueError):
        _graph_cc("CC", "Invalid", [1, 2], [3, 4])


def test_graph_gcc_creates_segments():
    x_vals = [10, 6, 0, 5]
    y_vals = [100, 90, 80, 70]
    segments = _build_gcc_segments(
        y_vals,
        x_vals,
        series_id="GCC",
        meta=GraphSeriesMeta(label="GCC", description="GCC"),
        is_utility_profile=False,
        decolour=False,
    )
    assert len(segments) == 2
    assert all("data_points" in seg for seg in segments)


def test_graph_gcc_vertical_segments_use_neutral_colour():
    x_vals = [10, 6, 6, 0]
    y_vals = [100, 90, 80, 70]
    segments = _build_gcc_segments(
        y_vals,
        x_vals,
        series_id="GCC",
        meta=GraphSeriesMeta(label="GCC", description="GCC"),
        is_utility_profile=False,
        decolour=False,
    )

    vertical_segments = [seg for seg in segments if seg.get("is_vertical")]
    assert vertical_segments
    assert all(seg["colour"] == LineColour.Black.value for seg in vertical_segments)


def test_classify_segment_utility_profile_colours():
    assert _classify_segment(5, is_utility_profile=False) == StreamLoc.ColdS
    assert _classify_segment(-5, is_utility_profile=False) == StreamLoc.HotS
    assert _classify_segment(0, is_utility_profile=False) == StreamLoc.Unassigned
    assert _classify_segment(0, is_utility_profile=False) == StreamLoc.Unassigned

    assert _classify_segment(5, is_utility_profile=True) == StreamLoc.HotU
    assert _classify_segment(-5, is_utility_profile=True) == StreamLoc.ColdU
    assert _classify_segment(0, is_utility_profile=True) == StreamLoc.Unassigned
    assert _classify_segment(0, is_utility_profile=True) == StreamLoc.Unassigned


# ----------------------------------------------------------------------------------------------------
# Integration Tests for Graph Set Creation
# ----------------------------------------------------------------------------------------------------


def test_create_graph_set_with_mock_zone():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TT.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GT.CC.value: {
            PT.T.value: MagicMock(to_list=lambda: [100, 200]),
            PT.H_HOT.value: MagicMock(to_list=lambda: [100, 150]),
            PT.H_COLD.value: MagicMock(to_list=lambda: [80, 130]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)
    assert graph_set["name"] == "ZoneA/Direct Integration"
    assert graph_set["target_type"] == TT.DI.value
    assert graph_set["zone_name"] == "ZoneA"
    assert graph_set["zone_address"] == "Site/ZoneA"
    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GT.CC.value


def test_create_graph_set_includes_net_load_curves():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TT.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GT.NLP.value: {
            PT.T.value: MagicMock(to_list=lambda: [200, 150, 100]),
            PT.H_NET_HOT.value: MagicMock(to_list=lambda: [0, -20, -40]),
            PT.H_NET_COLD.value: MagicMock(to_list=lambda: [30, 10, 0]),
            PT.H_HOT_UT.value: MagicMock(to_list=lambda: [5, 5, 5]),
            PT.H_COLD_UT.value: MagicMock(to_list=lambda: [25, 15, 5]),
            PT.H_HOT_HP.value: MagicMock(to_list=lambda: [0, 0, 0]),
            PT.H_COLD_HP.value: MagicMock(to_list=lambda: [0, 0, 0]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GT.NLP.value
    assert graph_set["graphs"][0]["name"] == "Net Load Curves: ZoneA/Direct Integration"
    assert len(graph_set["graphs"][0]["segments"]) == 4


def test_create_graph_set_includes_real_grand_composite_curve():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TT.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GT.GCC_R.value: {
            PT.T.value: MagicMock(to_list=lambda: [200, 150, 100]),
            PT.H_NET.value: MagicMock(to_list=lambda: [0, 20, 40]),
            PT.H_NET_UT.value: MagicMock(to_list=lambda: [0, 5, 10]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GT.GCC_R.value
    assert (
        graph_set["graphs"][0]["name"]
        == "Grand Composite Curve (Real): ZoneA/Direct Integration"
    )


def test_create_graph_set_includes_energy_transfer_diagram():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Energy Transfer Analysis"
    mock_target.type = TT.ET.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GT.ETD.value: {
            PT.T.value: MagicMock(to_list=lambda: [200, 150, 100]),
            PT.H_NET.value: MagicMock(to_list=lambda: [10, 40, 5]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GT.ETD.value
    assert (
        graph_set["graphs"][0]["name"]
        == "Energy Transfer Diagram: ZoneA/Energy Transfer Analysis"
    )


def test_create_graph_set_includes_exergy_graphs():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TT.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GT.GCC_X.value: {
            PT.T.value: MagicMock(to_list=lambda: [10, 5, 0]),
            "X(net)": MagicMock(to_list=lambda: [0, 4, 8]),
        },
        GT.NLP_X.value: {
            PT.T.value: MagicMock(to_list=lambda: [10, 5, 0]),
            "X(surplus)": MagicMock(to_list=lambda: [8, 4, 0]),
            "X(deficit)": MagicMock(to_list=lambda: [0, 2, 4]),
        },
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert {graph["type"] for graph in graph_set["graphs"]} == {
        GT.GCC_X.value,
        GT.NLP_X.value,
    }


# ----------------------------------------------------------------------------------------------------
# Tests for get_output_graph_data
# ----------------------------------------------------------------------------------------------------


def test_get_output_graph_data_single_zone(monkeypatch):
    zone = MagicMock(spec=Zone)
    zone.name = "Site"
    zone.address = "Site"
    zone.subzones = {}
    target = MagicMock()
    target.name = "Site/Direct Integration"
    target.type = TT.DI.value
    target.graphs = {}
    zone.targets = {TT.DI.value: target}

    monkeypatch.setattr(
        sys.modules[_create_graph_set.__module__],
        "_create_graph_set",
        lambda target, zone=None: {
            "name": target.name,
            "zone_address": zone.address,
            "graphs": [],
        },
    )
    result = get_output_graph_data(zone)
    assert result["Site/Direct Integration"]["graphs"] == []


def test_get_output_graph_data_keeps_same_target_type_for_multiple_subzones(
    monkeypatch,
):
    site = Zone(name="Site")
    area_a = Zone(name="AreaA", parent_zone=site)
    area_b = Zone(name="AreaB", parent_zone=site)
    site._subzones = {"AreaA": area_a, "AreaB": area_b}

    area_a._targets = {TT.DI.value: MagicMock()}
    area_a._targets[TT.DI.value].name = "Site/AreaA/Direct Integration"
    area_a._targets[TT.DI.value].type = TT.DI.value
    area_a._targets[TT.DI.value].graphs = {}
    area_a._targets[TT.DI.value].zone_name = "AreaA"
    area_b._targets = {TT.DI.value: MagicMock()}
    area_b._targets[TT.DI.value].name = "Site/AreaB/Direct Integration"
    area_b._targets[TT.DI.value].type = TT.DI.value
    area_b._targets[TT.DI.value].graphs = {}
    area_b._targets[TT.DI.value].zone_name = "AreaB"

    monkeypatch.setattr(
        sys.modules[_create_graph_set.__module__],
        "_create_graph_set",
        lambda target, zone=None: {
            "name": target.name,
            "target_type": target.type,
            "zone_name": zone.name,
            "zone_address": zone.address,
            "graphs": [],
        },
    )

    result = get_output_graph_data(site)

    assert set(result) == {
        "Site/AreaA/Direct Integration",
        "Site/AreaB/Direct Integration",
    }
    assert result["Site/AreaA/Direct Integration"]["zone_address"] == "Site/AreaA"
    assert result["Site/AreaB/Direct Integration"]["zone_address"] == "Site/AreaB"


def test_clean_composite_removes_redundant_points():
    x_vals = [0, 10, 20, 30, 30]
    y_vals = [0, 5, 10, 15, 30]
    y_clean, x_clean = clean_composite_curve(y_vals, x_vals)
    assert np.allclose(x_clean, [0, 30])
    assert np.allclose(y_clean, [0, 15])


def test_clean_composite_curve_ends_0():
    x_vals = [30, 50, 0, 30, 30, 30, 30]
    y_vals = [100, 80, 50, 40, 10, 5, 0]
    y_clean, x_clean = clean_composite_curve_ends(y_vals, x_vals)
    assert np.allclose(x_clean, [30, 50, 0, 30])
    assert np.allclose(y_clean, [100, 80, 50, 40])


def test_clean_composite_curve_ends_1():
    x_vals = [10, 10, 60, 10, 0, 10, 10]
    y_vals = [100, 80, 50, 40, 10, 5, 0]
    y_clean, x_clean = clean_composite_curve_ends(y_vals, x_vals)
    assert np.allclose(x_clean, [10, 60, 10, 0, 10])
    assert np.allclose(y_clean, [80, 50, 40, 10, 5])


def test_clean_composite_curve_ends_2():
    x_vals = [10, 0, 0, 0, 0, 0, 0]
    y_vals = [100, 80, 50, 40, 10, 5, 0]
    y_clean, x_clean = clean_composite_curve_ends(y_vals, x_vals)
    assert np.allclose(x_clean, [10, 0])
    assert np.allclose(y_clean, [100, 80])


def test_clean_composite_curve_ends_3():
    x_vals = [0, 0, 0, 0, 0, 0, 50]
    y_vals = [100, 80, 50, 40, 10, 5, 0]
    y_clean, x_clean = clean_composite_curve_ends(y_vals, x_vals)
    assert np.allclose(x_clean, [0, 50])
    assert np.allclose(y_clean, [5, 0])


def test_clean_composite_curve_ends_4():
    x_vals = [0, 0, 0, 0, 0, 0, 0]
    y_vals = [100, 80, 50, 40, 10, 5, 0]
    y_clean, x_clean = clean_composite_curve_ends(y_vals, x_vals)
    assert len(x_clean) == 0
    assert len(y_clean) == 0


def test_clean_composite_curve_ends_5():
    x_vals = [100, 100, 100, 100, 100, 100, 100]
    y_vals = [100, 80, 50, 40, 10, 5, 0]
    y_clean, x_clean = clean_composite_curve_ends(y_vals, x_vals)
    assert len(x_clean) == 0
    assert len(y_clean) == 0


def test_clean_composite_curve_pops_duplicate_edges():
    y_clean, x_clean = clean_composite_curve(
        y_array=[0, 1, 2, 3, 4],
        x_array=[0, 0, 1, 2, 2],
    )
    assert np.allclose(x_clean, [0, 2])
    assert np.allclose(y_clean, [1, 3])


def test_clean_composite_curve_forced_duplicate_edges(monkeypatch):
    monkeypatch.setattr(
        "OpenPinch.services.common.graph_data.clean_composite_curve_ends",
        lambda y_array, x_array: (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
        ),
    )
    y_clean, x_clean = clean_composite_curve(
        y_array=[0.0],
        x_array=[0.0],
    )
    assert np.allclose(x_clean, [0.0, 2.0])
    assert np.allclose(y_clean, [1.0, 3.0])
