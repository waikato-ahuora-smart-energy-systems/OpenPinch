"""Regression tests for graphs analysis routines."""

import sys
from unittest.mock import MagicMock

import pytest

from OpenPinch.classes import Zone
from OpenPinch.lib import *
from OpenPinch.services.common.graph_data import (
    _classify_segment,
    _create_curve,
    _create_graph_set,
    _graph_cc,
    _graph_gcc,
    get_output_graph_data,
)

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
    segments = _graph_gcc(y_vals, x_vals)
    assert len(segments) == 2
    assert all("data_points" in seg for seg in segments)


def test_graph_gcc_vertical_segments_use_neutral_colour():
    x_vals = [10, 6, 6, 0]
    y_vals = [100, 90, 80, 70]
    segments = _graph_gcc(y_vals, x_vals)

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
