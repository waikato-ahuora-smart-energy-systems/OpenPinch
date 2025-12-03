from unittest.mock import MagicMock

import pytest

from OpenPinch.analysis.graph_data import (
    _classify_segment,
    _create_curve,
    _create_graph_set,
    _graph_cc,
    _graph_gcc,
    get_output_graph_data,
    visualise_graphs,
)
from OpenPinch.classes import Zone
from OpenPinch.lib import *

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
    result = _graph_cc("Hot CC", StreamType.Hot.value, y_vals, x_vals)
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
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.graphs = {
        GT.CC.value: {
            PT.T.value: MagicMock(to_list=lambda: [100, 200]),
            PT.H_HOT.value: MagicMock(to_list=lambda: [100, 150]),
            PT.H_COLD.value: MagicMock(to_list=lambda: [80, 130]),
        }
    }
    graph_set = _create_graph_set(mock_zone, "ZoneA")
    assert graph_set["name"] == "ZoneA"
    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GT.CC.value


# ----------------------------------------------------------------------------------------------------
# Tests for visualise_graphs
# ----------------------------------------------------------------------------------------------------


def test_visualise_graphs_composite():
    graph_set = {"graphs": []}
    graph = MagicMock()
    graph.type = GT.CC.value
    graph.data = {
        PT.T.value: MagicMock(to_list=lambda: [100, 200]),
        PT.H_HOT.value: MagicMock(to_list=lambda: [150, 200]),
        PT.H_COLD.value: MagicMock(to_list=lambda: [100, 150]),
    }
    visualise_graphs(graph_set, graph)
    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GT.CC.value


def test_visualise_graphs_gcc_utility():
    graph_set = {"graphs": []}
    graph = MagicMock()
    graph.type = GT.GCC.value
    graph.data = {
        PT.T.value: MagicMock(to_list=lambda: [100, 200]),
        PT.H_NET.value: MagicMock(to_list=lambda: [0, -100]),
        PT.H_NET_A.value: MagicMock(to_list=lambda: [0, -100]),
        PT.H_NET_AI.value: MagicMock(to_list=lambda: [0, -100]),
        PT.H_NET_NP.value: MagicMock(to_list=lambda: [0, -100]),
        PT.H_NET_V.value: MagicMock(to_list=lambda: [0, -100]),
        PT.H_NET_PK.value: MagicMock(to_list=lambda: [0, -100]), 
        PT.H_NET_UT.value: MagicMock(to_list=lambda: [0, -100]), 
    }
    visualise_graphs(graph_set, graph)
    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == f"{GT.GCC.value}"


# ----------------------------------------------------------------------------------------------------
# Tests for get_output_graph_data
# ----------------------------------------------------------------------------------------------------


def test_get_output_graph_data_single_zone(monkeypatch):
    zone = MagicMock(spec=Zone)
    zone.name = "Site"
    zone.subzones = {}
    zone.targets = {TargetType.DI.value: MagicMock(name="TI", graphs={})}

    monkeypatch.setattr(
        "OpenPinch.analysis.graph_data._create_graph_set",
        lambda z, n: {"name": n, "graphs": []},
    )
    result = get_output_graph_data(zone)
    assert result[TargetType.DI.value]["graphs"] == []
