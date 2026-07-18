"""Regression tests for graphs analysis routines."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from OpenPinch.analysis.graphs.composite import (
    _graph_cc,
    _make_composite_graph,
    clean_composite_curve,
    clean_composite_curve_ends,
)
from OpenPinch.analysis.graphs.grand_composite import (
    _build_gcc_segments,
    _classify_segment,
    _make_gcc_graph,
    _normalise_gcc_flags,
    _segment_streamloc,
)
from OpenPinch.analysis.graphs.metadata import _GraphSeriesMeta
from OpenPinch.analysis.graphs.primitives import (
    _column_to_list,
    _create_curve,
    _normalise_graph_fields,
    _normalise_graph_values,
    _series_meta_from_key,
    _should_plot_series,
    _streamloc_colour,
)
from OpenPinch.analysis.graphs.service import (
    _create_graph_set,
    _make_energy_transfer_diagram_graph,
    build_graph_from_spec,
    get_output_graph_data,
    iter_available_graph_specs,
)
from OpenPinch.analysis.graphs.specifications import (
    COMPOSITE_GRAPH_SPECS,
    GCC_GRAPH_SPECS,
    GRAPH_BUILD_SPECS,
    _GraphBuildSpec,
)
from OpenPinch.domain.enums import (
    ArrowHead,
    GraphType,
    LineColour,
    ProblemTableLabel,
    StreamLoc,
    TargetType,
)
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.zone import Zone
from tests.support.paths import FIXTURES_ROOT

GRAPH_FIXTURE = FIXTURES_ROOT / "graph_data_cases.json"


def _graph_fixture(name: str):
    return json.loads(GRAPH_FIXTURE.read_text())[name]


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
    result = _graph_cc("Hot CC", StreamLoc.HotS, y_vals, x_vals)
    assert result[0]["title"] == "Hot CC"
    assert result[0]["arrow"] == ArrowHead.END.value


def test_graph_cc_invalid_type():
    with pytest.raises(TypeError, match="StreamLoc"):
        _graph_cc("CC", "Invalid", [1, 2], [3, 4])


def test_graph_cc_uses_total_site_arrow_direction():
    [segment] = _graph_cc(
        GraphType.TSP.value,
        StreamLoc.HotS,
        [220.0, 180.0, 140.0],
        [0.0, -30.0, -60.0],
        include_arrows=True,
    )

    assert segment["title"] == "Hot CC"
    assert segment["arrow"] == ArrowHead.START.value


def test_graph_cc_rejects_unmapped_stream_location_enum():
    with pytest.raises(
        ValueError, match="Unrecognised composite curve stream location"
    ):
        _graph_cc("CC", StreamLoc.Unassigned, [1, 2], [3, 4])


def test_graph_gcc_creates_segments():
    x_vals = [10, 6, 0, 5]
    y_vals = [100, 90, 80, 70]
    segments = _build_gcc_segments(
        y_vals,
        x_vals,
        series_id="GCC",
        meta=_GraphSeriesMeta(label="GCC", description="GCC"),
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
        meta=_GraphSeriesMeta(label="GCC", description="GCC"),
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


def test_graph_helper_normalisation_and_series_meta_edges():
    assert _normalise_graph_fields("H(net)") == ["H(net)"]
    assert _normalise_graph_fields(("H(hot)", "H(cold)")) == ["H(hot)", "H(cold)"]
    assert _normalise_graph_values("Hot", 2, "bad") == ["Hot", "Hot"]

    with pytest.raises(ValueError, match="bad"):
        _normalise_graph_values(["Hot"], 2, "bad")

    class BadBool:
        def __bool__(self):
            raise TypeError("no bool")

    with pytest.raises(ValueError, match="coercible to bool"):
        _normalise_gcc_flags([BadBool()], 1)

    meta = _series_meta_from_key("unregistered-series")
    assert meta.label == "unregistered-series"
    assert meta.description == "unregistered-series"


def test_graph_column_extraction_supports_table_column_and_index_protocols():
    table = ProblemTable(
        {ProblemTableLabel.T: [100.0, 80.0], ProblemTableLabel.H_NET: [0.0, 20.0]}
    )
    assert _column_to_list(table, ProblemTableLabel.H_NET) == [0.0, 20.0]

    class ColumnBacked:
        columns = ["heat"]
        col = {"heat": np.array([1.0, 2.0])}

    assert _column_to_list(ColumnBacked(), "heat") == [1.0, 2.0]

    class IndexBacked:
        def __getitem__(self, key):
            if key != "heat":
                raise KeyError(key)
            return (3.0, 4.0)

    assert _column_to_list(IndexBacked(), "heat") == [3.0, 4.0]

    with pytest.raises(KeyError, match="Column 'missing'"):
        _column_to_list(object(), "missing")


def test_graph_series_and_segment_low_level_edges():
    class FallbackValues:
        def __array__(self, dtype=None):
            raise TypeError("container conversion failed")

        def __iter__(self):
            return iter([None, "2.5"])

    assert _should_plot_series(FallbackValues()) is True
    assert _should_plot_series([np.nan, np.inf]) is False

    assert _classify_segment(np.nan, is_utility_profile=False) == StreamLoc.Unassigned
    assert _segment_streamloc(StreamLoc.HotU) == StreamLoc.HotU
    for value in (
        StreamLoc.ColdS.value,
        StreamLoc.HotS.value,
        StreamLoc.HotU.value,
        StreamLoc.ColdU.value,
        "unknown",
    ):
        with pytest.raises(TypeError, match="StreamLoc"):
            _segment_streamloc(value)
    assert _streamloc_colour(StreamLoc.Unassigned) == LineColour.Black.value
    assert _streamloc_colour("custom") == LineColour.Other.value


# ----------------------------------------------------------------------------------------------------
# Integration Tests for Graph Set Creation
# ----------------------------------------------------------------------------------------------------


def test_graph_build_specs_reference_valid_labels_and_metadata_lengths():
    assert all(isinstance(spec.graph_type, GraphType) for spec in GRAPH_BUILD_SPECS)
    assert all(
        field in ProblemTableLabel
        for spec in GRAPH_BUILD_SPECS
        for field in spec.value_fields
    )

    for spec in COMPOSITE_GRAPH_SPECS:
        assert spec.builder == "composite"
        assert len(spec.value_fields) == len(spec.stream_types)
        assert all(stream_type in StreamLoc for stream_type in spec.stream_types)

    for spec in GCC_GRAPH_SPECS:
        assert spec.builder == "gcc"
        assert len(spec.value_fields) == len(spec.utility_profile_flags)
        assert all(isinstance(flag, bool) for flag in spec.utility_profile_flags)


def test_iter_available_graph_specs_uses_canonical_order():
    target_graphs = {
        GraphType.ETD.value: {},
        GraphType.GCC.value: {},
        GraphType.CC.value: {},
        GraphType.NLP.value: {},
    }

    assert [spec.graph_type for spec in iter_available_graph_specs(target_graphs)] == [
        GraphType.CC,
        GraphType.GCC,
        GraphType.NLP,
        GraphType.ETD,
    ]


def test_build_graph_from_spec_rejects_unknown_builder():
    spec = _GraphBuildSpec(
        graph_type=GraphType.CC,
        label="Bad",
        builder="unknown",
    )

    with pytest.raises(ValueError, match="Unsupported graph builder"):
        build_graph_from_spec("Target", {GraphType.CC.value: {}}, spec)


def test_create_graph_set_with_mock_zone():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TargetType.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GraphType.CC.value: {
            ProblemTableLabel.T.value: MagicMock(to_list=lambda: [100, 200]),
            ProblemTableLabel.H_HOT.value: MagicMock(to_list=lambda: [100, 150]),
            ProblemTableLabel.H_COLD.value: MagicMock(to_list=lambda: [80, 130]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)
    assert graph_set["name"] == "ZoneA/Direct Integration"
    assert graph_set["target_type"] == TargetType.DI.value
    assert graph_set["zone_name"] == "ZoneA"
    assert graph_set["zone_address"] == "Site/ZoneA"
    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GraphType.CC.value


def test_create_graph_set_includes_net_load_curves():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TargetType.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GraphType.NLP.value: {
            ProblemTableLabel.T.value: MagicMock(to_list=lambda: [200, 150, 100]),
            ProblemTableLabel.H_NET_HOT.value: MagicMock(to_list=lambda: [0, -20, -40]),
            ProblemTableLabel.H_NET_COLD.value: MagicMock(to_list=lambda: [30, 10, 0]),
            ProblemTableLabel.H_HOT_UT.value: MagicMock(to_list=lambda: [5, 5, 5]),
            ProblemTableLabel.H_COLD_UT.value: MagicMock(to_list=lambda: [25, 15, 5]),
            ProblemTableLabel.H_HOT_HP.value: MagicMock(to_list=lambda: [0, 0, 0]),
            ProblemTableLabel.H_COLD_HP.value: MagicMock(to_list=lambda: [0, 0, 0]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GraphType.NLP.value
    assert graph_set["graphs"][0]["name"] == "Net Load Curves: ZoneA/Direct Integration"
    assert len(graph_set["graphs"][0]["segments"]) == 4


def test_create_graph_set_includes_real_grand_composite_curve():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TargetType.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GraphType.GCC_R.value: {
            ProblemTableLabel.T.value: MagicMock(to_list=lambda: [200, 150, 100]),
            ProblemTableLabel.H_NET.value: MagicMock(to_list=lambda: [0, 20, 40]),
            ProblemTableLabel.H_NET_UT.value: MagicMock(to_list=lambda: [0, 5, 10]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GraphType.GCC_R.value
    assert (
        graph_set["graphs"][0]["name"]
        == "Grand Composite Curve (Real): ZoneA/Direct Integration"
    )


def test_create_graph_set_includes_energy_transfer_diagram():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Energy Transfer Analysis"
    mock_target.type = TargetType.ET.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GraphType.ETD.value: {
            ProblemTableLabel.T.value: MagicMock(to_list=lambda: [200, 150, 100]),
            ProblemTableLabel.H_NET.value: MagicMock(to_list=lambda: [10, 40, 5]),
        }
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert len(graph_set["graphs"]) == 1
    assert graph_set["graphs"][0]["type"] == GraphType.ETD.value
    assert (
        graph_set["graphs"][0]["name"]
        == "Energy Transfer Diagram: ZoneA/Energy Transfer Analysis"
    )


def test_create_graph_set_includes_exergy_graphs():
    mock_target = MagicMock()
    mock_target.name = "ZoneA/Direct Integration"
    mock_target.type = TargetType.DI.value
    mock_target.zone_name = "ZoneA"
    mock_target.graphs = {
        GraphType.GCC_X.value: {
            ProblemTableLabel.T.value: MagicMock(to_list=lambda: [10, 5, 0]),
            "X(net)": MagicMock(to_list=lambda: [0, 4, 8]),
        },
        GraphType.NLP_X.value: {
            ProblemTableLabel.T.value: MagicMock(to_list=lambda: [10, 5, 0]),
            "X(surplus)": MagicMock(to_list=lambda: [8, 4, 0]),
            "X(deficit)": MagicMock(to_list=lambda: [0, 2, 4]),
        },
    }
    mock_zone = MagicMock(spec=Zone)
    mock_zone.name = "ZoneA"
    mock_zone.address = "Site/ZoneA"

    graph_set = _create_graph_set(mock_target, zone=mock_zone)

    assert {graph["type"] for graph in graph_set["graphs"]} == {
        GraphType.GCC_X.value,
        GraphType.NLP_X.value,
    }


def test_create_graph_set_includes_remaining_graph_families_from_static_fixture():
    table = _graph_fixture("base_table")
    target = SimpleNamespace(
        name="Site/Total Site",
        type=TargetType.TS.value,
        period_id="annual",
        zone_name="TargetZone",
        graphs={
            GraphType.SCC.value: table,
            GraphType.BCC.value: table,
            GraphType.GCC.value: table,
            GraphType.NLP_HP.value: table,
            GraphType.TSP.value: table,
            GraphType.SUGCC.value: table,
            GraphType.GCC_HP.value: table,
        },
    )
    zone = SimpleNamespace(name="ZoneFromContext", address="Site/ZoneFromContext")

    graph_set = _create_graph_set(target, zone=zone)

    assert graph_set["period_id"] == "annual"
    assert graph_set["zone_name"] == "ZoneFromContext"
    assert graph_set["zone_address"] == "Site/ZoneFromContext"
    assert {graph["type"] for graph in graph_set["graphs"]} == {
        GraphType.SCC.value,
        GraphType.BCC.value,
        GraphType.GCC.value,
        GraphType.NLP_HP.value,
        GraphType.TSP.value,
        GraphType.SUGCC.value,
        GraphType.GCC_HP.value,
    }
    assert all(graph["segments"] for graph in graph_set["graphs"])


def test_make_energy_transfer_diagram_uses_operation_fixture():
    graph = _make_energy_transfer_diagram_graph(
        "Site/Energy Transfer",
        {GraphType.ETD.value: _graph_fixture("energy_transfer_diagram")},
    )

    assert graph["type"] == GraphType.ETD.value
    assert len(graph["segments"]) == 2
    assert graph["segments"][0]["series_id"] == f"{GraphType.ETD.value}:Reactor"
    assert graph["segments"][1]["series_description"] == "C cascade"


def test_make_graph_helpers_validate_mismatched_metadata_lengths():
    table = _graph_fixture("base_table")

    with pytest.raises(ValueError, match="stream_type"):
        _make_composite_graph(
            "Bad",
            GraphType.CC.value,
            table,
            "Composite",
            value_field=[ProblemTableLabel.H_HOT, ProblemTableLabel.H_COLD],
            stream_type=[StreamLoc.HotS],
        )

    with pytest.raises(ValueError, match="is_utility_profile"):
        _make_gcc_graph(
            "Bad",
            GraphType.GCC.value,
            table,
            "Grand",
            value_field=[ProblemTableLabel.H_NET, ProblemTableLabel.H_NET_UT],
            is_utility_profile=[False],
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
    target.type = TargetType.DI.value
    target.graphs = {}
    zone.targets = {TargetType.DI.value: target}

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

    area_a._targets = {TargetType.DI.value: MagicMock()}
    area_a._targets[TargetType.DI.value].name = "Site/AreaA/Direct Integration"
    area_a._targets[TargetType.DI.value].type = TargetType.DI.value
    area_a._targets[TargetType.DI.value].graphs = {}
    area_a._targets[TargetType.DI.value].zone_name = "AreaA"
    area_b._targets = {TargetType.DI.value: MagicMock()}
    area_b._targets[TargetType.DI.value].name = "Site/AreaB/Direct Integration"
    area_b._targets[TargetType.DI.value].type = TargetType.DI.value
    area_b._targets[TargetType.DI.value].graphs = {}
    area_b._targets[TargetType.DI.value].zone_name = "AreaB"

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


def test_clean_composite_keeps_non_linear_middle_point_when_x_values_repeat():
    y_clean, x_clean = clean_composite_curve(
        y_array=[100.0, 90.0, 80.0],
        x_array=[0.0, 5.0, 0.0],
    )

    assert np.allclose(x_clean, [0.0, 5.0, 0.0])
    assert np.allclose(y_clean, [100.0, 90.0, 80.0])


def test_clean_composite_preserves_corner_after_consecutive_duplicate_point():
    y_clean, x_clean = clean_composite_curve(
        y_array=[
            280.0,
            279.2455,
            279.0,
            138.5254,
            138.5254,
            138.4254,
            138.4254,
            80.0,
            70.0,
        ],
        x_array=[
            180094.613,
            64778.462,
            27253.7105,
            27253.7105,
            27253.7105,
            11265.8857,
            11259.168,
            11259.168,
            0.0,
        ],
    )

    assert any(
        np.isclose(x, 27253.7105) and np.isclose(y, 138.5254)
        for x, y in zip(x_clean, y_clean)
    )


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
        "OpenPinch.analysis.graphs.composite.clean_composite_curve_ends",
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
