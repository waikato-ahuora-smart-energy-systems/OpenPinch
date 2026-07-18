"""Tests for Streamlit graphing helpers."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import OpenPinch.presentation.dashboard.dependencies as dashboard_dependencies
import OpenPinch.presentation.dashboard.exports as dashboard_exports
import OpenPinch.presentation.dashboard.rendering as dashboard
import OpenPinch.presentation.graphs.plotly as plotly_graphs
from OpenPinch.domain.enums import ProblemTableLabel as ProblemTableLabel
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import DirectIntegrationTarget, EnergyTransferTarget
from OpenPinch.domain.zone import Zone
from OpenPinch.presentation.reporting.problem_table import problem_table_frame


class _CtxExtra:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _SidebarStub:
    def __init__(self, st):
        self._st = st

    def selectbox(self, _label, options, index=0, **_kwargs):
        if options:
            return options[index]
        return None

    def divider(self):
        self._st.calls.append("sidebar.divider")

    def write(self, _text):
        self._st.calls.append("sidebar.write")

    def markdown(self, _text, **_kwargs):
        self._st.calls.append("sidebar.markdown")


class _StreamlitStub:
    def __init__(
        self, *, button_value: bool = False, text_input_value: str | None = None
    ):
        self.button_value = button_value
        self.text_input_value = text_input_value
        self.calls = []
        self.errors = []
        self.successes = []
        self.warnings = []
        self.infos = []
        self.sidebar = _SidebarStub(self)

    def set_page_config(self, **_kwargs):
        self.calls.append("set_page_config")

    def markdown(self, _text, **_kwargs):
        self.calls.append("markdown")

    def warning(self, msg):
        self.warnings.append(str(msg))

    def info(self, msg):
        self.infos.append(str(msg))

    def tabs(self, labels):
        self.calls.append(("tabs", tuple(labels)))
        return [_Ctx() for _ in labels]

    def columns(self, n):
        self.calls.append(("columns", n))
        return [_Ctx() for _ in range(n)]

    def plotly_chart(self, *_args, **_kwargs):
        self.calls.append("plotly_chart")

    def badge(self, _text):
        self.calls.append("badge")

    def dataframe(self, _df, **_kwargs):
        self.calls.append("dataframe")

    def text_input(self, _label, default, **_kwargs):
        return default if self.text_input_value is None else self.text_input_value

    def button(self, _label, **_kwargs):
        return self.button_value

    def error(self, msg):
        self.errors.append(str(msg))

    def success(self, msg):
        self.successes.append(str(msg))


def _make_target(name: str) -> DirectIntegrationTarget:
    hu = StreamCollection()
    hu.add(
        Stream(
            name="HP Steam",
            supply_temperature=180.0,
            target_temperature=170.0,
            heat_flow=75.0,
            is_process_stream=False,
        )
    )
    cu = StreamCollection()
    cu.add(
        Stream(
            name="Cooling Water",
            supply_temperature=30.0,
            target_temperature=40.0,
            heat_flow=40.0,
            is_process_stream=False,
        )
    )

    pt = ProblemTable(
        {
            ProblemTableLabel.T: [150.0, 100.0, 60.0],
            ProblemTableLabel.H_HOT: [30.0, 15.0, 0.0],
            ProblemTableLabel.H_COLD: [0.0, 10.0, 20.0],
        }
    )
    return DirectIntegrationTarget(
        zone_name=name,
        type="Direct Integration",
        cold_pinch=80.0,
        hot_pinch=120.0,
        hot_utility_target=100.0,
        cold_utility_target=60.0,
        heat_recovery_target=140.0,
        degree_of_int=0.7,
        hot_utilities=hu,
        cold_utilities=cu,
        pt=pt,
        pt_real=pt.copy,
    )


def test_collect_targets_and_problem_table_dataframe_helpers():
    parent = Zone(name="Parent")
    child = Zone(name="Child")
    parent.add_zone(child)

    parent_target = _make_target("Parent")
    child_target = _make_target("Child")
    parent.add_target(parent_target)
    child.add_target(child_target)

    targets = dashboard._collect_targets(parent)
    assert "Parent/Direct Integration" in targets
    assert "Child/Direct Integration" in targets

    assert problem_table_frame(None).empty
    empty_table = SimpleNamespace(data=np.array([]), columns=[])
    assert problem_table_frame(empty_table).empty

    df = problem_table_frame(parent_target.pt, round_decimals=1)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_plotly_helpers_cover_legend_logic():
    graph = {
        "type": "Total Site Profiles",
        "segments": [
            {
                "title": "Segment 1",
                "series": "Process",
                "series_id": "proc",
                "colour": 0,
                "data_points": [{"x": 0.0, "y": 40.0}, {"x": 10.0, "y": 60.0}],
            },
            {
                "title": "Vertical Utility",
                "series_description": "Utility leg",
                "is_vertical": True,
                "is_utility_stream": True,
                "data_points": [{"x": 5.0, "y": 20.0}, {"x": 5.0, "y": 30.0}],
            },
        ],
    }
    fig = plotly_graphs.build_plotly_figure(graph)
    assert len(fig.data) == 2
    assert len(fig.layout.annotations) == 0

    traces = plotly_graphs._segment_traces(
        segment={"data_points": [{"x": 1.0, "y": 1.0}], "title": "Flat"},
        graph={"type": "Grand Composite Curve"},
        legend_seen={},
    )
    assert len(traces) == 1

    traces = plotly_graphs._segment_traces(
        segment={"data_points": [], "title": "Empty"},
        graph={"type": "Grand Composite Curve"},
        legend_seen={},
    )
    assert traces == []

    assert plotly_graphs._is_vertical_segment([1.0, 1.0, 1.0])
    assert not plotly_graphs._is_vertical_segment([1.0, 2.0])
    assert plotly_graphs._legend_group_name("Segment 2") == "Segment"
    assert plotly_graphs._legend_group_name("OnlyName") == "OnlyName"


def test_build_download_handles_empty_path_success_and_oserror(monkeypatch, tmp_path):
    df = pd.DataFrame({"A": [1.0, 2.0]})

    st_empty = _StreamlitStub(button_value=True, text_input_value="   ")
    dashboard_exports.render_table_export(
        st_empty,
        "results/file.xlsx",
        dashboard_key="base",
        target_name="z",
        frame=df,
        table_kind="shifted",
    )
    assert st_empty.errors

    out_path = tmp_path / "table.xlsx"
    st_success = _StreamlitStub(button_value=True, text_input_value=str(out_path))
    dashboard_exports.render_table_export(
        st_success,
        str(out_path),
        dashboard_key="base",
        target_name="z",
        frame=df,
        table_kind="real",
    )
    assert out_path.exists()
    assert st_success.successes

    def _raise_oserror(*_args, **_kwargs):
        raise OSError("disk full")

    st_fail = _StreamlitStub(
        button_value=True, text_input_value=str(tmp_path / "broken.xlsx")
    )
    monkeypatch.setattr(dashboard_exports.Path, "write_bytes", _raise_oserror)
    dashboard_exports.render_table_export(
        st_fail,
        "unused.xlsx",
        dashboard_key="base",
        target_name="z",
        frame=df,
        table_kind="err",
    )
    assert any("Failed to save file" in msg for msg in st_fail.errors)


def test_apply_dashboard_theme_and_render_dashboard_branches(monkeypatch, tmp_path):
    st = _StreamlitStub(button_value=False)
    dashboard._apply_dashboard_theme(st)
    assert "markdown" in st.calls

    st_render = _StreamlitStub(
        button_value=False, text_input_value=str(tmp_path / "skip.xlsx")
    )
    monkeypatch.setitem(__import__("sys").modules, "streamlit", st_render)

    zone = Zone(name="Master")
    target = _make_target("Master")
    zone.add_target(target)

    graph_data = {
        "Master/Direct Integration": {
            "name": "Master/Direct Integration",
            "graphs": [
                {
                    "name": "GCC",
                    "type": "Site Utility Grand Composite Curve",
                    "segments": [
                        {
                            "title": "Curve 1",
                            "series_id": "g1",
                            "colour": 0,
                            "data_points": [
                                {"x": 0.0, "y": 70.0},
                                {"x": 20.0, "y": 120.0},
                            ],
                        }
                    ],
                }
            ],
        }
    }

    dashboard.render_streamlit_dashboard(
        zone,
        graph_data=graph_data,
        page_title="Master Dashboard",
        value_rounding=2,
    )
    assert "set_page_config" in st_render.calls
    assert any(call == "plotly_chart" for call in st_render.calls)
    assert not st_render.warnings


def test_render_dashboard_no_targets_issues_warning(monkeypatch):
    st_render = _StreamlitStub(button_value=False)
    monkeypatch.setitem(__import__("sys").modules, "streamlit", st_render)

    zone = Zone(name="NoTargets")
    dashboard.render_streamlit_dashboard(zone, graph_data={})

    assert st_render.warnings
    assert "No targets available" in st_render.warnings[0]


# ===== Merged from test_web_graphing_extra.py =====
"""Additional branch coverage tests for web graphing."""


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _SidebarExtra:
    def __init__(self, parent):
        self.parent = parent

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]

    def divider(self):
        return None

    def write(self, *_args, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None


class _StreamlitStubExtra:
    def __init__(self):
        self.sidebar = _SidebarExtra(self)
        self.infos = []
        self.calls = []

    def set_page_config(self, **_kwargs):
        self.calls.append("set_page_config")

    def markdown(self, *_args, **_kwargs):
        self.calls.append("markdown")

    def warning(self, _msg):
        return None

    def info(self, msg):
        self.infos.append(str(msg))

    def tabs(self, labels):
        return [_CtxExtra() for _ in labels]

    def columns(self, n):
        return [_CtxExtra() for _ in range(n)]

    def plotly_chart(self, *_args, **_kwargs):
        self.calls.append("plotly_chart")

    def badge(self, *_args, **_kwargs):
        self.calls.append("badge")

    def dataframe(self, *_args, **_kwargs):
        self.calls.append("dataframe")

    def text_input(self, _label, default, **_kwargs):
        return default

    def button(self, _label, **_kwargs):
        return False

    def error(self, _msg):
        return None

    def success(self, _msg):
        return None


def test_render_streamlit_dashboard_empty_graph_and_problem_tables(monkeypatch):
    st = _StreamlitStubExtra()
    monkeypatch.setitem(sys.modules, "streamlit", st)

    zone = Zone(name="Plant")
    target = DirectIntegrationTarget(
        zone_name="Plant",
        type="Direct Integration",
        pt=ProblemTable({ProblemTableLabel.T: []}),
        pt_real=ProblemTable({ProblemTableLabel.T: []}),
        cold_pinch=80.0,
        hot_pinch=120.0,
        hot_utility_target=100.0,
        cold_utility_target=50.0,
        heat_recovery_target=75.0,
        degree_of_int=0.5,
    )
    zone.add_target(target)

    dashboard.render_streamlit_dashboard(zone, graph_data={})

    assert any("No graphs available" in msg for msg in st.infos)
    assert any("No shifted problem table data" in msg for msg in st.infos)
    assert any("No real temperature Problem Table data" in msg for msg in st.infos)


def test_require_streamlit_imports_when_not_cached(monkeypatch):
    st = _StreamlitStub()
    sys.modules.pop("streamlit", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit":
            return st
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert dashboard_dependencies._require_streamlit() is st


def test_render_streamlit_dashboard_handles_energy_transfer_target(monkeypatch):
    st = _StreamlitStubExtra()
    monkeypatch.setitem(sys.modules, "streamlit", st)

    zone = Zone(name="Plant")
    target = EnergyTransferTarget(
        zone_name="Plant",
        type="Energy Transfer Analysis",
        pt=ProblemTable(
            {ProblemTableLabel.T: [150.0, 100.0], ProblemTableLabel.H_NET: [20.0, 0.0]}
        ),
        cold_pinch=80.0,
        hot_pinch=120.0,
        hot_utility_target=20.0,
        cold_utility_target=0.0,
        heat_recovery_target=50.0,
        degree_of_int=1.0,
        base_target_type="Direct Integration",
        base_target_name="Plant/Direct Integration",
    )
    zone.add_target(target)

    dashboard.render_streamlit_dashboard(
        zone,
        graph_data={
            "Plant/Energy Transfer Analysis": {
                "name": "Plant/Energy Transfer Analysis",
                "graphs": [
                    {
                        "type": "Energy Transfer Diagram",
                        "name": "ETD",
                        "segments": [
                            {
                                "title": "Transfer 1",
                                "colour": 0,
                                "data_points": [
                                    {"x": 20.0, "y": 150.0},
                                    {"x": 0.0, "y": 100.0},
                                ],
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert "plotly_chart" in st.calls
    assert any("No real temperature Problem Table data" in msg for msg in st.infos)


def test_segment_trace_vertical_colour_and_zero_length_cases():
    traces = plotly_graphs._segment_traces(
        segment={
            "data_points": [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
            "is_vertical": True,
            "is_utility_stream": True,
        },
        graph={"type": "Site Utility Grand Composite Curve"},
        legend_seen={},
    )
    assert len(traces) == 1

    traces2 = plotly_graphs._segment_traces(
        segment={
            "data_points": [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0}],
        },
        graph={"type": "Grand Composite Curve"},
        legend_seen={},
    )
    assert len(traces2) == 1

    assert plotly_graphs._is_vertical_segment([1.0]) is False
    assert plotly_graphs._legend_group_name("") == "Segment"
    assert (
        plotly_graphs._segment_colour(
            {
                "is_vertical": True,
                "is_utility_stream": False,
                "colour": 0,
            }
        )
        == "#111111"
    )
