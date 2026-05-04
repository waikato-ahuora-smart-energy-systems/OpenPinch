"""Tests for Streamlit graphing helpers."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from OpenPinch.classes.energy_target import EnergyTarget
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.enums import ArrowHead, ProblemTableLabel as PT
from OpenPinch.streamlit_webviewer import web_graphing as wg
import sys
from OpenPinch.lib.enums import ArrowHead


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


def _make_target(name: str) -> EnergyTarget:
    target = EnergyTarget(zone_name=name, type="DI")
    target.cold_pinch = 80.0
    target.hot_pinch = 120.0
    target.hot_utility_target = 100.0
    target.cold_utility_target = 60.0
    target.heat_recovery_target = 140.0
    target.degree_of_int = 0.7

    hu = StreamCollection()
    hu.add(
        Stream(
            name="HP Steam",
            t_supply=180.0,
            t_target=170.0,
            heat_flow=75.0,
            is_process_stream=False,
        )
    )
    cu = StreamCollection()
    cu.add(
        Stream(
            name="Cooling Water",
            t_supply=30.0,
            t_target=40.0,
            heat_flow=40.0,
            is_process_stream=False,
        )
    )
    target.hot_utilities = hu
    target.cold_utilities = cu

    pt = ProblemTable(
        {
            PT.T.value: [150.0, 100.0, 60.0],
            PT.H_HOT.value: [30.0, 15.0, 0.0],
            PT.H_COLD.value: [0.0, 10.0, 20.0],
        }
    )
    target.pt = pt
    target.pt_real = pt.copy
    return target


def test_collect_targets_and_problem_table_dataframe_helpers():
    parent = Zone(name="Parent")
    child = Zone(name="Child")
    parent.add_zone(child)

    parent_target = _make_target("Parent")
    child_target = _make_target("Child")
    parent.add_target(parent_target)
    child.add_target(child_target)

    targets = wg.collect_targets(parent)
    assert "Parent/DI" in targets
    assert "Child/DI" in targets

    assert wg.problem_table_to_dataframe(None).empty
    empty_table = SimpleNamespace(data=np.array([]), columns=[])
    assert wg.problem_table_to_dataframe(empty_table).empty

    df = wg.problem_table_to_dataframe(parent_target.pt, round_decimals=1)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_plotly_helpers_cover_arrow_and_legend_logic():
    graph = {
        "type": "Total Site Profiles",
        "segments": [
            {
                "title": "Segment 1",
                "series": "Process",
                "series_id": "proc",
                "arrow": ArrowHead.START.value,
                "colour": 0,
                "data_points": [{"x": 0.0, "y": 40.0}, {"x": 10.0, "y": 60.0}],
            },
            {
                "title": "Vertical Utility",
                "series_description": "Utility leg",
                "arrow": ArrowHead.END.value,
                "is_vertical": True,
                "is_utility_stream": True,
                "data_points": [{"x": 5.0, "y": 20.0}, {"x": 5.0, "y": 30.0}],
            },
        ],
    }
    fig = wg._build_plotly_graph(graph)
    assert len(fig.data) == 2
    assert len(fig.layout.annotations) >= 1

    traces, ann = wg._segment_trace(
        segment={"data_points": [{"x": 1.0, "y": 1.0}], "title": "Flat"},
        graph={"type": "Grand Composite Curve"},
        legend_seen={},
    )
    assert len(traces) == 1
    assert ann is None

    traces, ann = wg._segment_trace(
        segment={"data_points": [], "title": "Empty"},
        graph={"type": "Grand Composite Curve"},
        legend_seen={},
    )
    assert traces == []
    assert ann is None

    assert wg._is_vertical_segment([1.0, 1.0, 1.0])
    assert not wg._is_vertical_segment([1.0, 2.0])
    assert wg._legend_group_name("Segment 2") == "Segment"
    assert wg._legend_group_name("OnlyName") == "OnlyName"


def test_build_download_handles_empty_path_success_and_oserror(monkeypatch, tmp_path):
    df = pd.DataFrame({"A": [1.0, 2.0]})

    st_empty = _StreamlitStub(button_value=True, text_input_value="   ")
    wg._build_download(
        st=st_empty,
        default="results/file.xlsx",
        base_key="base",
        selected_target_name="z",
        df=df,
        key_suffix="shifted",
    )
    assert st_empty.errors

    out_path = tmp_path / "table.xlsx"
    st_success = _StreamlitStub(button_value=True, text_input_value=str(out_path))
    wg._build_download(
        st=st_success,
        default=str(out_path),
        base_key="base",
        selected_target_name="z",
        df=df,
        key_suffix="real",
    )
    assert out_path.exists()
    assert st_success.successes

    def _raise_oserror(*_args, **_kwargs):
        raise OSError("disk full")

    st_fail = _StreamlitStub(
        button_value=True, text_input_value=str(tmp_path / "broken.xlsx")
    )
    monkeypatch.setattr(wg, "open", _raise_oserror, raising=False)
    wg._build_download(
        st=st_fail,
        default="unused.xlsx",
        base_key="base",
        selected_target_name="z",
        df=df,
        key_suffix="err",
    )
    assert any("Failed to save file" in msg for msg in st_fail.errors)


def test_apply_dashboard_theme_and_render_dashboard_branches(monkeypatch, tmp_path):
    st = _StreamlitStub(button_value=False)
    wg._apply_dashboard_theme(st)
    assert "markdown" in st.calls

    st_render = _StreamlitStub(
        button_value=False, text_input_value=str(tmp_path / "skip.xlsx")
    )
    monkeypatch.setitem(__import__("sys").modules, "streamlit", st_render)

    zone = Zone(name="Master")
    target = _make_target("Master/DI")
    zone.add_target(target)

    graph_payload = {
        "Master/DI": {
            "name": "Master/DI",
            "graphs": [
                {
                    "name": "GCC",
                    "type": "Site Utility Grand Composite Curve",
                    "segments": [
                        {
                            "title": "Curve 1",
                            "series_id": "g1",
                            "arrow": ArrowHead.END.value,
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

    wg.render_streamlit_dashboard(
        zone,
        graph_payload=graph_payload,
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
    wg.render_streamlit_dashboard(zone, graph_payload={})

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
    target = EnergyTarget(zone_name="Plant", type="DI")
    target.cold_pinch = 80.0
    target.hot_pinch = 120.0
    target.hot_utility_target = 100.0
    target.cold_utility_target = 50.0
    target.heat_recovery_target = 75.0
    target.degree_of_int = 0.5
    target.hot_utilities = []
    target.cold_utilities = []
    target.pt = None
    target.pt_real = None
    zone.add_target(target)

    wg.render_streamlit_dashboard(zone, graph_payload={})

    assert any("No graphs available" in msg for msg in st.infos)
    assert any("No shifted problem table data" in msg for msg in st.infos)
    assert any("No real-temperature problem table data" in msg for msg in st.infos)


def test_segment_trace_vertical_colour_zero_length_and_arrow_fallbacks():
    traces, ann = wg._segment_trace(
        segment={
            "data_points": [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
            "arrow": ArrowHead.END.value,
            "is_vertical": True,
            "is_utility_stream": True,
        },
        graph={"type": "Site Utility Grand Composite Curve"},
        legend_seen={},
    )
    assert len(traces) == 1
    assert ann is not None

    traces2, ann2 = wg._segment_trace(
        segment={
            "data_points": [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0}],
            "arrow": ArrowHead.END.value,
        },
        graph={"type": "Grand Composite Curve"},
        legend_seen={},
    )
    assert len(traces2) == 1
    assert ann2 is None

    assert wg._is_vertical_segment([1.0]) is False
    assert wg._arrow_indices([1.0, 1.0], [2.0, 2.0], ArrowHead.START.value) == (0, 1)
    assert wg._arrow_indices([1.0, 1.0], [2.0, 2.0], ArrowHead.END.value) == (1, 0)
    assert wg._legend_group_name("") == "Segment"
