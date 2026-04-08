"""Additional branch coverage tests for web graphing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from OpenPinch.classes.energy_target import EnergyTarget
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.enums import ArrowHead
from OpenPinch.streamlit_webviewer import web_graphing as wg


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Sidebar:
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


class _StreamlitStub:
    def __init__(self):
        self.sidebar = _Sidebar(self)
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
        return [_Ctx() for _ in labels]

    def columns(self, n):
        return [_Ctx() for _ in range(n)]

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
    st = _StreamlitStub()
    monkeypatch.setitem(sys.modules, "streamlit", st)

    zone = Zone(name="Plant")
    target = EnergyTarget(name="Plant/DI")
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
