"""Tests covering plotting helpers in ``OpenPinch.utils.plots``."""

from __future__ import annotations

import numpy as np

from OpenPinch.utils.plots import (
    graph_simple_cc_plot,
    plot_t_h_curve,
    plot_t_h_curve_with_piecewise_and_bounds,
    _require_plotly
)


def test_plot_t_h_curve_and_piecewise_bounds(monkeypatch):
    shown = {"count": 0}

    def _fake_show(self):
        shown["count"] += 1
    
    go = _require_plotly()

    monkeypatch.setattr(go.Figure, "show", _fake_show)

    points = np.array(
        [
            [0.0, 20.0],
            [25.0, 40.0],
            [50.0, 80.0],
        ]
    )
    piecewise = np.array(
        [
            [0.0, 20.0],
            [50.0, 80.0],
        ]
    )

    plot_t_h_curve(points, title="base")
    plot_t_h_curve_with_piecewise_and_bounds(
        points=points,
        piecewise_points=piecewise,
        epsilon=3.0,
        title="piecewise",
    )

    assert shown["count"] == 2


def test_graph_simple_cc_plot_executes_show(monkeypatch):
    shown = {"called": False}

    class _FakeScatter:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.kwargs = kwargs

    class _FakeFigure:
        def __init__(self):
            self.data = []

        def add_trace(self, trace):
            self.data.append(trace)

        def update_layout(self, **kwargs):
            self.layout = kwargs

        def update_yaxes(self, **kwargs):
            self.yaxis = kwargs

        def update_xaxes(self, **kwargs):
            self.xaxis = kwargs

        def show(self):
            shown["called"] = True

    class _FakePlotly:
        Figure = _FakeFigure
        Scatter = _FakeScatter

    monkeypatch.setattr("OpenPinch.utils.plots._require_plotly", lambda: _FakePlotly)

    figure = graph_simple_cc_plot(
        Tc=[40, 30],
        Hc=[0, 10],
        Th=[120, 110],
        Hh=[0, 12],
    )
    assert shown["called"] is True
    assert len(figure.data) == 2
    assert figure.data[0].name == "Cold composite"
    assert figure.data[1].name == "Hot composite"
