"""Tests covering plotting helpers in ``OpenPinch.utils.plots``."""

from __future__ import annotations

import numpy as np

from OpenPinch.utils import plots


def test_plot_t_h_curve_and_piecewise_bounds(monkeypatch):
    shown = {"count": 0}

    def _fake_show(self):
        shown["count"] += 1

    monkeypatch.setattr(plots.go.Figure, "show", _fake_show)

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

    plots.plot_t_h_curve(points, title="base")
    plots.plot_t_h_curve_with_piecewise_and_bounds(
        points=points,
        piecewise_points=piecewise,
        epsilon=3.0,
        title="piecewise",
    )

    assert shown["count"] == 2
