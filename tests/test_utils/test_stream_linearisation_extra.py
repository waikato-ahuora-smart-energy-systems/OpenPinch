"""Additional coverage tests for stream linearisation helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.utils import stream_linearisation as sl


def test_get_piecewise_linearisation_for_streams_mismatched_lengths():
    streams = [SimpleNamespace(t_supply=150.0, t_target=80.0)]
    with pytest.raises(ValueError, match="different number of streams"):
        sl.get_piecewise_linearisation_for_streams(
            streams, t_h_data=[], dt_diff_max=0.2
        )


def test_get_piecewise_linearisation_for_streams_returns_last_mask_points():
    streams = [
        SimpleNamespace(t_supply=150.0, t_target=80.0),
        SimpleNamespace(t_supply=60.0, t_target=120.0),
    ]
    t_h_data = [
        [[0.0, 150.0], [1.0, 110.0], [2.0, 80.0]],
        [[0.0, 60.0], [1.0, 90.0], [2.0, 120.0]],
    ]

    out = sl.get_piecewise_linearisation_for_streams(
        streams, t_h_data=t_h_data, dt_diff_max=0.2
    )

    assert isinstance(out, dict)
    assert "t_h_points" in out
    assert out["t_h_points"][0] == [0.0, 60.0]


def test_get_piecewise_data_points_falls_back_to_rdp(monkeypatch):
    monkeypatch.setattr(
        sl,
        "_get_piecewise_breakpoints",
        lambda **_: (_ for _ in ()).throw(RuntimeError),
    )
    monkeypatch.setattr(
        sl,
        "_rdp",
        lambda curve, epsilon: np.array(
            [[curve[0, 0], curve[0, 1]], [curve[-1, 0], curve[-1, 1]]]
        ),
    )

    curve = [[0.0, 140.0], [1.0, 120.0], [2.0, 100.0]]
    out = sl.get_piecewise_data_points(curve=curve, is_hot_stream=True, dt_diff_max=0.1)

    assert out.shape == (2, 2)


def test_get_piecewise_data_points_raises_when_all_methods_fail(monkeypatch):
    monkeypatch.setattr(
        sl,
        "_get_piecewise_breakpoints",
        lambda **_: (_ for _ in ()).throw(RuntimeError),
    )
    monkeypatch.setattr(sl, "_rdp", lambda **_: (_ for _ in ()).throw(RuntimeError))

    with pytest.raises(ValueError, match="Piecewise linearisation failed"):
        sl.get_piecewise_data_points(
            curve=[[0.0, 100.0], [1.0, 90.0]],
            is_hot_stream=True,
            dt_diff_max=0.1,
        )


def test_refine_piecewise_points_hot_stream_branch_executes():
    x = np.linspace(0, 11, 12)
    y = 150.0 - 3.0 * x + 0.02 * x**2
    curve = np.column_stack([x, y])
    pw_points = curve[[0, 3, 6, 9, 11]]

    refined, max_err = sl._refine_pw_points_for_heating_or_cooling(
        curve=curve,
        pw_points=pw_points,
        eps_lb=0.01,
        hot_stream=True,
    )

    assert refined.shape == pw_points.shape
    assert max_err >= 0.0


def test_get_piecewise_breakpoints_reduces_epsilon_when_error_too_large(monkeypatch):
    calls = {"refine": 0}
    dense_points = np.column_stack(
        [np.arange(11, dtype=float), np.arange(11, dtype=float)]
    )

    monkeypatch.setattr(sl, "_rdp", lambda curve, epsilon: dense_points)

    def _fake_refine(curve, pw_points, eps_lb, hot_stream):
        calls["refine"] += 1
        max_err = 1.0 if calls["refine"] == 1 else 0.0
        return pw_points, max_err

    monkeypatch.setattr(sl, "_refine_pw_points_for_heating_or_cooling", _fake_refine)

    out = sl._get_piecewise_breakpoints(
        curve=np.array([[0.0, 0.0], [1.0, 1.0]]),
        epsilon=0.1,
        is_hot_stream=True,
    )

    assert out.shape[0] == 11
    assert calls["refine"] >= 2
