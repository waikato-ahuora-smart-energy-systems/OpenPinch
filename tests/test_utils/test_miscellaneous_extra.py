"""Additional coverage tests for miscellaneous helpers."""

import numpy as np
import pytest

from OpenPinch.utils import miscellaneous


def test_clean_composite_curve_pops_duplicate_edges():
    y_clean, x_clean = miscellaneous.clean_composite_curve(
        y_array=[0, 1, 2, 3, 4],
        x_array=[0, 0, 1, 2, 2],
    )
    assert np.allclose(x_clean, [0, 2])
    assert np.allclose(y_clean, [1, 3])


def test_clean_composite_curve_forced_duplicate_edges(monkeypatch):
    monkeypatch.setattr(
        miscellaneous,
        "clean_composite_curve_ends",
        lambda y_array, x_array: (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
        ),
    )
    y_clean, x_clean = miscellaneous.clean_composite_curve(
        y_array=[0.0],
        x_array=[0.0],
    )
    assert np.allclose(x_clean, [0.0, 2.0])
    assert np.allclose(y_clean, [1.0, 3.0])


def test_graph_simple_cc_plot_executes_show(monkeypatch):
    shown = {"called": False}
    monkeypatch.setattr(
        miscellaneous.plt,
        "show",
        lambda: shown.__setitem__("called", True),
    )
    miscellaneous.graph_simple_cc_plot(
        Tc=[40, 30],
        Hc=[0, 10],
        Th=[120, 110],
        Hh=[0, 12],
    )
    assert shown["called"] is True


def test_interp_with_plateaus_invalid_side():
    with pytest.raises(ValueError, match="side must be"):
        miscellaneous.interp_with_plateaus(
            h_vals=np.array([0.0, 1.0]),
            t_vals=np.array([100.0, 90.0]),
            targets=np.array([0.5]),
            side="middle",
        )


def test_interp_with_plateaus_single_point_returns_constant():
    out = miscellaneous.interp_with_plateaus(
        h_vals=np.array([2.0]),
        t_vals=np.array([95.0]),
        targets=np.array([1.0, 2.0, 3.0]),
        side="left",
    )
    assert np.allclose(out, [95.0, 95.0, 95.0])


def test_make_monotonic_size_one_is_identity():
    values = np.array([5.0])
    out = miscellaneous.make_monotonic(values, side="right")
    assert np.allclose(out, values)


def test_g_ineq_penalty_numpy_scalar_raises_for_unknown_return_type():
    class _WeirdNumber:
        def __pow__(self, exponent):
            return self

        def __rmul__(self, other):
            return self

    with pytest.raises(ValueError, match="unrecognised type"):
        miscellaneous.g_ineq_penalty(_WeirdNumber(), form="square")
