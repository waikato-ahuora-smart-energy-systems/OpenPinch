"""Regression tests for miscellaneous utility helpers."""

import math
import pytest
from OpenPinch.utils.miscellaneous import *
from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.utils import *
from OpenPinch.analysis.exergy_targeting import *
import numpy as np
from OpenPinch.utils import miscellaneous


"""Test cases for the get_value function."""


@pytest.mark.parametrize(
    ("payload", "zone_name", "val2", "expected"),
    [
        pytest.param(3.14, None, None, 3.14, id="float"),
        pytest.param(5, None, None, 5.0, id="int"),
        pytest.param("100", None, None, 100.0, id="numeric-string"),
        pytest.param({"value": 42.0}, None, None, 42.0, id="flat-dict"),
        pytest.param(
            {"value": "42.0"},
            None,
            None,
            42.0,
            id="dict-numeric-string",
        ),
        pytest.param(
            {"zone-a": {"value": 7.5}},
            "zone-a",
            None,
            7.5,
            id="zone-name-recurses-into-dict",
        ),
        pytest.param(
            {"zone-a": 11.25, "value": 99.0},
            "zone-a",
            None,
            11.25,
            id="zone-name-takes-precedence-over-value",
        ),
        pytest.param(
            {"value": {"value": 1.25}, "units": "kW"},
            None,
            None,
            1.25,
            id="dict-recurses-through-value-payload",
        ),
        pytest.param(
            ValueWithUnit(value=99.9, units="kW"),
            None,
            None,
            99.9,
            id="value-with-unit",
        ),
        pytest.param(
            {"value": 4.0, "multiplier": 2.5},
            None,
            None,
            10.0,
            id="multiplier-operator",
        ),
        pytest.param(
            {"value": 4.0, "multiply": 2.5},
            None,
            None,
            10.0,
            id="multiply-alias",
        ),
        pytest.param({"value": 4.0, "add": 2.5}, None, None, 6.5, id="add-operator"),
        pytest.param(
            {"value": 4.0, "subtract": 2.5},
            None,
            None,
            1.5,
            id="subtract-operator",
        ),
        pytest.param(
            {"value": 9.0, "divide": 3.0},
            None,
            None,
            3.0,
            id="divide-operator",
        ),
        pytest.param(
            {"value": 0.0, "divide": 3.0},
            None,
            None,
            0.0,
            id="divide-zero-numerator",
        ),
        pytest.param(
            {"value": 3.0, "power": 2.0},
            None,
            None,
            9.0,
            id="power-operator",
        ),
        pytest.param(
            {"value": 100.0, "log": 10.0},
            None,
            None,
            2.0,
            id="log-with-explicit-base",
        ),
        pytest.param(
            {"value": np.e**2, "log": {"base": "ignored"}},
            None,
            None,
            2.0,
            id="log-defaults-to-e-for-non-float-base",
        ),
        pytest.param(
            {"value": -10.0, "log": 10.0},
            None,
            None,
            0.0,
            id="log-non-positive-input",
        ),
        pytest.param({"value": 3.0, "exp": 2.0}, None, None, 8.0, id="exp-operator"),
        pytest.param(
            {"value": -1.0, "exp": 2.0},
            None,
            None,
            0.0,
            id="exp-non-positive-input",
        ),
        pytest.param(
            {"value": -4.0, "abs": True},
            None,
            None,
            4.0,
            id="abs-operator",
        ),
        pytest.param(
            {"value": ValueWithUnit(value=8.0, units="kW"), "add": 2.0},
            None,
            None,
            10.0,
            id="operator-with-valuewithunit-base",
        ),
        pytest.param(
            {"value": 10.0, "add": "2"},
            None,
            None,
            12.0,
            id="operator-with-numeric-string",
        ),
        pytest.param(
            {"value": {"value": 8.0}, "divide": {"value": 2.0}},
            None,
            None,
            4.0,
            id="nested-operator-payloads",
        ),
        pytest.param(
            {"zone-a": {"value": 2.0, "power": 3.0}},
            "zone-a",
            None,
            8.0,
            id="zone-name-with-operator-payload",
        ),
        pytest.param(
            {"units": "kW"},
            None,
            12.5,
            12.5,
            id="val2-fallback",
        ),
        pytest.param(
            {"units": "kW"},
            None,
            "12.5",
            12.5,
            id="val2-numeric-string-fallback",
        ),
    ],
)
def test_get_value_supported_inputs(payload, zone_name, val2, expected):
    assert get_value(payload, zone_name=zone_name, val2=val2) == pytest.approx(expected)


def test_get_value_does_not_mutate_payload():
    payload = {"value": 8.0, "add": 2.0}

    result = get_value(payload)

    assert result == pytest.approx(10.0)
    assert payload == {"value": 8.0, "add": 2.0}


def test_get_value_val2_fallback_does_not_mutate_payload():
    payload = {"units": "kW"}

    result = get_value(payload, val2=12.5)

    assert result == pytest.approx(12.5)
    assert payload == {"units": "kW"}


def test_get_value_none_raises_type_error():
    with pytest.raises(TypeError, match="Unsupported type"):
        get_value(None)


def test_get_value_dict_none_value_raises_type_error():
    with pytest.raises(TypeError, match="Unsupported type"):
        get_value({"value": None})


def test_get_value_zone_name_none_value_raises_type_error():
    with pytest.raises(TypeError, match="Unsupported type"):
        get_value({"zone-a": None}, zone_name="zone-a")


@pytest.mark.parametrize(
    ("payload", "zone_name", "val2", "error_type", "match"),
    [
        pytest.param(
            {"not_value": 10.0},
            None,
            None,
            KeyError,
            "value",
            id="missing-value-key",
        ),
        pytest.param(
            {"other-zone": 10.0},
            "zone-a",
            None,
            KeyError,
            "value",
            id="missing-zone-key-and-value",
        ),
        pytest.param(
            {"value": "abc", "units": "kW"},
            None,
            None,
            TypeError,
            "Unsupported string value",
            id="dict-value-with-invalid-string",
        ),
        pytest.param(
            {"value": 10.0, "add": "two"},
            None,
            None,
            TypeError,
            "Unsupported string value",
            id="operator-with-invalid-string",
        ),
        pytest.param(
            {"units": "kW"},
            None,
            "twelve point five",
            TypeError,
            "Unsupported string value",
            id="val2-invalid-string",
        ),
        pytest.param(True, None, None, TypeError, "Unsupported type", id="bool"),
        pytest.param(
            {"value": True},
            None,
            None,
            TypeError,
            "Unsupported type",
            id="dict-bool",
        ),
        pytest.param(
            {"zone-a": True},
            "zone-a",
            None,
            TypeError,
            "Unsupported type",
            id="zone-bool",
        ),
        pytest.param(None, None, None, TypeError, "Unsupported type", id="none"),
        pytest.param([1.0], None, None, TypeError, "Unsupported type", id="list"),
    ],
)
def test_get_value_invalid_inputs(payload, zone_name, val2, error_type, match):
    with pytest.raises(error_type, match=match):
        get_value(payload, val2=val2, zone_name=zone_name)


"""Test cases for the compute_capital_recovery_factor function."""


def test_crf_typical_case():
    """Test with typical values for interest and years."""
    i = 0.08
    n = 10
    result = compute_capital_recovery_factor(i, n)
    expected = i * (1 + i) ** n / ((1 + i) ** n - 1)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_crf_high_interest():
    """Test with a high interest rate."""
    result = compute_capital_recovery_factor(0.2, 5)
    assert result > 0.2


def test_crf_long_term():
    """Test for long project duration (n=30)."""
    result = compute_capital_recovery_factor(0.05, 30)
    assert result < 0.1


def test_crf_short_term():
    """Test with a short project duration (n=1)."""
    result = compute_capital_recovery_factor(0.1, 1)
    expected = 1.1
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_crf_zero_interest_raises():
    """Zero interest should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        compute_capital_recovery_factor(0.0, 10)


def test_crf_zero_years_raises():
    """Zero years should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        compute_capital_recovery_factor(0.08, 0)


def test_crf_negative_interest():
    """Negative interest should compute (though rarely used)."""
    result = compute_capital_recovery_factor(-0.01, 10)
    assert isinstance(result, float)


def test_clean_composite_removes_redundant_points():
    x_vals = [0, 10, 20, 30, 30]
    y_vals = [0, 5, 10, 15, 30]
    y_clean, x_clean = clean_composite_curve(y_vals, x_vals)
    assert np.allclose(x_clean, [0, 30])
    assert np.allclose(y_clean, [0, 15])


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


def test_g_ineq_penalty_square_default():
    assert g_ineq_penalty(2.0) == pytest.approx(40.0)


def test_g_ineq_penalty_square_custom_rho():
    assert g_ineq_penalty(-3.0, rho=2.0, form="square") == pytest.approx(18.0)


def test_g_ineq_penalty_square_root_smoothing():
    g, eta, rho = -0.2, 0.5, 4.0
    expected = 0.5 * rho * (g + (g**2 + eta**2) ** 0.5)
    assert g_ineq_penalty(
        g, eta=eta, rho=rho, form="square_root_smoothing"
    ) == pytest.approx(expected)


def test_g_ineq_penalty_invalid_form():
    with pytest.raises(ValueError, match="Unrecognised penalty function"):
        g_ineq_penalty(1.0, form="invalid")


# ===== Merged from test_miscellaneous_extra.py =====
"""Additional coverage tests for miscellaneous helpers."""


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
    figure = miscellaneous.graph_simple_cc_plot(
        Tc=[40, 30],
        Hc=[0, 10],
        Th=[120, 110],
        Hh=[0, 12],
    )
    assert shown["called"] is True
    assert len(figure.data) == 2
    assert figure.data[0].name == "Cold composite"
    assert figure.data[1].name == "Hot composite"


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
