import pytest

from OpenPinch.utils import linear_interpolation

"""Test cases for linear_interpolation function."""


def test_linear_interpolation_midpoint():
    assert linear_interpolation(5, 0, 10, 0, 100) == 50


def test_linear_interpolation_negative_slope():
    assert linear_interpolation(2, 0, 4, 100, 0) == 50


def test_linear_interpolation_flat_line():
    assert linear_interpolation(5, 0, 10, 20, 20) == 20


def test_linear_interpolation_returns_y1_at_x1():
    assert linear_interpolation(0, 0, 10, 5, 15) == 5


def test_linear_interpolation_returns_y2_at_x2():
    assert linear_interpolation(10, 0, 10, 5, 15) == 15


def test_linear_interpolation_same_x_raises():
    with pytest.raises(ValueError, match="x1 == x2"):
        linear_interpolation(7, 3, 3, 100, 200)