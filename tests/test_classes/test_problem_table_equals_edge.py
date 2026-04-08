"""Targeted edge-path tests for ProblemTable equality internals."""

import numpy as np

from OpenPinch.classes.problem_table import ProblemTable


class _FakeArray:
    def __init__(self, arr: np.ndarray, fake_dtype):
        self._arr = arr
        self.dtype = fake_dtype
        self.shape = arr.shape
        self.size = arr.size

    def __getitem__(self, key):
        return self._arr[key]


def test_equals_uses_numeric_allclose_fast_path():
    left = ProblemTable({"a": [1.0, 2.0], "b": [3.0, 4.0]}, add_default_labels=False)
    right = ProblemTable({"a": [1.0, 2.0], "b": [3.0, 4.0]}, add_default_labels=False)
    assert left._equals(right) is True


def test_equals_columnwise_allclose_branch_with_fake_object_dtype():
    left = ProblemTable({"a": [1.0, 2.0], "b": [3.0, 4.0]}, add_default_labels=False)
    right = ProblemTable({"a": [1.0, 20.0], "b": [3.0, 4.0]}, add_default_labels=False)

    left.data = _FakeArray(np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=float), object)
    right.data = _FakeArray(np.asarray([[1.0, 3.0], [20.0, 4.0]], dtype=float), object)

    assert left._equals(right) is False


def test_equals_object_nan_and_numeric_mismatch_branch():
    left = ProblemTable({"a": [0.0, 0.0, 0.0]}, add_default_labels=False)
    right = ProblemTable({"a": [0.0, 0.0, 0.0]}, add_default_labels=False)
    left.data = np.array([[np.nan], [1.0], ["x"]], dtype=object)
    right.data = np.array([[np.nan], [2.0], ["x"]], dtype=object)

    assert left._equals(right) is False
