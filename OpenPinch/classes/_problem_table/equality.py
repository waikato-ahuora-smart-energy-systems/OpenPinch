"""Equality helpers for :class:`OpenPinch.classes.problem_table.ProblemTable`."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


def tables_have_matching_columns(left, right) -> bool:
    """Return True when both tables expose the same column labels in order."""
    return left.columns == right.columns


def tables_have_matching_shapes(left, right) -> bool:
    """Return True when both table data arrays have the same shape."""
    return left.data.shape == right.data.shape


def numeric_arrays_equal(left, right, atol: float) -> bool:
    """Return True when two numeric arrays match within absolute tolerance."""
    return np.allclose(left, right, atol=atol, rtol=0.0, equal_nan=True)


def try_cast_object_column_to_float(column):
    """Return a float array for numeric-like object columns, otherwise None."""
    try:
        return column.astype(float)
    except ValueError, TypeError:
        return None


def scalar_values_equal(left, right, atol: float) -> bool:
    """Return True when two scalar cell values match under table equality rules."""
    if pd.isna(left) and pd.isna(right):
        return True

    numeric_types = (numbers.Real, np.number)
    if isinstance(left, numeric_types) and isinstance(right, numeric_types):
        return bool(np.isclose(left, right, atol=atol))

    return left == right


def object_columns_equal(left, right, atol: float) -> bool:
    """Return True when two possibly object-typed columns are value-equivalent."""
    if left.dtype != object and right.dtype != object:
        return numeric_arrays_equal(left, right, atol)

    cast_left = try_cast_object_column_to_float(left)
    cast_right = try_cast_object_column_to_float(right)
    if cast_left is not None and cast_right is not None:
        return numeric_arrays_equal(cast_left, cast_right, atol)

    return all(
        scalar_values_equal(value_left, value_right, atol)
        for value_left, value_right in zip(left, right)
    )


def arrays_are_empty(left) -> bool:
    """Return True when the table array has no cells to compare."""
    return left.size == 0


def table_data_presence_matches(left, right) -> bool:
    """Return True when both tables either have data arrays or both have none."""
    return left.data is None and right.data is None


def table_arrays_equal(left, right, atol: float) -> bool:
    """Return True when two table data arrays are equal under table rules."""
    left_array = left.data
    right_array = right.data

    if arrays_are_empty(left_array):
        return True

    if left_array.dtype != object and right_array.dtype != object:
        return numeric_arrays_equal(left_array, right_array, atol)

    return all(
        object_columns_equal(left_array[:, col_idx], right_array[:, col_idx], atol)
        for col_idx in range(left_array.shape[1])
    )


def problem_tables_equal(
    left, right, *, table_type, default_atol: float, atol=None
) -> bool:
    """Return True when two ProblemTable-like objects are equal."""
    if not isinstance(right, table_type):
        return False

    if not tables_have_matching_columns(left, right):
        return False

    if left.data is None or right.data is None:
        return table_data_presence_matches(left, right)

    if not tables_have_matching_shapes(left, right):
        return False

    effective_atol = default_atol if atol is None else atol
    return table_arrays_equal(left, right, effective_atol)
