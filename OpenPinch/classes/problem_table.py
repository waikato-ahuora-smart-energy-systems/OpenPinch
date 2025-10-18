"""Lightweight table structure used by the pinch analysis pipeline."""

from __future__ import annotations

import numbers
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from ..lib.config import tol
from ..lib.enums import ProblemTableLabel

PT = ProblemTableLabel


class ProblemTable:
    """NumPy-backed representation of the pinch problem table with pandas-like accessors."""

    _DEFAULT_ATOL = 1e-6

    def __init__(self, data_input: dict | list = None, add_default_labels: bool = True):
        """Initialise the table from a dictionary (column keyed) or list-of-columns structure."""
        if add_default_labels:
            self.columns = list([index.value for index in ProblemTableLabel])
        else:
            self.columns = list([key for key in data_input.keys()])

            for key in self.columns:
                if np.isnan(data_input[key]).all():
                    data_input.pop(key)
        self.col_index = {col: idx for idx, col in enumerate(self.columns)}

        if isinstance(data_input, dict):
            # Align data from dict into array using columns order
            self.data = np.array(
                [
                    data_input.get(col, [np.nan] * len(next(iter(data_input.values()))))
                    for col in self.columns
                ]
            ).T
        elif isinstance(data_input, list):
            data_input = self._pad_data_input(data_input, len(self.columns))
            self.data = np.array(data_input).T
        else:
            self.data = None

    class ColumnViewByIndex:
        """Expose read/write access to columns addressed by integer index."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, idx):
            return self.parent.data[:, idx]

        def __setitem__(self, idx, values):
            self.parent.data[:, idx] = values

    @property
    def icol(self):
        """Return a view for column access by integer position."""
        return self.ColumnViewByIndex(self)

    class ColumnViewByName:
        """Expose read/write access to columns addressed by column label."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, col_name):
            idx = self.parent.col_index[col_name]
            return self.parent.data[:, idx]

        def __setitem__(self, col_name, values):
            idx = self.parent.col_index[col_name]
            if self.parent.data is not None:
                self.parent.data[:, idx] = values
            else:
                data_input = {col_name: values}
                self.data = np.array(
                    [
                        data_input.get(
                            col, [np.nan] * len(next(iter(data_input.values())))
                        )
                        for col in self.parent.columns
                    ]
                ).T

    @property
    def col(self):
        """Return a view for column access by label."""
        return self.ColumnViewByName(self)

    class ColumnsViewByName:
        """Vectorised view over multiple labelled columns."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, col_names):
            idxs = []
            for col_name in col_names:
                idxs.append(self.parent.col_index[col_name])
            return self.parent.data[:, idxs]

        def __setitem__(self, col_name, values):
            idx = self.parent.col_index[col_name]
            if self.parent.data is not None:
                self.parent.data[:, idx] = values
            else:
                data_input = {col_name: values}
                self.data = np.array(
                    [
                        data_input.get(
                            col, [np.nan] * len(next(iter(data_input.values())))
                        )
                        for col in self.parent.columns
                    ]
                ).T

    @property
    def cols(self):
        """Return a vectorised view that reads multiple labelled columns."""
        return self.ColumnsViewByName(self)

    class LocationByRowByColName:
        """Row/column accessor mirroring ``DataFrame.loc`` semantics."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, key):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            return self.parent.data[row_idx, col_idx]

        def __setitem__(self, key, value):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            self.parent.data[row_idx, col_idx] = value

    @property
    def loc(self):
        """Expose row/column access using label semantics (``loc``)."""
        return self.LocationByRowByColName(self)

    class LocationByRowByCol:
        """Row/column accessor mirroring ``DataFrame.iloc`` semantics."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, key):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            return self.parent.data[row_idx, col_idx]

        def __setitem__(self, key, value):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            self.parent.data[row_idx, col_idx] = value

    @property
    def iloc(self):
        """Expose row/column access using positional semantics (``iloc``)."""
        return self.LocationByRowByCol(self)

    def __len__(self):
        """Return the number of rows stored in the table."""
        if isinstance(self.data, np.ndarray):
            return self.data.shape[0]
        else:
            return 0

    def __getitem__(self, keys):
        """Extract a subset of columns and return them as a new ``ProblemTable``."""
        data_input = {}
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            data_input[key] = self.col[key]
        return ProblemTable(data_input, add_default_labels=False)

    def _equals(self, other: "ProblemTable", *, atol: float | None = None) -> bool:
        """Return True when two tables match within ``atol`` absolute tolerance."""
        if not isinstance(other, ProblemTable):
            return False

        if self.columns != other.columns:
            return False

        if self.data is None or other.data is None:
            return self.data is None and other.data is None

        if self.data.shape != other.data.shape:
            return False

        atol = self._DEFAULT_ATOL if atol is None else atol
        left = self.data
        right = other.data

        if left.size == 0:
            return True

        if left.dtype != object and right.dtype != object:
            return np.allclose(left, right, atol=atol, rtol=0.0, equal_nan=True)

        numeric_types = (numbers.Real, np.number)
        for col_idx in range(left.shape[1]):
            col_left = left[:, col_idx]
            col_right = right[:, col_idx]

            if col_left.dtype != object and col_right.dtype != object:
                if not np.allclose(
                    col_left, col_right, atol=atol, rtol=0.0, equal_nan=True
                ):
                    return False
                continue

            try:
                cast_left = col_left.astype(float)
                cast_right = col_right.astype(float)
            except (ValueError, TypeError):
                pass
            else:
                if not np.allclose(
                    cast_left, cast_right, atol=atol, rtol=0.0, equal_nan=True
                ):
                    return False
                continue

            for value_left, value_right in zip(col_left, col_right):
                if pd.isna(value_left) and pd.isna(value_right):
                    continue
                if isinstance(value_left, numeric_types) and isinstance(
                    value_right, numeric_types
                ):
                    if not np.isclose(value_left, value_right, atol=atol):
                        return False
                    continue
                if value_left != value_right:
                    return False

        return True

    def __eq__(self, other):
        """Return ``True`` when two tables hold identical values."""
        return self._equals(other)

    def __ne__(self, other):
        """Return ``True`` when two tables differ."""
        return not self.__eq__(other)

    @property
    def shape(self):
        """Tuple describing ``(rows, columns)`` for the buffer."""
        return self.data.shape

    @property
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the buffer into a pandas DataFrame."""
        return pd.DataFrame(self.data.copy, columns=self.columns)

    @property
    def copy(self):
        """Return a deep copy of the table."""
        return deepcopy(self)

    def _pad_data_input(self, data_input, n_cols):
        """Pad a list-of-columns input so it matches ``n_cols`` length."""
        current_cols = len(data_input)
        if current_cols < n_cols:
            n_rows = len(data_input[0])  # assume all rows are same length
            padding = [[np.nan] * n_rows for _ in range(n_cols - current_cols)]
            data_input += padding
        return data_input

    def to_list(self, col: str = None):
        """Return table data as Python lists; optionally restrict to a single column."""
        if isinstance(col, str):
            ls = self.col[col].T.tolist()
        elif col == None:
            ls = self.data.T.tolist()
        return ls[0] if len(ls) == 1 else ls

    def delta_col(self, key, shift: int = 1) -> np.ndarray:
        """Compute difference between successive entries in a column."""
        idx = self.col_index[key]
        col_values = self.data[:, idx]
        delta = np.roll(col_values, shift) - col_values
        delta[0] = 0.0          
        return delta

    def shift(self, key, shift: int = 1, filler_value: float = 0.0) -> np.ndarray:
        """Return a shifted copy of a column filling vacated positions with ``filler_value``."""
        idx = self.col_index[key]
        col_values = self.data[:, idx]
        values = np.roll(col_values, shift)
        if shift > 0:
            for i in range(shift):
                values[i] = filler_value
        elif shift < 0:
            for i in range(shift, 0):
                values[i] = filler_value
        return values

    def round(self, decimals):
        """Round the underlying NumPy buffer in-place."""
        self.data = np.round(self.data, decimals)

    @staticmethod
    def linear_interpolation(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
        """Perform linear interpolation to estimate ``y`` at ``x``."""
        if x1 == x2:
            raise ValueError("Cannot perform interpolation when x1 == x2 (undefined slope).")
        m = (y1 - y2) / (x1 - x2)
        c = y1 - m * x1
        return m * x + c

    def get_pinch_loc(
        self, col: Union[int, str, ProblemTableLabel] = PT.H_NET.value
    ) -> Tuple[int, int, bool]:
        """Return the row indices of the hot and cold pinch temperatures."""
        column = col.value if isinstance(col, ProblemTableLabel) else col
        h_net = (
            np.asarray(self.col[column]) if isinstance(column, str) else np.asarray(self.icol[column])
        )
        n = h_net.size

        abs_arr = np.abs(h_net)
        zeros_mask = abs_arr < tol

        has_zero = np.any(zeros_mask)
        all_zero = np.all(zeros_mask)

        if has_zero and not all_zero:
            first_zero = np.flatnonzero(zeros_mask)[0]
            if first_zero > 0:
                row_h = first_zero
            else:
                nz_after = np.flatnonzero(~zeros_mask)
                row_h = nz_after[0] - 1 if nz_after.size else n - 1

            last_zero = np.flatnonzero(zeros_mask)[-1]
            if last_zero < n - 1:
                row_c = last_zero
            else:
                nz_before_rev = np.flatnonzero(~zeros_mask[::-1])
                row_c = n - nz_before_rev[0] if nz_before_rev.size else 0
        else:
            row_h = n - 1
            row_c = 0

        valid = row_h <= row_c
        return row_h, row_c, valid

    def get_pinch_temperatures(
        self,
        col_T: str = PT.T.value,
        col_H: Union[int, str, ProblemTableLabel] = PT.H_NET.value,
    ) -> Tuple[float | None, float | None]:
        """Determine the hottest hot and coldest cold pinch temperatures."""
        hot_idx, cold_idx, valid = self.get_pinch_loc(col_H)
        if valid:
            return self.loc[hot_idx, col_T], self.loc[cold_idx, col_T]
        return None, None

    def shift_heat_cascade(
        self, dh: float, col: Union[int, str, ProblemTableLabel]
    ) -> "ProblemTable":
        """Shift a column in the heat cascade by ``dh`` and return a copy of the table."""
        if isinstance(col, ProblemTableLabel):
            target = col.value
            self.col[target] += dh
        elif isinstance(col, str):
            self.col[col] += dh
        else:
            self.icol[col] += dh
        return self.copy

    def insert_temperature_interval(
        self, T_ls: List[float] | float
    ) -> Tuple["ProblemTable", int]:
        """Insert temperature intervals assuming a strictly descending ``T`` column."""
        values = [T_ls] if isinstance(T_ls, float) else T_ls

        for T_new in values:
            col = self.col_index
            T_col = self.data[:, col[PT.T.value]]

            insert_index = None
            for i in range(1, len(T_col)):
                if T_col[i - 1] - tol > T_new > T_col[i] + tol:
                    insert_index = i
                    break

            if insert_index is None:
                return self, 0

            row_top = self.data[insert_index - 1]
            row_bot = self.data[insert_index]

            cp_hot = row_bot[col[PT.CP_HOT.value]]
            cp_cold = row_bot[col[PT.CP_COLD.value]]
            mcp_net = row_bot[col[PT.MCP_NET.value]]

            delta_above = row_top[col[PT.T.value]] - T_new
            delta_below = T_new - row_bot[col[PT.T.value]]

            row_dict = {
                PT.T.value: T_new,
                PT.DELTA_T.value: delta_above,
                PT.CP_HOT.value: cp_hot,
                PT.DELTA_H_HOT.value: delta_above * cp_hot,
                PT.CP_COLD.value: cp_cold,
                PT.DELTA_H_COLD.value: delta_above * cp_cold,
                PT.MCP_NET.value: mcp_net,
                PT.DELTA_H_NET.value: delta_above * mcp_net,
            }

            icol_T = col[PT.T.value]
            for key in [
                PT.H_HOT.value,
                PT.H_COLD.value,
                PT.H_NET.value,
                PT.H_NET_NP.value,
                PT.H_NET_A.value,
                PT.H_NET_V.value,
            ]:
                idx = col[key]
                if not np.isnan(row_bot[idx]):
                    row_dict[key] = self.linear_interpolation(
                        T_new, row_bot[icol_T], row_top[icol_T], row_bot[idx], row_top[idx]
                    )

            self.insert(row_dict, insert_index)

            self.data[insert_index + 1, col[PT.DELTA_T.value]] = delta_below
            self.data[insert_index + 1, col[PT.DELTA_H_HOT.value]] = delta_below * cp_hot
            self.data[insert_index + 1, col[PT.DELTA_H_COLD.value]] = delta_below * cp_cold
            self.data[insert_index + 1, col[PT.DELTA_H_NET.value]] = delta_below * mcp_net

        return self, 1

    def insert(self, row_dict: dict, index: int):
        """Insert a single row (dict of column: value) at the specified index."""
        new_row = np.full(self.data.shape[1], np.nan)
        for key, value in row_dict.items():
            # if key in self.col_index:
            new_row[self.col_index[key]] = value
        self.data = np.insert(self.data, index, new_row, axis=0)

    def update_row(self, index: int, row_dict: dict):
        """Update selected columns for the row at ``index`` using values from ``row_dict``."""
        for key, value in row_dict.items():
            if key in self.col_index:
                self.data[index, self.col_index[key]] = value

    def delete_row(self, index: int):
        """Remove a row at ``index`` from the buffer."""
        self.data = np.delete(self.data, index, axis=0)

    def sort_by_column(self, column: str, ascending: bool = True):
        """Sort rows in-place by the given column."""
        if column not in self.col_index:
            raise KeyError(f"Column {column} not found")
        col_data = self.data[:, self.col_index[column]]
        order = np.argsort(col_data)
        if not ascending:
            order = order[::-1]
        self.data = self.data[order]
