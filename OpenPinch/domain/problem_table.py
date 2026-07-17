"""Lightweight table structure used by the pinch analysis pipeline."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np

from ._problem_table.equality import (
    problem_tables_equal,
)
from ._problem_table.intervals import insert_temperature_intervals
from ._problem_table.types import ProblemTableColumnUpdates
from .configuration import tol
from .enums import ProblemTableLabel

PT = ProblemTableLabel


class ProblemTable:
    """NumPy-backed pinch problem table with enum-friendly accessors."""

    _DEFAULT_ATOL = 1e-6

    def __init__(
        self,
        data_input: dict[str | ProblemTableLabel, object] | list | None = None,
        add_default_labels: bool = True,
    ):
        """Initialise the table from a dictionary or list-of-columns structure."""
        if isinstance(data_input, dict):
            data_input = self._validate_column_mapping(data_input)

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

    @staticmethod
    def _validate_column_name(col_name: str | ProblemTableLabel) -> str:
        """Return the canonical string label for a supported column key."""
        if isinstance(col_name, ProblemTableLabel):
            return col_name.value
        if isinstance(col_name, str):
            return col_name
        raise TypeError("Column labels must be strings or ProblemTableLabel values.")

    @classmethod
    def _validate_column_names(
        cls, col_names: str | ProblemTableLabel | Sequence[str | ProblemTableLabel]
    ) -> list[str]:
        """Return canonical string labels for one or more supported column keys."""
        if isinstance(col_names, (str, ProblemTableLabel)):
            return [cls._validate_column_name(col_names)]
        if not isinstance(col_names, Sequence):
            raise TypeError(
                "Column labels must be strings, ProblemTableLabel values, or "
                "sequences of those values."
            )
        return [cls._validate_column_name(col_name) for col_name in col_names]

    @classmethod
    def _validate_column_mapping(
        cls, data_input: dict[str | ProblemTableLabel, object]
    ) -> dict[str, object]:
        """Return a copy of ``data_input`` with any enum keys normalised to strings."""
        validated: dict[str, object] = {}
        for key, values in data_input.items():
            col_name = cls._validate_column_name(key)
            if col_name in validated:
                raise ValueError(
                    f"Duplicate column label {col_name!r} found after "
                    "key normalisation."
                )
            validated[col_name] = values
        return validated

    def _initialise_named_column(self, col_name: str, values) -> None:
        """Initialise the table data from a single named column mapping."""
        data_input = {col_name: values}
        self.data = np.array(
            [
                data_input.get(col, [np.nan] * len(next(iter(data_input.values()))))
                for col in self.columns
            ]
        ).T

    def _column_index_for(self, col_name: str | ProblemTableLabel) -> int:
        """Return the integer column index for a string label or ProblemTableLabel."""
        return self.col_index[self._validate_column_name(col_name)]

    def _get_column_by_name(self, col_name: str | ProblemTableLabel):
        """Return a raw NumPy column view for the given label."""
        idx = self._column_index_for(col_name)
        return self.data[:, idx]

    def _set_column_by_name(self, col_name: str | ProblemTableLabel, values) -> None:
        """Assign values to a named column, initialising the table if needed."""
        col_name = self._validate_column_name(col_name)
        idx = self._column_index_for(col_name)
        if self.data is not None:
            self.data[:, idx] = values
        else:
            self._initialise_named_column(col_name, values)

    def _slice_columns(
        self, keys: str | ProblemTableLabel | Sequence[str | ProblemTableLabel]
    ) -> "ProblemTable":
        """Build a new ProblemTable containing only the requested columns."""
        data_input = {}
        for key in self._validate_column_names(keys):
            data_input[key] = self[key]
        return ProblemTable(data_input, add_default_labels=False)

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
        """Expose read/write access to columns addressed by label or enum."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, col_name):
            return self.parent._get_column_by_name(col_name)

        def __setitem__(self, col_name, values):
            self.parent._set_column_by_name(col_name, values)

    @property
    def col(self):
        """Return a view for column access by string label or ProblemTableLabel."""
        return self.ColumnViewByName(self)

    class ColumnsViewByName:
        """Vectorised view over multiple labelled columns or enums."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, col_names):
            idxs = []
            for col_name in self.parent._validate_column_names(col_names):
                idxs.append(self.parent.col_index[col_name])
            return self.parent.data[:, idxs]

        def __setitem__(self, col_name, values):
            self.parent._set_column_by_name(col_name, values)

    @property
    def cols(self):
        """Return a vectorised view over multiple labelled columns or enums."""
        return self.ColumnsViewByName(self)

    class LocationByRowByColName:
        """Row/column accessor mirroring ``DataFrame.loc`` semantics."""

        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, key):
            row_idx, col_key = key
            col_key = self.parent._validate_column_name(col_key)
            col_idx = self.parent._column_index_for(col_key)
            return self.parent.data[row_idx, col_idx]

        def __setitem__(self, key, value):
            row_idx, col_key = key
            col_key = self.parent._validate_column_name(col_key)
            col_idx = self.parent._column_index_for(col_key)
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
            if isinstance(col_key, numbers.Integral):
                col_idx = int(col_key)
            else:
                col_idx = self.parent._column_index_for(col_key)
            return self.parent.data[row_idx, col_idx]

        def __setitem__(self, key, value):
            row_idx, col_key = key
            if isinstance(col_key, numbers.Integral):
                col_idx = int(col_key)
            else:
                col_idx = self.parent._column_index_for(col_key)
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

    def __getitem__(self, key: str | ProblemTableLabel):
        """Return a raw NumPy column view.

        Use ``slice(...)`` for subtable extraction.
        """
        if isinstance(key, Sequence) and not isinstance(key, (str, ProblemTableLabel)):
            raise TypeError(
                "ProblemTable[...] only supports single-column access. "
                "Use `pt.slice([...])` for subtable extraction."
            )
        return self._get_column_by_name(key)

    def __setitem__(self, key: str | ProblemTableLabel, values) -> None:
        """Assign values to a single column by string label or ProblemTableLabel."""
        if isinstance(key, Sequence) and not isinstance(key, (str, ProblemTableLabel)):
            raise TypeError(
                "ProblemTable[...] only supports single-column assignment. "
                "Use `pt.slice([...])` for subtable extraction."
            )
        self._set_column_by_name(key, values)

    def slice(
        self, keys: str | ProblemTableLabel | Sequence[str | ProblemTableLabel]
    ) -> "ProblemTable":
        """Return a new ProblemTable containing only the requested columns."""
        return self._slice_columns(keys)

    def _equals(self, other: "ProblemTable", *, atol: float | None = None) -> bool:
        """Return True when two tables match within ``atol`` absolute tolerance."""
        return problem_tables_equal(
            self,
            other,
            table_type=ProblemTable,
            default_atol=self._DEFAULT_ATOL,
            atol=atol,
        )

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

    def to_list(self, col: str | ProblemTableLabel | None = None):
        """Return table data as Python lists; optionally restrict to a single column."""
        if col is not None:
            ls = self[col].T.tolist()
        elif col is None:
            ls = self.data.T.tolist()
        return ls[0] if len(ls) == 1 else ls

    def round(self, decimals):
        """Round the underlying NumPy buffer in-place."""
        self.data = np.round(self.data, decimals)

    def pinch_idx(
        self, col: Union[int, str, ProblemTableLabel] = PT.H_NET
    ) -> Tuple[int, int, bool]:
        """Return the row indices of the hot and cold pinch temperatures."""
        if isinstance(col, int):
            h_net = np.asarray(self.icol[col])
        else:
            h_net = np.asarray(self[col])
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

    def pinch_temperatures(
        self,
        col_T: str | ProblemTableLabel = PT.T,
        col_H: Union[int, str, ProblemTableLabel] = PT.H_NET,
    ) -> Tuple[float | None, float | None]:
        """Determine the hottest hot and coldest cold pinch temperatures."""
        hot_idx, cold_idx, valid = self.pinch_idx(col_H)
        if valid:
            return self.loc[hot_idx, col_T], self.loc[cold_idx, col_T]
        return None, None

    def shift_heat_cascade(
        self, dh: float, col: Union[int, str, ProblemTableLabel]
    ) -> "ProblemTable":
        """Shift a heat-cascade column by ``dh`` and return a table copy."""
        if isinstance(col, (ProblemTableLabel, str)):
            self[col] += dh
        else:
            self.icol[col] += dh
        return self.copy

    def share_temperature_intervals(self, other: "ProblemTable") -> Tuple[int, int]:
        """Mutate both tables so they use the union of their temperature intervals.

        Returns a tuple containing
        ``(rows_inserted_into_self, rows_inserted_into_other)``.
        """
        if not isinstance(other, ProblemTable):
            raise TypeError("`other` must be a ProblemTable instance.")

        inserted_self = self.insert_temperature_interval(other[PT.T].tolist())
        inserted_other = other.insert_temperature_interval(self[PT.T].tolist())
        return inserted_self, inserted_other

    def insert_temperature_interval(self, T_ls: List[float] | float) -> int:
        """Insert any missing temperature intervals and return count inserted."""
        return insert_temperature_intervals(self, T_ls)

    def insert(self, row_dict: dict, index: int):
        """Insert a single row (dict of column: value) at the specified index."""
        new_row = np.full(self.data.shape[1], np.nan)
        for key, value in row_dict.items():
            col_name = self._validate_column_name(key)
            new_row[self.col_index[col_name]] = value
        self.data = np.insert(self.data, index, new_row, axis=0)

    def update_row(self, index: int, row_dict: dict):
        """Update selected columns for one row using values from ``row_dict``."""
        for key, value in row_dict.items():
            col_name = self._validate_column_name(key)
            if col_name in self.col_index:
                self.data[index, self.col_index[col_name]] = value

    def _validate_T_col(self, T_col: np.ndarray | None) -> np.ndarray:
        """Validate and cast the source temperature column used to align updates."""
        if T_col is None:
            raise TypeError("`T_col` is required when updates are provided.")
        if not isinstance(T_col, np.ndarray):
            raise TypeError("`T_col` must be a 1D numpy.ndarray.")
        if T_col.ndim != 1:
            raise ValueError("`T_col` must be a 1D numpy.ndarray.")
        try:
            return T_col.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "`T_col` must contain numeric temperature values."
            ) from exc

    def _validate_updates(
        self, updates: ProblemTableColumnUpdates, T_col: np.ndarray
    ) -> ProblemTableColumnUpdates:
        """Validate update data and normalise keys to canonical column names."""
        if not isinstance(updates, dict):
            raise TypeError("`updates` must be a dictionary of ProblemTable columns.")

        expected_len = T_col.shape[0]
        normalised: ProblemTableColumnUpdates = {}

        for key, values in updates.items():
            col_name = self._validate_column_name(key)
            if col_name == self._validate_column_name(PT.T):
                raise ValueError(
                    "`ProblemTable.update()` does not accept updates to the "
                    "temperature column. Use interval helpers or construct a "
                    "new ProblemTable instead."
                )
            if col_name not in self.col_index:
                raise KeyError(f"Column {col_name} not found")
            if not isinstance(values, np.ndarray):
                raise TypeError(
                    f"Update for column {col_name} must be a 1D numpy.ndarray."
                )
            if values.ndim != 1:
                raise ValueError(
                    f"Update for column {col_name} must be a 1D numpy.ndarray."
                )
            if values.shape[0] != expected_len:
                raise ValueError(
                    f"Update for column {col_name} has length {values.shape[0]} but "
                    f"`T_col` has length {expected_len}."
                )
            normalised[col_name] = values

        return normalised

    def update(
        self,
        updates: ProblemTableColumnUpdates | None = None,
        T_col: np.ndarray | None = None,
    ) -> "ProblemTable":
        """Assign aligned column values in-place using an explicit source T column."""
        if not updates:
            return self
        if self.data is None:
            raise ValueError("Cannot update columns on an uninitialised ProblemTable.")

        T_col = self._validate_T_col(T_col)
        updates = self._validate_updates(updates, T_col)
        target_temperatures = np.asarray(self[PT.T], dtype=float)

        if target_temperatures.shape != T_col.shape or not np.allclose(
            a=target_temperatures,
            b=T_col,
            atol=tol,
            rtol=tol,
        ):
            source_pt = ProblemTable({PT.T: T_col, **updates})
            self.share_temperature_intervals(source_pt)
            for col_name in updates:
                updates[col_name] = source_pt[col_name]

        for col_name, values in updates.items():
            self[col_name] = values

        return self

    def delete_row(self, index: int):
        """Remove a row at ``index`` from the buffer."""
        self.data = np.delete(self.data, index, axis=0)

    def sort_by_column(self, column: str | ProblemTableLabel, ascending: bool = True):
        """Sort rows in-place by the given column."""
        column = self._validate_column_name(column)
        if column not in self.col_index:
            raise KeyError(f"Column {column} not found")
        col_data = self.data[:, self.col_index[column]]
        order = np.argsort(col_data)
        if not ascending:
            order = order[::-1]
        self.data = self.data[order]
