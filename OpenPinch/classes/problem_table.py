"""Lightweight table structure used by the pinch analysis pipeline."""

from __future__ import annotations

import numbers
from copy import deepcopy
from typing import List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..lib.config import tol
from ..lib.enums import ProblemTableLabel
from ..utils import *

PT = ProblemTableLabel
INTERVAL_CORE_KEYS = (
    PT.T.value,
    PT.DELTA_T.value,
    PT.CP_HOT.value,
    PT.DELTA_H_HOT.value,
    PT.CP_COLD.value,
    PT.DELTA_H_COLD.value,
    PT.MCP_NET.value,
    PT.DELTA_H_NET.value,
    PT.RCP_HOT.value,    
    PT.RCP_COLD.value,
)
INTERPOLATION_KEYS = (
    PT.H_HOT.value,
    PT.H_COLD.value,
    PT.H_NET.value,
    PT.H_NET_NP.value,
    PT.H_NET_A.value,
    PT.H_NET_V.value,
)


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
        return pd.DataFrame(self.data.copy(), columns=self.columns)

    @property
    def copy(self):
        """Return a deep copy of the table."""
        return deepcopy(self)
    
    # TODO: create a automated series of properties and setters from the PT enum key list


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

    def round(self, decimals):
        """Round the underlying NumPy buffer in-place."""
        self.data = np.round(self.data, decimals)

    def pinch_idx(
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

    def pinch_temperatures(
        self,
        col_T: str = PT.T.value,
        col_H: Union[int, str, ProblemTableLabel] = PT.H_NET.value,
    ) -> Tuple[float | None, float | None]:
        """Determine the hottest hot and coldest cold pinch temperatures."""
        hot_idx, cold_idx, valid = self.pinch_idx(col_H)
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
    ) -> int:
        """Insert any missing temperature intervals and return count inserted."""
        if self.data is None or self.data.shape[0] < 2:
            return 0
        T_vals = np.atleast_1d(np.asarray(T_ls, dtype=float))
        T_vals = self._temps_needing_insertion(T_vals)        
        top_temps, interval_map, bottom_temps = self._categorise_insertion_targets(T_vals)
        if top_temps.size == 0 and bottom_temps.size == 0 and not interval_map:
            return 0
        new_data, inserted = self._apply_interval_map(interval_map, top_temps, bottom_temps)
        self.data = new_data
        return inserted

    def _temps_needing_insertion(self, T_vals: np.ndarray) -> np.ndarray:
        """Filter temperatures that are not within tolerance of existing rows."""
        if T_vals.size == 0:
            return T_vals
        T_col = self.data[:, self.col_index[PT.T.value]]
        gaps = np.abs(T_col[:, None] - T_vals[None, :])
        mask = np.nanmin(gaps, axis=0) > tol
        return T_vals[mask]

    def _categorise_insertion_targets(
        self, T_vals: np.ndarray
    ) -> Tuple[np.ndarray, dict[int, List[float]], np.ndarray]:
        """Split candidate temperatures into top, middle, and bottom insertions."""
        if T_vals.size == 0:
            return np.array([], dtype=float), {}, np.array([], dtype=float)
        data = self.data
        T_col = data[:, self.col_index[PT.T.value]]
        top_mask = T_vals > T_col[0]
        bottom_mask = T_vals < T_col[-1]
        middle_mask = ~(top_mask | bottom_mask)

        top_temps = self._dedupe_monotonic(np.sort(T_vals[top_mask])[::-1])
        bottom_temps = self._dedupe_monotonic(np.sort(T_vals[bottom_mask])[::-1])

        mid_temps = T_vals[middle_mask]
        mid_idx = np.empty(0, dtype=int)
        if mid_temps.size:
            idx = np.searchsorted(-T_col, -mid_temps, side="left")
            upper = T_col[idx - 1]
            lower = T_col[idx]
            inside = (upper - tol > mid_temps) & (mid_temps > lower + tol)
            mid_temps = mid_temps[inside]
            mid_idx = idx[inside].astype(int)
        interval_map = self._group_middle_inserts(mid_idx, mid_temps)
        return top_temps, interval_map, bottom_temps

    def _dedupe_monotonic(self, values: np.ndarray) -> np.ndarray:
        """Remove near-duplicate values from a monotonic sequence."""
        if values.size == 0:
            return values
        kept = [values[0]]
        for val in values[1:]:
            if abs(kept[-1] - val) > tol:
                kept.append(val)
        return np.asarray(kept, dtype=float)

    def _group_middle_inserts(
        self, idx: np.ndarray, T_vals: np.ndarray
    ) -> dict[int, List[float]]:
        """Group candidate temperatures between existing rows."""
        if T_vals.size == 0:
            return {}
        order = np.lexsort((-T_vals, idx))
        grouped: dict[int, List[float]] = {}
        for i in order:
            key = int(idx[i])
            bucket = grouped.setdefault(key, [])
            if bucket and abs(bucket[-1] - T_vals[i]) <= tol:
                continue
            bucket.append(float(T_vals[i]))
        return grouped

    def _apply_interval_map(
        self,
        interval_map: dict[int, List[float]],
        top_temps: np.ndarray,
        bottom_temps: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Insert grouped temperatures (top, middle, bottom) and return expanded array."""
        n_rows, n_cols = self.data.shape
        total_new = (
            top_temps.size
            + bottom_temps.size
            + sum(len(vals) for vals in interval_map.values())
        )
        new_data = np.zeros((n_rows + total_new, n_cols), dtype=self.data.dtype)

        row_adjustments: dict[int, np.ndarray] = {}
        out_idx = 0
        inserted_total = 0

        if top_temps.size:
            top_block, neighbor_adjusted = self._build_top_or_bottom_block(self.data[0], top_temps)
            count_top = top_block.shape[0]
            new_data[out_idx : out_idx + count_top] = top_block
            out_idx += count_top
            inserted_total += count_top
            row_adjustments[0] = neighbor_adjusted

        for row_idx in range(n_rows):
            row_current = row_adjustments.pop(row_idx, self.data[row_idx]).copy()
            new_data[out_idx] = row_current
            out_idx += 1

            next_idx = row_idx + 1
            if next_idx not in interval_map:
                continue

            temps_mid = np.asarray(interval_map[next_idx], dtype=float)
            row_bot_orig = self.data[next_idx]
            row_bot_adjusted = row_adjustments.get(next_idx, row_bot_orig).copy()

            block, row_bot_adjusted = self._build_insert_block(row_current, row_bot_orig, temps_mid)
            count_mid = block.shape[0]
            if count_mid:
                new_data[out_idx : out_idx + count_mid] = block
                out_idx += count_mid
                inserted_total += count_mid
            row_adjustments[next_idx] = row_bot_adjusted

        for idx_adj, adjusted_row in sorted(row_adjustments.items()):
            new_data[out_idx] = adjusted_row.copy()
            out_idx += 1

        if bottom_temps.size:
            last_row = new_data[out_idx - 1].copy()
            bottom_block, last_adjusted = self._build_top_or_bottom_block(last_row, bottom_temps, False)
            new_data[out_idx - 1] = last_adjusted
            count_bottom = bottom_block.shape[0]
            if count_bottom:
                new_data[out_idx : out_idx + count_bottom] = bottom_block
                out_idx += count_bottom
                inserted_total += count_bottom

        return new_data, inserted_total

    def _build_insert_block(
        self, row_top: np.ndarray, row_bot: np.ndarray, T_vals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct interpolated rows between existing bounds."""
        rows, indices = self._initialise_insert_rows(row_top, row_bot, T_vals)
        self._interpolate_heat_columns(rows, row_top, row_bot, T_vals, indices[0])
        bottom = self._adjust_bottom_row(row_bot, T_vals, indices)
        return rows, bottom

    def _build_top_or_bottom_block(
        self, row_neighbor: np.ndarray, T_vals: np.ndarray, is_top_block: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build rows to prepend above the current hottest interval."""
        n_cols = self.data.shape[1]
        if T_vals.size == 0:
            return np.empty((0, n_cols), dtype=self.data.dtype)

        col = self.col_index
        T_idx = col[PT.T.value]
        delta_T_idx = col[PT.DELTA_T.value]

        temps_sorted = np.sort(T_vals)
        block = np.full((temps_sorted.size, n_cols), np.nan, dtype=self.data.dtype)
        neighbor = row_neighbor.copy()

        for i, temp in enumerate(temps_sorted):
            row = block[i]
            row[T_idx] = temp
            row[delta_T_idx] = temp - neighbor[T_idx] if is_top_block else neighbor[T_idx] - temp
            for num in PT:
                key = num.value
                row_idx = col[key]
                if key in [PT.T.value, PT.DELTA_T.value]:
                    continue
                elif key in INTERPOLATION_KEYS or np.isnan(neighbor[row_idx]):
                    row[row_idx] = neighbor[row_idx]
                else:
                    row[row_idx] = 0.0
            neighbor = row

        if is_top_block:
            for i, temp in enumerate(block[:,delta_T_idx]):
                if i == 0:
                    row_neighbor[delta_T_idx] = temp
                elif i + 1 == block[:,0].size:
                    block[i-1][delta_T_idx] = block[i][delta_T_idx]
                    block[i][delta_T_idx] = 0.0
                else:
                    block[i-1][delta_T_idx] = block[i][delta_T_idx]

        return block[::-1], row_neighbor

    def _initialise_insert_rows(
        self, row_top: np.ndarray, row_bot: np.ndarray, T_vals: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Initialise new rows with temperature and delta values."""
        idx = self.col_index
        indices = [idx[key] for key in INTERVAL_CORE_KEYS]
        t_i = indices[0]
        deltas = row_top[t_i] - T_vals
        rows = np.full((T_vals.size, self.data.shape[1]), np.nan, dtype=self.data.dtype)
        rows[:, t_i] = T_vals
        rows[:, indices[1]] = deltas
        for cp_idx, dh_idx in (
            (indices[2], indices[3]),
            (indices[4], indices[5]),
            (indices[6], indices[7]),
        ):
            rows[:, cp_idx] = row_bot[cp_idx]; rows[:, dh_idx] = deltas * row_bot[cp_idx]
        return rows, tuple(indices)

    def _interpolate_heat_columns(
        self,
        rows: np.ndarray,
        row_top: np.ndarray,
        row_bot: np.ndarray,
        T_vals: np.ndarray,
        t_idx: int,
    ):
        """Fill heat-related columns using linear interpolation."""
        denom = row_top[t_idx] - row_bot[t_idx]
        if abs(denom) <= tol:
            return
        ratio = (T_vals - row_bot[t_idx]) / denom
        for key in INTERPOLATION_KEYS:
            col_idx = self.col_index[key]
            if np.isnan(row_bot[col_idx]):
                continue
            rows[:, col_idx] = row_bot[col_idx] + ratio * (row_top[col_idx] - row_bot[col_idx])

    def _adjust_bottom_row(
        self,
        row_bot: np.ndarray,
        T_vals: np.ndarray,
        indices: Tuple[int, ...],
    ) -> np.ndarray:
        """Update the existing lower row to account for inserted intervals."""
        t_i, dt_i, cph_i, dhh_i, cpc_i, dhc_i, mcp_i, dhn_i, _, _ = indices
        adjusted = row_bot.copy()
        delta = T_vals[-1] - row_bot[t_i]
        adjusted[dt_i] = delta
        for cp_idx, dh_idx in (
            (cph_i, dhh_i),
            (cpc_i, dhc_i),
            (mcp_i, dhn_i),
        ):
            adjusted[dh_idx] = delta * row_bot[cp_idx]
        return adjusted

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

    def update(
        self,
        updates: Mapping[Union[str, ProblemTableLabel], Sequence[float]],
    ) -> "ProblemTable":
        """Assign column values in-place using a mapping of ``column -> iterable``."""
        if not updates:
            return self
        if self.data is None:
            raise ValueError("Cannot update columns on an uninitialised ProblemTable.")

        for key, values in updates.items():
            col_name = key.value if isinstance(key, ProblemTableLabel) else key
            self.col[col_name] = values

        return self

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
