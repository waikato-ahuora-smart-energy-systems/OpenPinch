"""Lightweight table structure used by the pinch analysis pipeline."""

from __future__ import annotations

import numbers
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from ..lib.config import tol
from ..lib.enums import ProblemTableLabel
from ..lib.problem_table_types import ProblemTableColumnUpdates
from ._problem_table._equality import (
    problem_tables_equal,
)
from ._problem_table._problem_table_constants import (
    HEAT_CAPACITY_PAIRS,
    INTERPOLATION_KEYS,
)

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
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the buffer into a pandas DataFrame."""
        return pd.DataFrame(self.data.copy(), columns=self.columns)

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
        if self.data is None or self.data.shape[0] < 2 == 0:
            return 0
        T_vals = np.atleast_1d(np.asarray(T_ls, dtype=float))
        # Get unique missing values
        T_insert = self._Ts_needing_insertion(T_vals)
        # S
        top_temps, interval_map, bottom_temps = self._categorise_insertion_targets(
            T_insert
        )
        if top_temps.size == 0 and bottom_temps.size == 0 and not interval_map:
            return 0

        new_data, inserted = self._apply_interval_map(
            interval_map, top_temps, bottom_temps
        )
        self.data = new_data
        return inserted

    def _Ts_needing_insertion(self, T_vals: np.ndarray) -> np.ndarray:
        """Filter temperatures that are not within tolerance of existing rows."""
        if T_vals.size == 0:
            return T_vals
        T_col = self.data[:, self._column_index_for(PT.T)]
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
        T_col = data[:, self._column_index_for(PT.T)]
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
        """Insert grouped temperatures and return the expanded array."""
        n_rows, n_cols = self.data.shape
        total_new = (
            top_temps.size
            + bottom_temps.size
            + sum(len(vals) for vals in interval_map.values())
        )

        if total_new == 0:
            return self.data, 0

        # 1) Create the expanded buffer sized for original + inserted rows.
        new_data = np.zeros((n_rows + total_new, n_cols), dtype=self.data.dtype)
        row_meta: List[dict] = [{} for _ in range(new_data.shape[0])]

        # 2) Copy existing rows into the new buffer (top to bottom).
        new_data[:n_rows] = self.data
        for idx in range(n_rows):
            row_meta[idx] = {"type": "orig", "orig_idx": idx}

        # 3) Append placeholder rows for every new temperature (only T populated).
        T_idx = self._column_index_for(PT.T)
        next_row = n_rows
        next_row = self._append_placeholders(
            new_data, row_meta, next_row, top_temps, label="top"
        )
        for lower_idx, temps in interval_map.items():
            next_row = self._append_placeholders(
                new_data,
                row_meta,
                next_row,
                temps,
                label="mid",
                lower_idx=lower_idx,
            )
        self._append_placeholders(
            new_data, row_meta, next_row, bottom_temps, label="bottom"
        )

        # 4) Sort rows by descending temperature so indices align with final order.
        order = np.argsort(new_data[:, T_idx])[::-1]
        new_data = new_data[order]
        row_meta = [row_meta[idx] for idx in order]

        # 5) Rebuild top and bottom placeholder rows using existing helper.
        inserted_total = total_new

        top_positions = [
            idx for idx, meta in enumerate(row_meta) if meta.get("type") == "top"
        ]
        self._rebuild_edge_block(new_data, row_meta, top_positions, is_top=True)

        bottom_positions = [
            idx for idx, meta in enumerate(row_meta) if meta.get("type") == "bottom"
        ]
        self._rebuild_edge_block(new_data, row_meta, bottom_positions, is_top=False)

        # 6) Build and insert middle blocks, adjusting adjacent rows along the way.
        orig_positions = {
            meta["orig_idx"]: idx
            for idx, meta in enumerate(row_meta)
            if meta.get("type") == "orig"
        }
        mid_positions: dict[int, List[int]] = defaultdict(list)
        for idx, meta in enumerate(row_meta):
            if meta.get("type") == "mid":
                mid_positions[meta["lower_idx"]].append(idx)

        for lower_idx, positions in sorted(mid_positions.items()):
            upper_pos = orig_positions.get(lower_idx - 1)
            lower_pos = orig_positions.get(lower_idx)
            self._insert_mid_block(new_data, positions, upper_pos, lower_pos)

        return new_data, inserted_total

    def _build_mid_block(
        self, row_top: np.ndarray, row_bot: np.ndarray, T_vals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct interpolated rows and return updated neighbour rows."""
        rows, (t_idx, delta_idx) = self._initialise_insert_rows(
            row_top, row_bot, T_vals
        )
        top_adjusted = row_top.copy()
        bottom_adjusted = row_bot.copy()

        if rows.size == 0:
            return rows, top_adjusted, bottom_adjusted

        self._interpolate_heat_columns(rows, row_top, row_bot, t_idx)
        temps_chain = np.concatenate(
            ([row_top[t_idx]], rows[:, t_idx], [row_bot[t_idx]])
        )
        for i in range(rows.shape[0]):
            rows[i, delta_idx] = temps_chain[i + 1] - temps_chain[i + 2]
        top_adjusted[delta_idx] = temps_chain[0] - temps_chain[1]
        bottom_adjusted = self._adjust_bottom_row(row_bot, rows, t_idx, delta_idx)
        self._update_heat_capacity_pairs(rows, top_adjusted, bottom_adjusted, delta_idx)
        return rows, top_adjusted, bottom_adjusted

    def _rebuild_edge_block(
        self,
        new_data: np.ndarray,
        row_meta: List[dict],
        positions: List[int],
        *,
        is_top: bool,
    ) -> None:
        """Regenerate top or bottom placeholder rows after temperature reordering."""
        if not positions:
            return

        edge_type = "top" if is_top else "bottom"
        T_idx = self._column_index_for(PT.T)

        if is_top:
            neighbor_idx = next(
                (
                    idx
                    for idx, meta in enumerate(row_meta)
                    if meta.get("type") != edge_type
                ),
                None,
            )
        else:
            neighbor_idx = next(
                (
                    idx
                    for idx in range(len(row_meta) - 1, -1, -1)
                    if row_meta[idx].get("type") != edge_type
                ),
                None,
            )

        if neighbor_idx is None:
            return

        neighbor_row = new_data[neighbor_idx].copy()
        sorted_positions = sorted(positions)
        temps = new_data[np.asarray(sorted_positions), T_idx]

        block, neighbor_adjusted = self._build_top_or_bottom_block(
            neighbor_row, temps, is_top_block=is_top
        )
        for idx_pos, row_vals in zip(sorted_positions, block):
            new_data[idx_pos] = row_vals
        new_data[neighbor_idx] = neighbor_adjusted

    def _insert_mid_block(
        self,
        new_data: np.ndarray,
        positions: List[int],
        upper_pos: int | None,
        lower_pos: int | None,
    ) -> None:
        """Populate middle placeholder rows between existing neighbours."""
        if not positions or upper_pos is None or lower_pos is None:
            return

        positions = sorted(positions)
        if upper_pos < 0 or lower_pos < 0:
            return
        T_idx = self._column_index_for(PT.T)
        temps_mid = new_data[np.asarray(positions), T_idx]

        block, top_adjusted, bottom_adjusted = self._build_mid_block(
            new_data[upper_pos].copy(), new_data[lower_pos].copy(), temps_mid
        )

        new_data[upper_pos] = top_adjusted
        for idx_pos, row_vals in zip(positions, block):
            new_data[idx_pos] = row_vals
        new_data[lower_pos] = bottom_adjusted

    def _append_placeholders(
        self,
        new_data: np.ndarray,
        row_meta: List[dict],
        start_idx: int,
        temps: Sequence[float],
        *,
        label: str,
        lower_idx: int | None = None,
    ) -> int:
        """Append placeholder rows for a given temperature sequence."""
        temps_arr = np.asarray(temps, dtype=float)
        if temps_arr.size == 0:
            return start_idx
        T_idx = self._column_index_for(PT.T)
        next_row = start_idx
        for temp in temps_arr:
            new_data[next_row, T_idx] = float(temp)
            meta = {"type": label}
            if lower_idx is not None:
                meta["lower_idx"] = lower_idx
            row_meta[next_row] = meta
            next_row += 1
        return next_row

    def _populate_from_neighbor(
        self,
        target_row: np.ndarray,
        neighbor_row: np.ndarray,
        *,
        copy_interpolation: bool,
        zero_non_interpolation: bool,
        relevant_cp_source: np.ndarray | None = None,
    ) -> None:
        """Populate non-interpolated columns based on a neighbouring row."""
        t_col = self._validate_column_name(PT.T)
        delta_t_col = self._validate_column_name(PT.DELTA_T)
        for key in self.columns:
            if key in (t_col, delta_t_col):
                continue
            col_idx = self.col_index[key]
            if key in INTERPOLATION_KEYS and not copy_interpolation:
                continue
            value = neighbor_row[col_idx]
            if key in INTERPOLATION_KEYS:
                if copy_interpolation:
                    target_row[col_idx] = value
                continue
            if relevant_cp_source is not None:
                for cp_key, dh_key in HEAT_CAPACITY_PAIRS:
                    if key == cp_key:
                        cp_idx = self.col_index[cp_key]
                        target_row[cp_idx] = relevant_cp_source[cp_idx]
                        break
            if zero_non_interpolation:
                target_row[col_idx] = 0.0 if not np.isnan(value) else np.nan
            else:
                target_row[col_idx] = value

    def _build_top_or_bottom_block(
        self, row_neighbor: np.ndarray, T_vals: np.ndarray, is_top_block: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build rows to prepend above the current hottest interval."""
        n_cols = self.data.shape[1]
        if T_vals.size == 0:
            return np.empty((0, n_cols), dtype=self.data.dtype), row_neighbor.copy()

        T_idx = self._column_index_for(PT.T)
        delta_T_idx = self._column_index_for(PT.DELTA_T)

        temps_sorted = np.sort(T_vals)
        block = np.full((temps_sorted.size, n_cols), np.nan, dtype=self.data.dtype)
        neighbor = row_neighbor.copy()

        for i, temp in enumerate(temps_sorted):
            row = block[i]
            row[T_idx] = temp
            row[delta_T_idx] = (
                temp - neighbor[T_idx] if is_top_block else neighbor[T_idx] - temp
            )
            self._populate_from_neighbor(
                row,
                neighbor,
                copy_interpolation=True,
                zero_non_interpolation=True,
            )
            neighbor = row

        if is_top_block:
            for i, temp in enumerate(block[:, delta_T_idx]):
                if i == 0:
                    row_neighbor[delta_T_idx] = temp
                elif i + 1 == block[:, 0].size:
                    block[i - 1][delta_T_idx] = block[i][delta_T_idx]
                    block[i][delta_T_idx] = 0.0
                else:
                    block[i - 1][delta_T_idx] = block[i][delta_T_idx]

        return block[::-1], row_neighbor

    def _initialise_insert_rows(
        self, row_top: np.ndarray, row_bot: np.ndarray, T_vals: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Initialise new rows with temperature and delta values."""
        n_rows = T_vals.size
        n_cols = self.data.shape[1]
        rows = np.full((n_rows, n_cols), np.nan, dtype=self.data.dtype)
        if n_rows == 0:
            t_idx = self._column_index_for(PT.T)
            delta_idx = self._column_index_for(PT.DELTA_T)
            return rows, (t_idx, delta_idx)

        temps_sorted = np.sort(T_vals)[::-1]
        t_idx = self._column_index_for(PT.T)
        delta_idx = self._column_index_for(PT.DELTA_T)

        for i, temp in enumerate(temps_sorted):
            row = rows[i]
            row[t_idx] = temp
            self._populate_from_neighbor(
                row,
                row_bot,
                copy_interpolation=False,
                zero_non_interpolation=False,
                relevant_cp_source=row_bot,
            )

        return rows, (t_idx, delta_idx)

    def _interpolate_heat_columns(
        self,
        rows: np.ndarray,
        row_top: np.ndarray,
        row_bot: np.ndarray,
        t_idx: int,
    ):
        """Fill heat-related columns using linear interpolation."""
        if rows.size == 0:
            return

        temps = rows[:, t_idx]
        denom = row_top[t_idx] - row_bot[t_idx]
        if abs(denom) <= tol:
            for key in INTERPOLATION_KEYS:
                col_idx = self.col_index[key]
                rows[:, col_idx] = row_bot[col_idx]
            return

        ratio = (temps - row_bot[t_idx]) / denom
        for key in INTERPOLATION_KEYS:
            col_idx = self.col_index[key]
            bot_val = row_bot[col_idx]
            top_val = row_top[col_idx]
            if np.isnan(bot_val):
                continue
            if np.isnan(top_val):
                rows[:, col_idx] = bot_val
                continue
            rows[:, col_idx] = bot_val + ratio * (top_val - bot_val)

    def _adjust_bottom_row(
        self,
        row_bot: np.ndarray,
        rows: np.ndarray,
        t_idx: int,
        delta_idx: int,
    ) -> np.ndarray:
        """Update the existing lower row to account for inserted intervals."""
        adjusted = row_bot.copy()
        if rows.size == 0:
            return adjusted
        last_temp = rows[-1, t_idx]
        adjusted[delta_idx] = last_temp - adjusted[t_idx]
        for cp_key, dh_key in HEAT_CAPACITY_PAIRS:
            cp_idx = self.col_index[cp_key]
            dh_idx = self.col_index[dh_key]
            adjusted[dh_idx] = adjusted[delta_idx] * adjusted[cp_idx]
        return adjusted

    def _update_heat_capacity_pairs(
        self,
        rows: np.ndarray,
        top_adjusted: np.ndarray,
        bottom_adjusted: np.ndarray,
        delta_idx: int,
    ) -> None:
        """Recalculate delta-H columns using heat capacities and delta-T."""
        for cp_key, dh_key in HEAT_CAPACITY_PAIRS:
            cp_idx = self.col_index[cp_key]
            dh_idx = self.col_index[dh_key]
            if rows.size:
                rows[:, dh_idx] = rows[:, delta_idx] * rows[:, cp_idx]
            top_adjusted[dh_idx] = top_adjusted[delta_idx] * top_adjusted[cp_idx]
            bottom_adjusted[dh_idx] = (
                bottom_adjusted[delta_idx] * bottom_adjusted[cp_idx]
            )

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

    def export(
        self,
        filename: str = "problem_table",
        sheet_name: str = "ProblemTable",
        include_index: bool = False,
    ) -> Path:
        """Export the table to ``results/<filename>.xlsx`` and return the path."""
        if self.data is None:
            raise ValueError("Cannot export an uninitialised ProblemTable.")

        results_dir = Path(__file__).resolve().parents[2] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        name = Path(filename).stem or "problem_table"
        output_path = results_dir / f"{name}.xlsx"

        df = self.to_dataframe
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=include_index)

        return output_path
