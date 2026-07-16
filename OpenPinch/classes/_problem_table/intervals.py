"""Temperature-interval insertion engine owned by ProblemTable."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import List, Tuple

import numpy as np

from ...lib.config import tol
from ...lib.enums import ProblemTableLabel
from .constants import HEAT_CAPACITY_PAIRS, INTERPOLATION_KEYS

PT = ProblemTableLabel


class TemperatureIntervalEngine:
    """Apply interval insertion operations to one parent table."""

    def __init__(self, table):
        self._table = table

    def __getattr__(self, name):
        return getattr(self._table, name)

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


def insert_temperature_intervals(table, temperatures) -> int:
    """Insert missing intervals through a dedicated table-owned engine."""
    if table.data is None or table.data.shape[0] < 2 == 0:
        return 0
    engine = TemperatureIntervalEngine(table)
    values = np.atleast_1d(np.asarray(temperatures, dtype=float))
    missing = engine._Ts_needing_insertion(values)
    top_temps, interval_map, bottom_temps = engine._categorise_insertion_targets(
        missing
    )
    if top_temps.size == 0 and bottom_temps.size == 0 and not interval_map:
        return 0
    new_data, inserted = engine._apply_interval_map(
        interval_map, top_temps, bottom_temps
    )
    table.data = new_data
    return inserted
