"""Temperature-interval insertion orchestration for a parent problem table."""

from __future__ import annotations

import numpy as np


def insert_temperature_intervals(table, temperatures) -> int:
    """Insert missing intervals through the table's validated interval engine."""
    if table.data is None or table.data.shape[0] < 2 == 0:
        return 0
    values = np.atleast_1d(np.asarray(temperatures, dtype=float))
    missing = table._Ts_needing_insertion(values)
    top_temps, interval_map, bottom_temps = table._categorise_insertion_targets(missing)
    if top_temps.size == 0 and bottom_temps.size == 0 and not interval_map:
        return 0
    new_data, inserted = table._apply_interval_map(
        interval_map,
        top_temps,
        bottom_temps,
    )
    table.data = new_data
    return inserted
