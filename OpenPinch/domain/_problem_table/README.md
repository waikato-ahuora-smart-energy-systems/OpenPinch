# Problem Table Helpers

## Overview
Private helpers behind `OpenPinch.domain.problem_table.ProblemTable`: the tolerance-aware equality functions, the temperature-interval insertion engine and the typed update keyword arguments. `ProblemTable` calls them with itself as the table; the analysis package imports the update types and the canonical temperature grid.

## Interface
- `problem_tables_equal` (`equality.py`): equality of two ProblemTable-like objects by column labels, shape and cell values within an absolute tolerance; helper functions in the same file handle object-typed columns and scalars.
- `insert_temperature_intervals` (`intervals.py`): insert missing temperatures into a table and return the number of rows added.
- `canonical_temperature_values` (`intervals.py`): descending, rounded, de-duplicated temperature grid using the domain tolerance.
- `temperature_grids_match` (`intervals.py`): whether two grids agree element-wise within tolerance.
- `TemperatureIntervalEngine` (`intervals.py`): applies top, middle and bottom row insertion to one parent table, interpolating heat columns.
- `INTERPOLATION_KEYS`, `HEAT_CAPACITY_PAIRS` (`constants.py`): column labels that are interpolated, and the (CP, delta-H) column pairs refreshed after insertion.
- `ProblemTableColumnKey`, `ProblemTableColumnUpdates`, `ProblemTableUpdateKwargs` (`types.py`): typing for `ProblemTable.update` arguments (`T_col`, `updates`).

## Invariants
- Temperature differences are compared after rounding to six more decimal places than the domain tolerance `tol`, and a gap counts only when it exceeds `tol`. [intervals.py:_temperature_gap_exceeds_tolerance]
- The canonical grid is finite values only, rounded to the tolerance decimals, in descending order. [intervals.py:canonical_temperature_values]
- Table rows run from the hottest to the coldest temperature; the interval width in row `i` is `T[i-1] - T[i]` and the hottest row has width zero. [intervals.py:TemperatureIntervalEngine._refresh_interval_widths]
- Rows inserted above the hottest or below the coldest row copy the interpolated columns of the neighbouring row and zero all other columns (NaN stays NaN). [intervals.py:TemperatureIntervalEngine._build_top_or_bottom_block]
- Rows inserted between two rows linearly interpolate the `INTERPOLATION_KEYS` columns by temperature; a NaN in the lower row leaves the column NaN; non-interpolated columns copy the lower row. [intervals.py:TemperatureIntervalEngine._interpolate_heat_columns]
- Table equality treats NaN equal to NaN, uses `rtol=0` and an absolute tolerance on numeric columns, and requires identical column label order; an object column with mixed cells falls back to `scalar_values_equal`, which keeps NumPy's default relative tolerance. [equality.py:numeric_arrays_equal] [equality.py:problem_tables_equal] [equality.py:scalar_values_equal]
- `ProblemTable.update` rejects the temperature column in `updates`. [../problem_table.py:ProblemTable._validate_updates]

## Side Effects
- `insert_temperature_intervals` replaces `table.data` on the table passed in. [intervals.py:insert_temperature_intervals]
- Imports `pandas` (for `isna`) and `numpy`. [equality.py:scalar_values_equal]

## Verification
- Run: `uv run --no-sync pytest tests/domain/test_problem_table_kernels.py`
