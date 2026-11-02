# Reporting

## Overview
Converts solved targets and tables into external forms: report contracts and JSON-style dictionaries, Excel workbooks, problem-table dataframes and files, stream CSV files and heat pump summaries. Result adapters are looked up by target class and return the typed reporting contracts in `contracts`. The dashboard and the application layer call these modules.

## Interface
- `serialize_target`, `target_to_result` (`results.py`): serialise a target to a plain dictionary (`Value` objects become dictionaries), or return its reporting contract.
- `adapt_analysis_result`, `RESULT_ADAPTERS` (`adapters.py`): choose the adapter for a target by walking its class hierarchy; `RESULT_ADAPTERS` is a read-only mapping.
- `THERMAL_RESULT_ADAPTERS`, `direct_result`, `indirect_result`, `hpr_result`, `orc_result` (`thermal.py`): build `TargetResults` for utility-summary, direct, indirect, heat pump or refrigeration, and ORC targets.
- `SPECIALIST_RESULT_ADAPTERS`, `specialist_result` (`specialist.py`): deep copies of heat recovery dt-min, HPR performance map, HEN synthesis and utility placement results.
- `export_target_summary_to_excel_with_units`, `build_summary_dataframe` (`workbook.py`): write the summary and problem-table workbook, or build the summary dataframe with value and unit columns.
- `problem_table_frame`, `export_problem_table` (`problem_table.py`): detached dataframe view of a `ProblemTable`; write one to an `.xlsx` file.
- `write_text_as_plain_strings`, `FORMULA_LEADING_CHARACTERS` (`excel_text.py`): store text cells as strings and quote-prefix formula-like text.
- `export_stream_collection` (`stream_collection.py`): write ordered streams as CSV.
- `hpr_summary`, `hpr_failure_summary` (`hpr.py`): plain-JSON summaries of a solved HPR target and of an `HPRTargetingError`.
- Subdirectories: none.

## Invariants
- Reporting does no numerical execution; a non-reportable target, or one with no registered family, raises `NotImplementedError`. [adapters.py:adapt_analysis_result]
- Scalar metrics pass through `coerce_output_value` with the target's configuration, so those with an output-unit rule appear in the configured units; per-machine and per-stage tuples, the HPR objective, the design vector and the scalar `orc_condensing_temperature` are passed through unchanged. [thermal.py:_metric] [thermal.py:orc_result]
- For a multiperiod target a utility's duty is reduced to the target's own period. [thermal.py:_period_duty]
- Summary rows are footer rows for totals and content rows otherwise. [thermal.py:_utility_summary_result]
- Text starting with `=`, `+`, `-`, `@`, tab or carriage return is stored as a string with a quote prefix. [excel_text.py:write_text_as_plain_strings]
- Workbook sheet names are at most 31 characters, free of forbidden characters and unique case-insensitively; problem-table values are rounded to 2 decimals. [workbook.py:_unique_sheet_name] [workbook.py:_write_problem_tables]
- A workbook is written to a scratch file and moved into place only when complete; on failure the scratch file, and a path this call reserved, are removed. [workbook.py:export_target_summary_to_excel_with_units]
- An `out_dir` ending in `.xlsx` is used as the file name; otherwise the file is named `<project>_<timestamp>.xlsx`, made unique atomically. [workbook.py:_reserve_workbook_path]
- Utility columns are prefixed `HU:` or `CU:` so they cannot overwrite other columns. [workbook.py:_utility_columns]
- `export_problem_table` refuses an uninitialised table. [problem_table.py:export_problem_table]

## Side Effects
- Writes `.xlsx` files and creates directories. [workbook.py:export_target_summary_to_excel_with_units] [problem_table.py:export_problem_table]
- `export_problem_table` and the Excel summary export default to the current working directory; `export_problem_table` keeps only the stem of `filename`. [problem_table.py:export_problem_table]
- `export_stream_collection` defaults to the working directory, as `export_problem_table` does; it writes a CSV, and `filename` is not sanitised. [stream_collection.py:export_stream_collection]
- `write_text_as_plain_strings` mutates the openpyxl workbook it is given. [excel_text.py:write_text_as_plain_strings]
- Uses the openpyxl engine through pandas. [workbook.py:export_target_summary_to_excel_with_units]

## Verification
- Run: `uv run --no-sync pytest tests/presentation/test_workbook_reporting.py tests/presentation/test_excel_text_cells.py tests/presentation/test_hpr_target_results.py`
